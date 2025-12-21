# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose: bool = False):
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : Namespace
        Arguments containing input folder, output folder, and processing options.
    config : Dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    def _normalize_instrument_name(name: str) -> str:
        name = str(name).strip().lower()
        if name in ('voice', 'vocal'):
            return 'vocals'
        return name

    instruments = prefer_target_instrument(config)[:]
    model_outputs = instruments[:]  # stems the model actually predicts (before adding derived residuals)
    # Ensure both vocals and other are present when available in the model config
    if 'vocals' in getattr(config.training, 'instruments', []) and 'vocals' not in instruments:
        instruments.insert(0, 'vocals')
    if 'other' in getattr(config.training, 'instruments', []) and 'other' not in instruments:
        instruments.append('other')

    write_instruments = None
    if getattr(args, 'output_instruments', None):
        write_instruments = []
        for instr_name in args.output_instruments:
            norm = _normalize_instrument_name(instr_name)
            if norm and norm not in write_instruments:
                write_instruments.append(norm)
    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
    subtype = args.pcm_type

    def _is_nonempty_file(path: str) -> bool:
        try:
            return os.path.isfile(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    def _build_output_paths(instr: str, file_name: str, src_path: str):
        instr_label = 'vocal' if instr == 'vocals' else instr
        dirnames, fname = format_filename(
            args.filename_template,
            instr=instr_label,
            start_time=int(start_time),
            file_name=file_name,
            dir_name=os.path.dirname(src_path),
            model_type=args.model_type,
            model=os.path.splitext(os.path.basename(args.start_check_point))[0],
        )

        if dirnames:
            output_dir = os.path.join(args.store_dir, *dirnames)
        else:
            output_dir = args.store_dir or '.'

        output_path = os.path.join(output_dir, f"{fname}.{codec}")
        output_img_path = os.path.join(output_dir, f"{fname}.jpg") if args.draw_spectro > 0 else None
        return output_dir, output_path, output_img_path

    for path in mixture_paths:
        print(f"Processing track: {path}")
        file_name = os.path.splitext(os.path.basename(path))[0]

        if getattr(args, 'skip_existing', False):
            if write_instruments is not None:
                expected_instruments = write_instruments[:]
            else:
                expected_instruments = instruments[:]
                if args.extract_instrumental and 'instrumental' not in expected_instruments:
                    expected_instruments.append('instrumental')
                # If a model only outputs vocals, "other" will be derived as a residual
                if model_outputs == ['vocals'] and 'other' not in expected_instruments:
                    expected_instruments.append('other')

            all_outputs_exist = True
            for instr in expected_instruments:
                _, output_path, output_img_path = _build_output_paths(instr, file_name, path)
                if not _is_nonempty_file(output_path):
                    all_outputs_exist = False
                    break
                if args.draw_spectro > 0 and output_img_path and not _is_nonempty_file(output_img_path):
                    all_outputs_exist = False
                    break

            if all_outputs_exist:
                print(f"Skipping track (all outputs exist): {path}")
                continue

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        # Use the same mix scale as the model output for any residual computations.
        # (If normalization is enabled, both `mix` and `waveforms_orig` are in the normalized domain.)
        mix_for_residual = mix

        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        instruments_to_write = write_instruments if write_instruments is not None else instruments
        requested_set = set(instruments_to_write)

        # Derive missing stems as residuals when requested.
        # If the model produced a single non-vocal stem (e.g. an "instrument" target),
        # derive vocals as the residual.
        if ('vocals' in requested_set or 'instrumental' in requested_set) and 'vocals' not in waveforms_orig:
            ref_instr = None
            target_instr = getattr(config.training, 'target_instrument', None)
            if target_instr and target_instr in waveforms_orig:
                ref_instr = target_instr
            elif len(waveforms_orig) == 1:
                ref_instr = next(iter(waveforms_orig.keys()))
            if ref_instr is not None:
                waveforms_orig['vocals'] = mix_for_residual - waveforms_orig[ref_instr]

        # If the model only produced vocals, allow "instrument" to be requested as a residual stem.
        if 'instrument' in requested_set and 'instrument' not in waveforms_orig and 'vocals' in waveforms_orig:
            waveforms_orig['instrument'] = mix_for_residual - waveforms_orig['vocals']

        # If the model only produced vocals, derive "other" as the residual
        if model_outputs == ['vocals'] and 'vocals' in waveforms_orig and 'other' not in waveforms_orig:
            if write_instruments is None or 'other' in requested_set:
                waveforms_orig['other'] = mix_for_residual - waveforms_orig['vocals']
                if write_instruments is None and 'other' not in instruments:
                    instruments.append('other')

        if args.extract_instrumental or 'instrumental' in requested_set:
            ref_instr = None
            if 'vocals' in waveforms_orig:
                ref_instr = 'vocals'
            else:
                target_instr = getattr(config.training, 'target_instrument', None)
                if target_instr and target_instr in waveforms_orig:
                    ref_instr = target_instr
                elif len(waveforms_orig) > 0:
                    ref_instr = next(iter(waveforms_orig.keys()))
            if ref_instr is not None:
                waveforms_orig['instrumental'] = mix_for_residual - waveforms_orig[ref_instr]
                if write_instruments is None and args.extract_instrumental and 'instrumental' not in instruments:
                    instruments.append('instrumental')

        instruments_to_write = write_instruments if write_instruments is not None else instruments
        missing = [instr for instr in instruments_to_write if instr not in waveforms_orig]
        if missing:
            print(
                f"Cannot produce requested instruments {missing}. "
                f"Available: {sorted(list(waveforms_orig.keys()))}"
            )
            continue

        for instr in instruments_to_write:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)
            output_dir, output_path, output_img_path = _build_output_paths(instr, file_name, path)
            os.makedirs(output_dir, exist_ok=True)

            if getattr(args, 'skip_existing', False) and _is_nonempty_file(output_path):
                print("Skip existing file:", output_path)
            else:
                sf.write(output_path, estimates.T, sr, subtype=subtype)
                print("Wrote file:", output_path)

            if args.draw_spectro > 0 and output_img_path:
                if getattr(args, 'skip_existing', False) and _is_nonempty_file(output_img_path):
                    print("Skip existing file:", output_img_path)
                else:
                    draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                    print("Wrote file:", output_img_path)
        

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
