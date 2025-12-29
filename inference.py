# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import math
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

    max_split_seconds = 20 * 60

    split_overlap_seconds = 1.0

    def _get_audio_info(path: str):
        try:
            return sf.info(path)
        except Exception:
            return None

    def _safe_audio_duration_seconds(path: str):
        info = _get_audio_info(path)
        if info is not None and info.frames > 0 and info.samplerate > 0:
            return info.frames / info.samplerate
        try:
            return librosa.get_duration(filename=path)
        except Exception:
            return None

    def _load_audio_segment(path: str, offset_sec: float, duration_sec: float):
        return librosa.load(path, sr=sample_rate, mono=False, offset=offset_sec, duration=duration_sec)

    def _read_audio_frames(sf_desc: sf.SoundFile, start_frame: int, num_frames: int) -> np.ndarray:
        sf_desc.seek(start_frame)
        data = sf_desc.read(num_frames, dtype='float32', always_2d=True)
        return data.T

    def _resample_if_needed(mix_arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return mix_arr
        resampled = []
        for ch in mix_arr:
            resampled.append(
                librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_best')
            )
        return np.stack(resampled, axis=0).astype(np.float32)

    def _prepare_mix(mix_arr: np.ndarray, announce: bool = True):
        # If mono audio we must adjust it depending on model
        if len(mix_arr.shape) == 1:
            mix_arr = np.expand_dims(mix_arr, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    if announce:
                        print('Convert mono track to stereo...')
                    mix_arr = np.concatenate([mix_arr, mix_arr], axis=0)
        return mix_arr

    def _compute_norm_params_long(sf_desc: sf.SoundFile, total_frames: int, src_sr: int, target_sr: int):
        total = 0
        mean = 0.0
        m2 = 0.0
        block_frames = int(round(10.0 * src_sr))
        if block_frames <= 0:
            return None
        for start in range(0, total_frames, block_frames):
            frames = min(block_frames, total_frames - start)
            if frames <= 0:
                continue
            try:
                mix_seg = _read_audio_frames(sf_desc, start, frames)
            except Exception as e:
                print(f'Cannot read track segment for normalization: {path}')
                print(f'Error message: {str(e)}')
                return None
            mix_seg = _resample_if_needed(mix_seg, src_sr, target_sr)
            mix_seg = _prepare_mix(mix_seg, announce=False)
            mono = mix_seg.mean(0).astype(np.float64)
            if mono.size == 0:
                continue
            seg_count = mono.size
            seg_mean = float(mono.mean())
            seg_var = float(mono.var())
            if total == 0:
                mean = seg_mean
                m2 = seg_var * seg_count
                total = seg_count
                continue
            delta = seg_mean - mean
            new_total = total + seg_count
            mean = mean + delta * seg_count / new_total
            m2 = m2 + seg_var * seg_count + (delta ** 2) * total * seg_count / new_total
            total = new_total
        if total == 0:
            return None
        std = math.sqrt(m2 / total) if total > 0 else 0.0
        if std == 0.0:
            std = 1.0
        return {"mean": mean, "std": std}

    def _apply_normalization(mix_arr: np.ndarray, norm_params: dict):
        if norm_params is None:
            return normalize_audio(mix_arr)
        return (mix_arr - norm_params["mean"]) / norm_params["std"], norm_params

    def _process_mix_segment(mix_arr: np.ndarray, norm_params: dict):
        if 'normalize' in config.inference and config.inference['normalize'] is True:
            mix_arr, norm_params = _apply_normalization(mix_arr, norm_params)

        mix_for_residual = mix_arr
        waveforms = demix(config, model, mix_arr, device, model_type=args.model_type, pbar=detailed_pbar)

        if args.use_tta:
            waveforms = apply_tta(config, model, mix_arr, waveforms, device, args.model_type)

        instruments_to_write_local = write_instruments if write_instruments is not None else instruments
        requested_set_local = set(instruments_to_write_local)

        # Derive missing stems as residuals when requested.
        # If the model produced a single non-vocal stem (e.g. an "instrument" target),
        # derive vocals as the residual.
        if ('vocals' in requested_set_local or 'instrumental' in requested_set_local) and 'vocals' not in waveforms:
            ref_instr = None
            target_instr = getattr(config.training, 'target_instrument', None)
            if target_instr and target_instr in waveforms:
                ref_instr = target_instr
            elif len(waveforms) == 1:
                ref_instr = next(iter(waveforms.keys()))
            if ref_instr is not None:
                waveforms['vocals'] = mix_for_residual - waveforms[ref_instr]

        # If the model only produced vocals, allow "instrument" to be requested as a residual stem.
        if 'instrument' in requested_set_local and 'instrument' not in waveforms and 'vocals' in waveforms:
            waveforms['instrument'] = mix_for_residual - waveforms['vocals']

        # If the model only produced vocals, derive "other" as the residual
        if model_outputs == ['vocals'] and 'vocals' in waveforms and 'other' not in waveforms:
            if write_instruments is None or 'other' in requested_set_local:
                waveforms['other'] = mix_for_residual - waveforms['vocals']
                if write_instruments is None and 'other' not in instruments:
                    instruments.append('other')

        if args.extract_instrumental or 'instrumental' in requested_set_local:
            ref_instr = None
            if 'vocals' in waveforms:
                ref_instr = 'vocals'
            else:
                target_instr = getattr(config.training, 'target_instrument', None)
                if target_instr and target_instr in waveforms:
                    ref_instr = target_instr
                elif len(waveforms) > 0:
                    ref_instr = next(iter(waveforms.keys()))
            if ref_instr is not None:
                waveforms['instrumental'] = mix_for_residual - waveforms[ref_instr]
                if write_instruments is None and args.extract_instrumental and 'instrumental' not in instruments:
                    instruments.append('instrumental')

        instruments_to_write_local = write_instruments if write_instruments is not None else instruments
        missing = [instr for instr in instruments_to_write_local if instr not in waveforms]
        if missing:
            print(
                f"Cannot produce requested instruments {missing}. "
                f"Available: {sorted(list(waveforms.keys()))}"
            )
            return None, None, norm_params

        return waveforms, instruments_to_write_local, norm_params

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

        info = _get_audio_info(path)
        if info is not None and info.frames > 0 and info.samplerate > 0:
            duration_sec = info.frames / info.samplerate
        else:
            duration_sec = _safe_audio_duration_seconds(path)

        long_file = duration_sec is not None and duration_sec > max_split_seconds

        if not long_file:
            try:
                mix, sr = librosa.load(path, sr=sample_rate, mono=False)
            except Exception as e:
                print(f'Cannot read track: {format(path)}')
                print(f'Error message: {str(e)}')
                continue

        if long_file and info is not None and info.frames > 0 and info.samplerate > 0:
            file_sr = info.samplerate
            total_frames = info.frames
            if verbose:
                print(f"Long file detected ({duration_sec / 60:.2f} min). Split into {max_split_seconds / 60:.0f} min parts.")
            segment_frames_src = int(round(max_split_seconds * file_sr))
            if segment_frames_src <= 0:
                print("Invalid segment size, fallback to non-splitting.")
                long_file = False
            else:
                overlap_frames_src = int(round(split_overlap_seconds * file_sr))
                if overlap_frames_src > 0:
                    overlap_frames_src = min(overlap_frames_src, segment_frames_src // 2)
                overlap_tgt_frames = int(round((overlap_frames_src / file_sr) * sample_rate))

                if 'normalize' in config.inference and config.inference['normalize'] is True:
                    with sf.SoundFile(path, mode='r') as sf_desc:
                        norm_params = _compute_norm_params_long(sf_desc, total_frames, file_sr, sample_rate)
                else:
                    norm_params = None

                num_segments = int(math.ceil(total_frames / segment_frames_src))
                writers = {}
                audio_skip = {}
                img_skip = {}
                spectro_buffers = {}
                output_targets = {}
                instruments_to_write = []
                prev_tails = {}
                try:
                    with sf.SoundFile(path, mode='r') as sf_desc:
                        for seg_idx in range(num_segments):
                            start_src = seg_idx * segment_frames_src
                            end_src = min(start_src + segment_frames_src, total_frames)
                            if end_src <= start_src:
                                continue
                            read_start_src = start_src if seg_idx == 0 else max(0, start_src - overlap_frames_src)
                            read_frames_src = end_src - read_start_src
                            if read_frames_src <= 0:
                                continue
                            try:
                                mix = _read_audio_frames(sf_desc, read_start_src, read_frames_src)
                            except Exception as e:
                                print(f'Cannot read track: {format(path)}')
                                print(f'Error message: {str(e)}')
                                break
                            if mix.size == 0:
                                continue
                            mix = _resample_if_needed(mix, file_sr, sample_rate)
                            mix = _prepare_mix(mix, announce=(seg_idx == 0))

                            lead_overlap_src = start_src - read_start_src
                            lead_overlap_tgt = int(round((lead_overlap_src / file_sr) * sample_rate))
                            segment_tgt_frames = int(round(((end_src - start_src) / file_sr) * sample_rate))
                            expected_len = lead_overlap_tgt + segment_tgt_frames
                            if expected_len > 0:
                                if mix.shape[1] < expected_len:
                                    pad = expected_len - mix.shape[1]
                                    mix = np.pad(mix, ((0, 0), (0, pad)), mode='constant')
                                elif mix.shape[1] > expected_len:
                                    mix = mix[:, :expected_len]

                            waveforms_orig, instruments_to_write, norm_params = _process_mix_segment(mix, norm_params)
                            if waveforms_orig is None:
                                break

                            # Initialize output targets and skip flags on the first successful segment
                            if not output_targets:
                                for instr in instruments_to_write:
                                    output_dir, output_path, output_img_path = _build_output_paths(instr, file_name, path)
                                    output_targets[instr] = (output_dir, output_path, output_img_path)
                                    if args.skip_existing and _is_nonempty_file(output_path):
                                        audio_skip[instr] = True
                                        print("Skip existing file:", output_path)
                                    else:
                                        audio_skip[instr] = False
                                    if args.draw_spectro > 0 and output_img_path:
                                        img_skip[instr] = args.skip_existing and _is_nonempty_file(output_img_path)
                                        if img_skip[instr]:
                                            print("Skip existing file:", output_img_path)
                                    else:
                                        img_skip[instr] = True
                                    spectro_buffers[instr] = []

                            for instr in instruments_to_write:
                                estimates = waveforms_orig[instr]
                                if 'normalize' in config.inference and config.inference['normalize'] is True:
                                    estimates = denormalize_audio(estimates, norm_params)

                                if expected_len > 0:
                                    if estimates.shape[1] < expected_len:
                                        pad = expected_len - estimates.shape[1]
                                        estimates = np.pad(estimates, ((0, 0), (0, pad)), mode='constant')
                                    elif estimates.shape[1] > expected_len:
                                        estimates = estimates[:, :expected_len]

                                head = estimates[:, :lead_overlap_tgt] if lead_overlap_tgt > 0 else None
                                body = estimates[:, lead_overlap_tgt:lead_overlap_tgt + segment_tgt_frames]
                                if body.shape[1] < segment_tgt_frames:
                                    pad = segment_tgt_frames - body.shape[1]
                                    body = np.pad(body, ((0, 0), (0, pad)), mode='constant')

                                output_dir, output_path, output_img_path = output_targets[instr]
                                if not audio_skip.get(instr, False):
                                    os.makedirs(output_dir, exist_ok=True)
                                    if instr not in writers:
                                        sf_format = codec.upper()
                                        writers[instr] = sf.SoundFile(
                                            output_path,
                                            mode='w',
                                            samplerate=sample_rate,
                                            channels=estimates.shape[0],
                                            subtype=subtype,
                                            format=sf_format,
                                        )

                                    if overlap_tgt_frames > 0 and seg_idx > 0 and prev_tails.get(instr) is not None:
                                        prev_tail = prev_tails[instr]
                                        head_len = head.shape[1] if head is not None else 0
                                        fade_len = min(overlap_tgt_frames, head_len, prev_tail.shape[1])
                                        if fade_len > 0:
                                            fade = np.linspace(0.0, 1.0, fade_len, dtype=estimates.dtype)
                                            cross = prev_tail[:, -fade_len:] * (1.0 - fade) + head[:, :fade_len] * fade
                                            writers[instr].write(cross.T)

                                    if overlap_tgt_frames > 0 and seg_idx < num_segments - 1 and body.shape[1] > overlap_tgt_frames:
                                        writers[instr].write(body[:, :body.shape[1] - overlap_tgt_frames].T)
                                        prev_tails[instr] = body[:, -overlap_tgt_frames:]
                                    else:
                                        writers[instr].write(body.T)
                                        prev_tails[instr] = None

                                if not img_skip.get(instr, True):
                                    needed = int(args.draw_spectro * sample_rate)
                                    existing = sum(buf.shape[1] for buf in spectro_buffers[instr])
                                    remaining = max(needed - existing, 0)
                                    if remaining > 0:
                                        spectro_buffers[instr].append(body[:, :remaining])

                    for instr, writer in writers.items():
                        writer.close()
                        if not audio_skip.get(instr, False):
                            _, output_path, _ = output_targets[instr]
                            print("Wrote file:", output_path)

                    if args.draw_spectro > 0 and output_targets:
                        for instr in instruments_to_write:
                            output_dir, output_path, output_img_path = output_targets[instr]
                            if img_skip.get(instr, True) or output_img_path is None:
                                continue
                            if getattr(args, 'skip_existing', False) and _is_nonempty_file(output_img_path):
                                print("Skip existing file:", output_img_path)
                                continue
                            if spectro_buffers[instr]:
                                spectro_audio = np.concatenate(spectro_buffers[instr], axis=1)
                                draw_spectrogram(spectro_audio.T, sample_rate, args.draw_spectro, output_img_path)
                                print("Wrote file:", output_img_path)
                finally:
                    for writer in writers.values():
                        try:
                            writer.close()
                        except Exception:
                            pass
                continue

        if long_file and (info is None or info.frames <= 0 or info.samplerate <= 0):
            print("Warning: cannot read file metadata for sample-based splitting. Falling back to time-based splitting.")
            if 'normalize' in config.inference and config.inference['normalize'] is True:
                norm_params = None
            else:
                norm_params = None
            num_segments = int(math.ceil(duration_sec / max_split_seconds)) if duration_sec else 0
            writers = {}
            audio_skip = {}
            img_skip = {}
            spectro_buffers = {}
            output_targets = {}
            instruments_to_write = []
            try:
                for seg_idx in range(num_segments):
                    offset = seg_idx * max_split_seconds
                    seg_dur = min(max_split_seconds, duration_sec - offset)
                    if seg_dur <= 0:
                        continue
                    try:
                        mix, sr = _load_audio_segment(path, offset, seg_dur)
                    except Exception as e:
                        print(f'Cannot read track: {format(path)}')
                        print(f'Error message: {str(e)}')
                        break

                    mix = _prepare_mix(mix)

                    waveforms_orig, instruments_to_write, norm_params = _process_mix_segment(mix, norm_params)
                    if waveforms_orig is None:
                        break

                    # Initialize output targets and skip flags on the first successful segment
                    if not output_targets:
                        for instr in instruments_to_write:
                            output_dir, output_path, output_img_path = _build_output_paths(instr, file_name, path)
                            output_targets[instr] = (output_dir, output_path, output_img_path)
                            if args.skip_existing and _is_nonempty_file(output_path):
                                audio_skip[instr] = True
                                print("Skip existing file:", output_path)
                            else:
                                audio_skip[instr] = False
                            if args.draw_spectro > 0 and output_img_path:
                                img_skip[instr] = args.skip_existing and _is_nonempty_file(output_img_path)
                                if img_skip[instr]:
                                    print("Skip existing file:", output_img_path)
                            else:
                                img_skip[instr] = True
                            spectro_buffers[instr] = []

                    for instr in instruments_to_write:
                        estimates = waveforms_orig[instr]
                        if 'normalize' in config.inference and config.inference['normalize'] is True:
                            estimates = denormalize_audio(estimates, norm_params)

                        output_dir, output_path, output_img_path = output_targets[instr]
                        if not audio_skip.get(instr, False):
                            os.makedirs(output_dir, exist_ok=True)
                            if instr not in writers:
                                sf_format = codec.upper()
                                writers[instr] = sf.SoundFile(
                                    output_path,
                                    mode='w',
                                    samplerate=sr,
                                    channels=estimates.shape[0],
                                    subtype=subtype,
                                    format=sf_format,
                                )
                            writers[instr].write(estimates.T)

                        if not img_skip.get(instr, True):
                            needed = int(args.draw_spectro * sr)
                            existing = sum(buf.shape[1] for buf in spectro_buffers[instr])
                            remaining = max(needed - existing, 0)
                            if remaining > 0:
                                spectro_buffers[instr].append(estimates[:, :remaining])

                for instr, writer in writers.items():
                    writer.close()
                    if not audio_skip.get(instr, False):
                        _, output_path, _ = output_targets[instr]
                        print("Wrote file:", output_path)

                if args.draw_spectro > 0 and output_targets:
                    for instr in instruments_to_write:
                        output_dir, output_path, output_img_path = output_targets[instr]
                        if img_skip.get(instr, True) or output_img_path is None:
                            continue
                        if getattr(args, 'skip_existing', False) and _is_nonempty_file(output_img_path):
                            print("Skip existing file:", output_img_path)
                            continue
                        if spectro_buffers[instr]:
                            spectro_audio = np.concatenate(spectro_buffers[instr], axis=1)
                            draw_spectrogram(spectro_audio.T, sr, args.draw_spectro, output_img_path)
                            print("Wrote file:", output_img_path)
            finally:
                for writer in writers.values():
                    try:
                        writer.close()
                    except Exception:
                        pass
            continue

        # If mono audio we must adjust it depending on model
        mix = _prepare_mix(mix)

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
