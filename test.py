import os
import subprocess
from pathlib import Path
import sys

from pynvml import *
import threading
import time

class GPUMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.max_vram = 0
        self._stop_event = threading.Event()
        self._thread = None
        self.handle = None

    def start(self):
        """監視スレッドを開始"""
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.device_id)
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.start()
        except NVMLError as e:
            print(f"NVML初期化エラー: {e}")

    def _monitor_loop(self):
        """VRAM使用量を定期的にチェックするループ"""
        while not self._stop_event.is_set():
            try:
                # デバイス全体のメモリ情報を取得
                mem_info = nvmlDeviceGetMemoryInfo(self.handle) # ★ 修正点2: 関数名の変更
                
                # 現在使用されているVRAM量を取得し、MBに変換
                current_vram_used = mem_info.used / (1024 * 1024) 
                
                self.max_vram = max(self.max_vram, current_vram_used)
                
                time.sleep(0.1)  # 100msごとにチェック

            except NVMLError as e:
                # エラー処理
                # print(f"NVML Error: {e}")
                pass
            except Exception as e:
                # print(f"Error during monitoring: {e}")
                pass


    def stop(self):
        """監視スレッドを停止し、最大値を返す"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join() # スレッドが終了するのを待つ
        
        # nvmlのシャットダウン処理は、スレッドが終了した後に行う
        try:
            if self.handle:
                 nvmlShutdown()
        except NVMLError:
            pass
            
        print(f"--- GPU VRAM使用量 ---")
        print(f"最大VRAM使用量 (GPU {self.device_id}): {self.max_vram:.2f} MB")
        return self.max_vram





def inference(
    model_type: str,
    config_path: str,
    start_check_point: str,
    input_folder: str,
    store_dir: str,
    output_instruments: list[str] | None = None,
):
    cmd = [
        sys.executable,
        "inference.py",
        "--skip_existing",
        "--model_type", model_type,
        "--config_path", str(Path(config_path)),
        "--start_check_point", start_check_point,
        "--input_folder", str(Path(input_folder)),
        "--store_dir", str(Path(store_dir)),
    ]

    if output_instruments:
        cmd.extend(["--output_instruments", *output_instruments])

    print("Executing command:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,      # 異常終了時に例外を送出
    )

if __name__ == '__main__':
    # gpu_monitor = GPUMonitor(device_id=0) # 使用するGPU ID
    
    # gpu_monitor.start()
    

    import time

    start = time.time()       # 開始時刻


    input_folder = '/mnt/r/sora_movie/audio_original'

    # input_folder = '/mnt/r/sora_movie/audio_original'
    # store_dir = '/mnt/r/sora_movie/audio_split'

    # 声で使用
    model_type = 'mel_band_roformer'
    config_path = 'checkpoint/big_beta5e.yaml'
    start_check_point = 'checkpoint/big_beta5e.ckpt'
    store_dir = '/mnt/r/sora_movie/audio_voice'

    inference(model_type, config_path, start_check_point, input_folder, store_dir, output_instruments=["vocals"])

    # 声以外で使用
    model_type = 'bs_roformer'
    # /home/virtual1/Music-Source-Separation-Training/models/bs_roformer/bs_roformer.pyを変更
    config_path = 'checkpoint/bs_hyperace.yaml'
    start_check_point = 'checkpoint/bs_hyperace.ckpt'
    store_dir = '/mnt/r/sora_movie/audio_inst'

    inference(model_type, config_path, start_check_point, input_folder, store_dir, output_instruments=["instrument"])


    print(f"実行時間: {time.time() - start:.3f} 秒")
        

    # gpu_monitor.stop()







