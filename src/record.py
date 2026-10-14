"""Thu âm tiếng gõ dừa từ mic, tự tách mỗi lần gõ thành 1 file wav, gán nhãn bằng phím, ghi metadata.csv.

Cách dùng:
  python -m src.record --list-devices
  python -m src.record --device 1 --session 2026-10-14_lot1 --device-name "pi0_inmp441" --classes im,m,om

Phím trong lúc thu:
  1..9  : chọn nhãn hiện hành theo thứ tự --classes (1=im, 2=m, 3=om); áp cho các lần gõ tiếp theo
  0     : nhãn 'unknown' (gán sau khi bổ quả, xem docs/data_collection_protocol.md)
  n     : sang quả mới (fruit_id tăng), reset đếm lần gõ
  u     : xoá bản ghi cuối (file + dòng metadata)
  q     : thoát

Phát hiện gõ: năng lượng RMS khung 10 ms vượt ngưỡng (--threshold, so với nền đo lúc khởi động)
rồi cắt cửa sổ [onset - 20 ms, onset + 480 ms] (8000 mẫu @16 kHz), thời gian chết 300 ms.
Schema metadata.csv giống src/convert_mendeley.py để gộp dữ liệu.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.convert_mendeley import META_FIELDS

SR = 16_000
BLOCK = 160  # 10 ms
PRE = int(0.020 * SR)
POST = int(0.480 * SR)
WIN = PRE + POST
DEAD_TIME_S = 0.3


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def key_reader(q: "queue.Queue[str]", stop: threading.Event) -> None:
    """Đọc phím không chặn: msvcrt trên Windows, termios trên POSIX."""
    if sys.platform == "win32":
        import msvcrt

        while not stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                q.put(ch)
            else:
                time.sleep(0.02)
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            import select

            while not stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    q.put(sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


class TapRecorder:
    def __init__(self, a, classes: list[str]):
        self.a, self.classes = a, classes
        self.session_dir = Path(a.out) / a.session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.session_dir / "metadata.csv"
        self.fruit_id = a.start_fruit
        self.tap_idx = 0
        self.label = "unknown"
        self.ring = np.zeros(SR * 2, dtype=np.float32)  # 2 s buffer vòng
        self.ring_pos = 0
        self.total = 0  # tổng mẫu đã nhận
        self.pending: int | None = None  # tổng-mẫu-tuyệt-đối của onset đang chờ đủ POST
        self.last_onset_total = -10**9
        self.noise_rms = 1e-4
        self.calib_blocks = 0
        self.records: list[dict] = []
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    # -- audio callback (thread của PortAudio): chỉ đẩy vào queue
    def callback(self, indata, frames, t, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.audio_q.put(indata[:, 0].copy())

    def push(self, block: np.ndarray) -> None:
        n = len(block)
        idx = (self.ring_pos + np.arange(n)) % len(self.ring)
        self.ring[idx] = block
        self.ring_pos = (self.ring_pos + n) % len(self.ring)
        self.total += n

    def take(self, start_total: int, length: int) -> np.ndarray:
        """Lấy `length` mẫu bắt đầu từ chỉ số tuyệt đối `start_total` trong buffer vòng."""
        start = (self.ring_pos - (self.total - start_total)) % len(self.ring)
        idx = (start + np.arange(length)) % len(self.ring)
        return self.ring[idx].copy()

    def process_block(self, block: np.ndarray) -> None:
        self.push(block)
        rms = float(np.sqrt(np.mean(block**2)) + 1e-9)
        if self.calib_blocks < 100:  # 1 s đầu đo nền
            self.noise_rms = 0.9 * self.noise_rms + 0.1 * rms if self.calib_blocks else rms
            self.calib_blocks += 1
            if self.calib_blocks == 100:
                print(f"nền RMS = {self.noise_rms:.5f}; ngưỡng = x{self.a.threshold}")
            return
        if self.pending is None:
            if rms > self.a.threshold * self.noise_rms and (self.total - self.last_onset_total) > DEAD_TIME_S * SR:
                # onset ≈ đầu block này; lùi thêm 1 block để không mất sườn lên
                self.pending = max(self.total - 2 * BLOCK, 0)
                self.last_onset_total = self.total
        elif self.total - self.pending >= POST:
            seg = self.take(self.pending - PRE, WIN) if self.pending >= PRE else np.pad(self.take(0, self.pending + POST), (PRE - self.pending, 0))
            self.pending = None
            self.save(seg)

    def save(self, seg: np.ndarray) -> None:
        self.tap_idx += 1
        name = f"f{self.fruit_id:03d}_t{self.tap_idx:02d}.wav"
        path = self.session_dir / name
        peak = float(np.max(np.abs(seg)))
        sf.write(path, (np.clip(seg / peak * 0.9 if peak > 0 else seg, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
        row = {
            "file": name,
            "fruit_id": f"{self.a.session}_f{self.fruit_id:03d}",
            "ridge": str(self.tap_idx),
            "label": self.label,
            "device": self.a.device_name,
            "date": dt.date.today().isoformat(),
            "source": "recorded",
            "notes": self.a.notes,
        }
        self.records.append(row)
        self.rewrite_meta()
        print(f"  [{len(self.records):3d}] quả {self.fruit_id:03d} gõ {self.tap_idx:02d} nhãn={self.label:<8} peak={peak:.3f} -> {name}")

    def undo(self) -> None:
        if not self.records:
            print("  (không có gì để xoá)")
            return
        row = self.records.pop()
        (self.session_dir / row["file"]).unlink(missing_ok=True)
        self.tap_idx = max(self.tap_idx - 1, 0)
        self.rewrite_meta()
        print(f"  đã xoá {row['file']}")

    def rewrite_meta(self) -> None:
        with self.meta_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=META_FIELDS)
            w.writeheader()
            w.writerows(self.records)

    def handle_key(self, ch: str) -> bool:
        if ch == "q":
            return False
        if ch == "n":
            self.fruit_id += 1
            self.tap_idx = 0
            print(f"-> quả mới: {self.fruit_id:03d}")
        elif ch == "u":
            self.undo()
        elif ch == "0":
            self.label = "unknown"
            print("-> nhãn: unknown")
        elif ch.isdigit() and 1 <= int(ch) <= len(self.classes):
            self.label = self.classes[int(ch) - 1]
            print(f"-> nhãn: {self.label}")
        return True

    def run(self) -> None:
        import sounddevice as sd

        if self.meta_path.exists():  # tiếp tục phiên cũ
            with self.meta_path.open(encoding="utf-8", newline="") as f:
                self.records = list(csv.DictReader(f))
            if self.records:
                last = self.records[-1]
                self.fruit_id = int(last["fruit_id"].rsplit("_f", 1)[1])
                self.tap_idx = int(last["ridge"])
                print(f"tiếp tục phiên {self.a.session}: {len(self.records)} bản ghi, quả hiện tại {self.fruit_id:03d}")

        print(__doc__.split("Phím trong lúc thu:")[1].split("Phát hiện gõ")[0])
        print(f"nhãn: {dict(enumerate(self.classes, 1))}; quả {self.fruit_id:03d}; đang đo nền 1 s, giữ yên lặng...")
        keys: "queue.Queue[str]" = queue.Queue()
        stop = threading.Event()
        threading.Thread(target=key_reader, args=(keys, stop), daemon=True).start()
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK, device=self.a.device, callback=self.callback):
                while True:
                    try:
                        self.process_block(self.audio_q.get(timeout=0.05))
                    except queue.Empty:
                        pass
                    try:
                        if not self.handle_key(keys.get_nowait()):
                            break
                    except queue.Empty:
                        pass
        finally:
            stop.set()
            print(f"\nkết thúc: {len(self.records)} bản ghi trong {self.session_dir} (metadata.csv)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", type=int, default=None, help="chỉ số mic theo --list-devices")
    p.add_argument("--device-name", default="unknown_mic", help="tên thiết bị ghi vào metadata (vd pi0_inmp441, esp32s3_sph0645, piezo)")
    p.add_argument("--session", default=dt.datetime.now().strftime("%Y-%m-%d_%H%M"))
    p.add_argument("--out", type=Path, default=Path("data/raw/recorded"))
    p.add_argument("--classes", default="im,m,om")
    p.add_argument("--start-fruit", type=int, default=1)
    p.add_argument("--threshold", type=float, default=8.0, help="hệ số RMS so với nền để coi là tiếng gõ")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    if a.list_devices:
        list_devices()
        return 0
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    TapRecorder(a, classes).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
