"""Trích đặc trưng âm thanh tiếng gõ dừa.

Pipeline: load wav -> phát hiện onset -> cắt cửa sổ cố định quanh tiếng gõ -> chuẩn hoá
-> MFCC (cho RF/SVM) và log-mel spectrogram (cho CNN).

Tham số front-end cố định để khớp với bộ trích mel trên thiết bị nhúng
(ESP-DSP / TFLite Micro micro-frontend đều dùng FFT 512, hop 10 ms, 40 mel ở 16 kHz):
  SR=16000, N_FFT=512 (32 ms), HOP=160 (10 ms), N_MELS=40, cửa sổ gõ 20 ms + 480 ms = 8000 mẫu.

Chạy CLI:
  python -m src.features --meta data/processed/mendeley/metadata.csv --out data/processed/features.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000
N_FFT = 512
HOP = 160
N_MELS = 40
N_MFCC = 20
PRE_MS = 20
POST_MS = 480
FMIN = 60.0
FMAX = 8000.0


def window_samples(sr: int = SR, pre_ms: int = PRE_MS, post_ms: int = POST_MS) -> int:
    return int(round(sr * (pre_ms + post_ms) / 1000))


WIN = window_samples()  # 8000
# Số khung thời gian của log-mel với center=True: 1 + WIN // HOP
N_FRAMES = 1 + WIN // HOP  # 51


# ----------------------------------------------------------------------------- I/O
def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Đọc wav -> mono float32 ở `sr` (resample nếu cần)."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32)


# ------------------------------------------------------------------- onset / crop
ENV_FRAME = 80  # 5 ms ở 16 kHz
MIN_GAP_MS = 300  # hai lần gõ cách nhau ít nhất 300 ms
REL_THRESH = 0.1  # đỉnh gõ phải >= 10% đỉnh mạnh nhất (năng lượng) trong bản ghi
MIN_TAIL_MS = 250  # bỏ lần gõ bị cắt cụt ở cuối bản ghi (còn < 250 ms sau onset)


def energy_envelope(y: np.ndarray, frame: int = ENV_FRAME) -> np.ndarray:
    n = len(y) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return (y[: n * frame].reshape(n, frame).astype(np.float32) ** 2).mean(axis=1)


def detect_onsets(
    y: np.ndarray,
    sr: int = SR,
    min_gap_ms: int = MIN_GAP_MS,
    rel_thresh: float = REL_THRESH,
    min_tail_ms: int = MIN_TAIL_MS,
) -> list[int]:
    """Tất cả vị trí bắt đầu tiếng gõ trong bản ghi (chỉ số mẫu), tăng dần.

    Thuật toán thuần năng lượng (dễ port sang MCU): đường bao năng lượng khung 5 ms ->
    cực đại cục bộ trong cửa sổ ±min_gap -> giữ đỉnh >= rel_thresh * đỉnh lớn nhất ->
    lùi từ đỉnh về điểm năng lượng rơi dưới 10% đỉnh đó (tối đa 50 ms) = onset.
    Bản ghi Mendeley chứa ~3 lần gõ / 3 s nên hàm này trả về nhiều onset.
    """
    env = energy_envelope(y)
    if env.size == 0 or env.max() <= 0:
        return []
    frame = ENV_FRAME
    gap = max(1, int(min_gap_ms / 1000 * sr / frame))
    thr = rel_thresh * float(env.max())
    peaks: list[int] = []
    for i in np.nonzero(env >= thr)[0]:
        lo, hi = max(0, i - gap), min(len(env), i + gap + 1)
        if env[i] >= env[lo:hi].max() and (not peaks or i - peaks[-1] >= gap):
            peaks.append(int(i))
    onsets: list[int] = []
    back_max = int(0.05 * sr / frame)
    for pk in peaks:
        k = pk
        while k > 0 and pk - k < back_max and env[k - 1] > 0.1 * env[pk]:
            k -= 1
        onset = k * frame
        if len(y) - onset >= int(min_tail_ms / 1000 * sr):
            onsets.append(int(onset))
    return onsets


def detect_onset(y: np.ndarray, sr: int = SR) -> int:
    """Onset của lần gõ mạnh nhất (năng lượng trong 50 ms sau onset lớn nhất). 0 nếu im lặng."""
    ons = detect_onsets(y, sr, min_tail_ms=0)
    if not ons:
        return 0
    span = int(0.05 * sr)
    return max(ons, key=lambda o: float(np.sum(y[o : o + span] ** 2)))


def crop_tap(
    y: np.ndarray, sr: int = SR, pre_ms: int = PRE_MS, post_ms: int = POST_MS, onset: int | None = None
) -> np.ndarray:
    """Cắt cửa sổ cố định [onset - pre, onset + post), zero-pad hai đầu nếu thiếu."""
    if onset is None:
        onset = detect_onset(y, sr)
    pre = int(round(sr * pre_ms / 1000))
    total = window_samples(sr, pre_ms, post_ms)
    start = onset - pre
    out = np.zeros(total, dtype=np.float32)
    src_a, src_b = max(start, 0), min(start + total, len(y))
    if src_b > src_a:
        dst_a = src_a - start
        out[dst_a : dst_a + (src_b - src_a)] = y[src_a:src_b]
    return out


def normalize(y: np.ndarray) -> np.ndarray:
    """Trừ DC rồi chia peak về 1.0 -> bớt phụ thuộc lực gõ và khoảng cách mic."""
    y = y.astype(np.float32) - np.float32(np.mean(y)) if y.size else y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y / peak if peak > 0 else y


def preprocess(path: str | Path, sr: int = SR) -> np.ndarray:
    """load -> crop quanh tiếng gõ -> normalize. Trả về đoạn WIN mẫu."""
    return normalize(crop_tap(load_audio(path, sr), sr))


# --------------------------------------------------------------------- features
def log_mel(seg: np.ndarray, sr: int = SR, n_mels: int = N_MELS, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Log-mel spectrogram (dB, ref=max, top_db=80) shape (n_mels, n_frames)."""
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=FMIN, fmax=FMAX, power=2.0, center=True
    )
    return librosa.power_to_db(S, ref=np.max, top_db=80.0).astype(np.float32)


def mfcc(seg: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC, n_fft: int = N_FFT, hop: int = HOP, n_mels: int = N_MELS) -> np.ndarray:
    """MFCC shape (n_mfcc, n_frames), tính từ cùng mel filterbank với log_mel."""
    S_db = log_mel(seg, sr, n_mels, n_fft, hop)
    return librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc).astype(np.float32)


def mfcc_stats(seg: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Vector cố định cho RF/SVM: mean và std theo thời gian của MFCC -> shape (2*n_mfcc,)."""
    m = mfcc(seg, sr, n_mfcc)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


# ---------------------------------------------------------------------- dataset
def read_metadata(meta_csv: str | Path) -> list[dict]:
    with Path(meta_csv).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def segments_from_file(path: str | Path, sr: int = SR, per_tap: bool = True) -> list[np.ndarray]:
    """Các đoạn WIN mẫu đã chuẩn hoá, một đoạn cho mỗi lần gõ (per_tap) hoặc chỉ lần gõ mạnh nhất."""
    y = load_audio(path, sr)
    onsets = detect_onsets(y, sr) if per_tap else [detect_onset(y, sr)]
    if not onsets:
        onsets = [detect_onset(y, sr)]
    return [normalize(crop_tap(y, sr, onset=o)) for o in onsets]


def build_dataset(
    meta_csv: str | Path, classes: list[str], sr: int = SR, per_tap: bool = True, verbose: bool = False
) -> dict:
    """Đọc metadata.csv, lọc theo `classes`, trích đặc trưng cho từng lần gõ trong từng file.

    Trả về dict: X_mfcc (N, 2*N_MFCC), X_mel (N, N_MELS, N_FRAMES), y (N,) int, groups (N,) fruit_id,
    files (N,), tap_idx (N,), classes. Đường dẫn wav tính tương đối so với thư mục chứa metadata.csv.
    """
    meta_csv = Path(meta_csv)
    root = meta_csv.parent
    rows = [r for r in read_metadata(meta_csv) if r["label"] in classes]
    idx = {c: i for i, c in enumerate(classes)}
    X_mfcc, X_mel, y, groups, files, tap_idx = [], [], [], [], [], []
    it = rows
    if verbose:
        from tqdm import tqdm

        it = tqdm(rows, desc="features")
    for r in it:
        for k, seg in enumerate(segments_from_file(root / r["file"], sr, per_tap)):
            X_mfcc.append(mfcc_stats(seg, sr))
            X_mel.append(log_mel(seg, sr))
            y.append(idx[r["label"]])
            groups.append(r["fruit_id"])
            files.append(r["file"])
            tap_idx.append(k)
    return {
        "X_mfcc": np.stack(X_mfcc) if X_mfcc else np.zeros((0, 2 * N_MFCC), np.float32),
        "X_mel": np.stack(X_mel) if X_mel else np.zeros((0, N_MELS, N_FRAMES), np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "groups": np.asarray(groups),
        "files": np.asarray(files),
        "tap_idx": np.asarray(tap_idx, dtype=np.int64),
        "classes": np.asarray(classes),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, default=Path("data/processed/mendeley/metadata.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/features.npz"))
    p.add_argument("--classes", default="im,m,om", help="danh sách nhãn, ví dụ 'im,m,om' hoặc 'm,om'")
    p.add_argument("--single-tap", action="store_true", help="chỉ lấy lần gõ mạnh nhất mỗi file (mặc định: mọi lần gõ)")
    a = p.parse_args(argv)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    ds = build_dataset(a.meta, classes, per_tap=not a.single_tap, verbose=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **ds)
    from collections import Counter

    print(f"Lưu {a.out}: X_mfcc {ds['X_mfcc'].shape}, X_mel {ds['X_mel'].shape}")
    print("Phân bố nhãn:", {classes[k]: v for k, v in sorted(Counter(ds["y"].tolist()).items())})
    print("Số quả      :", len(set(ds["groups"].tolist())))
    print("Số file     :", len(set(ds["files"].tolist())), "| lần gõ / file:", dict(sorted(Counter(Counter(ds["files"].tolist()).values()).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
