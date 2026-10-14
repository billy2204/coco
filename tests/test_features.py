"""Test cho src/features.py trên tín hiệu tổng hợp (không cần dataset thật)."""
import numpy as np
import pytest

from src import features as F

SR = 16_000


def synth_tap(sr=SR, onset_s=1.0, total_s=3.0, f0=900.0, decay=40.0, noise=1e-3, seed=0):
    """Mô phỏng tiếng gõ: nền im lặng + sin tắt dần (exponential decay) bắt đầu tại onset_s."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(total_s * sr)) / sr
    y = rng.normal(0, noise, t.shape).astype(np.float32)
    start = int(onset_s * sr)
    tt = t[start:] - onset_s
    y[start:] += (np.exp(-decay * tt) * np.sin(2 * np.pi * f0 * tt)).astype(np.float32)
    return y


def test_detect_onset_near_true_onset():
    y = synth_tap(onset_s=1.0)
    idx = F.detect_onset(y, SR)
    assert abs(idx - SR * 1.0) < 0.03 * SR  # sai số < 30 ms


def test_detect_onsets_finds_every_tap():
    """3 lần gõ cách nhau 1 s, lần thứ 2 yếu hơn (50%)."""
    y = synth_tap(onset_s=0.5, total_s=3.0)
    y += 0.5 * synth_tap(onset_s=1.5, total_s=3.0, noise=0.0)
    y += synth_tap(onset_s=2.4, total_s=3.0, noise=0.0)
    ons = F.detect_onsets(y, SR)
    assert len(ons) == 3
    for o, t in zip(ons, [0.5, 1.5, 2.4]):
        assert abs(o - t * SR) < 0.03 * SR


def test_detect_onsets_drops_truncated_tap_and_weak_peaks():
    y = synth_tap(onset_s=0.5, total_s=3.0)
    y += synth_tap(onset_s=2.9, total_s=3.0, noise=0.0)  # còn 100 ms -> bỏ
    y += 0.05 * synth_tap(onset_s=1.5, total_s=3.0, noise=0.0)  # 5% biên độ = 0.25% năng lượng -> bỏ
    assert F.detect_onsets(y, SR) == F.detect_onsets(y, SR)  # deterministic
    assert len(F.detect_onsets(y, SR)) == 1


def test_detect_onset_fallback_on_silence():
    y = np.zeros(SR, dtype=np.float32)
    idx = F.detect_onset(y, SR)
    assert 0 <= idx < len(y)


def test_crop_tap_fixed_length_and_alignment():
    y = synth_tap(onset_s=1.0)
    seg = F.crop_tap(y, SR, pre_ms=20, post_ms=480)
    assert seg.shape == (F.window_samples(SR, 20, 480),)
    assert seg.shape[0] == 8000
    # Đỉnh năng lượng phải nằm ngay sau phần pre (20 ms = 320 mẫu)
    assert 300 <= int(np.argmax(np.abs(seg))) <= 700


def test_crop_tap_zero_pads_when_onset_near_end():
    y = synth_tap(onset_s=2.9, total_s=3.0)
    seg = F.crop_tap(y, SR, pre_ms=20, post_ms=480)
    assert seg.shape[0] == 8000
    assert np.all(seg[-1000:] == 0)


def test_normalize_peak_and_dc():
    y = np.array([0.5, 0.7, 0.9, 0.7, 0.5], dtype=np.float32)
    z = F.normalize(y)
    assert abs(float(np.mean(z))) < 1e-6
    assert np.isclose(float(np.max(np.abs(z))), 1.0)
    assert np.all(F.normalize(np.zeros(10, dtype=np.float32)) == 0)


def test_mfcc_shapes():
    seg = F.crop_tap(synth_tap(), SR)
    mat = F.mfcc(seg, SR, n_mfcc=20)
    assert mat.shape[0] == 20
    vec = F.mfcc_stats(seg, SR, n_mfcc=20)
    assert vec.shape == (40,)
    assert np.all(np.isfinite(vec))


def test_log_mel_shape_and_range():
    seg = F.crop_tap(synth_tap(), SR)
    mel = F.log_mel(seg, SR, n_mels=40)
    assert mel.shape[0] == 40
    assert mel.shape[1] == F.N_FRAMES
    assert np.all(np.isfinite(mel))
    assert mel.max() <= 0.0 + 1e-6  # top_db ref=max -> giá trị lớn nhất là 0 dB


def test_mel_params_match_embedded_frontend():
    assert F.N_FFT == 512 and F.HOP == 160 and F.N_MELS == 40 and F.SR == 16_000


def test_build_dataset_from_metadata(tmp_path):
    import csv
    import soundfile as sf

    root = tmp_path
    rows = []
    for i, (fid, lab) in enumerate([("c1", "im"), ("c2", "m"), ("c3", "om"), ("c3", "om")]):
        rel = f"{lab}/{fid}_{'ABC'[i % 3]}.wav"
        (root / lab).mkdir(exist_ok=True)
        sf.write(root / rel, (synth_tap(f0=600 + 100 * i) * 32767).astype(np.int16), SR)
        rows.append({"file": rel, "fruit_id": fid, "ridge": "ABC"[i % 3], "label": lab,
                     "device": "t", "date": "", "source": "t", "notes": ""})
    meta = root / "metadata.csv"
    with meta.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ds = F.build_dataset(meta, classes=["im", "m", "om"])
    assert ds["X_mfcc"].shape == (4, 40)
    assert ds["X_mel"].shape == (4, 40, F.N_FRAMES)
    assert list(ds["y"]) == [0, 1, 2, 2]
    assert list(ds["groups"]) == ["c1", "c2", "c3", "c3"]
    assert list(ds["tap_idx"]) == [0, 0, 0, 0]

    ds2 = F.build_dataset(meta, classes=["m", "om"])
    assert ds2["X_mfcc"].shape[0] == 3 and list(ds2["y"]) == [0, 1, 1]


def test_build_dataset_expands_multi_tap_files(tmp_path):
    import csv
    import soundfile as sf

    y = synth_tap(onset_s=0.5, total_s=3.0) + synth_tap(onset_s=1.6, total_s=3.0, noise=0.0)
    sf.write(tmp_path / "a.wav", (y / np.abs(y).max() * 32767 * 0.9).astype(np.int16), SR)
    meta = tmp_path / "metadata.csv"
    with meta.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "fruit_id", "ridge", "label", "device", "date", "source", "notes"])
        w.writeheader()
        w.writerow({"file": "a.wav", "fruit_id": "c9", "ridge": "A", "label": "m", "device": "", "date": "", "source": "", "notes": ""})
    ds = F.build_dataset(meta, classes=["m"])
    assert ds["X_mel"].shape[0] == 2 and list(ds["tap_idx"]) == [0, 1] and list(ds["groups"]) == ["c9", "c9"]
    ds1 = F.build_dataset(meta, classes=["m"], per_tap=False)
    assert ds1["X_mel"].shape[0] == 1
