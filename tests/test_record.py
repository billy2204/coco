"""Test logic tách gõ của src/record.py bằng cách nạp block âm thanh tổng hợp (không cần mic)."""
import csv
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from src import record as R
from tests.test_features import synth_tap


def make_recorder(tmp_path):
    a = SimpleNamespace(out=tmp_path, session="s1", device_name="test", notes="", threshold=8.0, start_fruit=1)
    return R.TapRecorder(a, ["im", "m", "om"])


def feed(rec, y):
    for i in range(0, len(y) - R.BLOCK + 1, R.BLOCK):
        rec.process_block(y[i : i + R.BLOCK])


def test_three_taps_become_three_files(tmp_path):
    rec = make_recorder(tmp_path)
    y = np.zeros(int(1.2 * R.SR), dtype=np.float32) + np.random.default_rng(1).normal(0, 1e-3, int(1.2 * R.SR)).astype(np.float32)
    taps = synth_tap(onset_s=0.5, total_s=4.0) + synth_tap(onset_s=1.6, total_s=4.0, noise=0) + synth_tap(onset_s=2.7, total_s=4.0, noise=0)
    y = np.concatenate([y, taps.astype(np.float32)])  # 1,2 s nền để hiệu chuẩn + 3 gõ
    rec.handle_key("2")  # nhãn m
    feed(rec, y)
    assert len(rec.records) == 3
    assert [r["label"] for r in rec.records] == ["m", "m", "m"]
    assert [r["ridge"] for r in rec.records] == ["1", "2", "3"]
    for r in rec.records:
        seg, sr = sf.read(tmp_path / "s1" / r["file"], dtype="float32")
        assert sr == R.SR and len(seg) == R.WIN
        # đỉnh nằm ngay sau phần PRE (20 ms)
        assert R.PRE - 100 <= int(np.argmax(np.abs(seg))) <= R.PRE + 800
    with (tmp_path / "s1" / "metadata.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and rows[0]["fruit_id"] == "s1_f001"


def test_new_fruit_and_undo(tmp_path):
    rec = make_recorder(tmp_path)
    base = np.random.default_rng(2).normal(0, 1e-3, int(1.2 * R.SR)).astype(np.float32)
    feed(rec, np.concatenate([base, synth_tap(onset_s=0.3, total_s=1.5).astype(np.float32)]))
    assert len(rec.records) == 1
    rec.handle_key("n")
    assert rec.fruit_id == 2 and rec.tap_idx == 0
    feed(rec, synth_tap(onset_s=0.3, total_s=1.5).astype(np.float32))
    assert len(rec.records) == 2 and rec.records[-1]["fruit_id"] == "s1_f002"
    rec.handle_key("u")
    assert len(rec.records) == 1
    assert not (tmp_path / "s1" / "f002_t01.wav").exists()
    assert rec.handle_key("q") is False


def test_silence_produces_nothing(tmp_path):
    rec = make_recorder(tmp_path)
    feed(rec, np.random.default_rng(3).normal(0, 1e-3, 3 * R.SR).astype(np.float32))
    assert rec.records == []
