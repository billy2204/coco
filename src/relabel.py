"""Cập nhật nhãn trong metadata.csv của một phiên thu sau khi bổ quả.

fruits.csv (ghi tay sau khi bổ) cần ít nhất 2 cột: fruit_id, label_final.
fruit_id có thể là 'f003' hoặc đầy đủ '<session>_f003'.

Chạy:
  python -m src.relabel --session 2026-10-20_lot1 [--fruits fruits.csv] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.convert_mendeley import META_FIELDS

VALID = {"im", "m", "om", "unknown"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/recorded"))
    p.add_argument("--fruits", default="fruits.csv", help="đường dẫn tương đối trong thư mục phiên, hoặc tuyệt đối")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    sdir = a.root / a.session
    meta_path = sdir / "metadata.csv"
    fruits_path = Path(a.fruits) if Path(a.fruits).is_absolute() else sdir / a.fruits
    if not meta_path.exists() or not fruits_path.exists():
        print(f"Thiếu {meta_path} hoặc {fruits_path}")
        return 1

    labels: dict[str, str] = {}
    with fruits_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fid = r["fruit_id"].strip()
            if not fid.startswith(a.session):
                fid = f"{a.session}_{fid}"
            lab = r["label_final"].strip().lower()
            if lab not in VALID:
                print(f"Nhãn không hợp lệ '{lab}' cho {fid} (hợp lệ: {sorted(VALID)})")
                return 1
            labels[fid] = lab

    with meta_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    changed = Counter()
    missing = set()
    for r in rows:
        new = labels.get(r["fruit_id"])
        if new is None:
            missing.add(r["fruit_id"])
            continue
        if r["label"] != new:
            changed[(r["label"], new)] += 1
            r["label"] = new

    print(f"{len(rows)} bản ghi, {len(labels)} quả trong fruits.csv")
    for (old, new), n in changed.items():
        print(f"  {old} -> {new}: {n} bản ghi")
    if missing:
        print(f"  [cảnh báo] {len(missing)} quả không có trong fruits.csv: {sorted(missing)[:10]}")
    still_unknown = sum(r["label"] == "unknown" for r in rows)
    if still_unknown:
        print(f"  [cảnh báo] còn {still_unknown} bản ghi nhãn unknown")
    if a.dry_run:
        print("(dry-run, không ghi)")
        return 0
    backup = meta_path.with_suffix(".csv.bak")
    backup.write_bytes(meta_path.read_bytes())
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi {meta_path} (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
