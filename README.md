# coco — Phân loại độ chín dừa bằng âm thanh gõ

Gõ vào quả dừa → thu âm → phân loại độ chín (`im` non / `m` vừa / `om` già).
Mục tiêu triển khai on-device: ESP32-S3 (TFLite Micro) hoặc Raspberry Pi Zero 2 W.

## Cấu trúc

```
data/raw/        dữ liệu gốc (Mendeley xlsx, wav tự thu) — không commit
data/processed/  wav 16 kHz đã tách + metadata.csv + features.npz
notebooks/       khám phá dữ liệu
src/             features.py, train_baseline.py, train_cnn.py, record.py, convert_mendeley.py
models/          model đã train (.h5 / .tflite / .joblib)
docs/            hardware.md, data_collection_protocol.md
tests/           pytest
```

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

- Caladcad J.A., Piedad E. Jr. et al., *Acoustic Signal Dataset: Tall Coconut Fruit Species*,
  Mendeley Data V1 (2023), DOI [10.17632/hxh8kd3snj.1](https://doi.org/10.17632/hxh8kd3snj.1), CC BY 4.0.
  Bài báo gốc: Caladcad et al. 2020, *Determining Philippine coconut maturity level using machine
  learning algorithms based on acoustic signal*, Computers and Electronics in Agriculture 172, 105327.
- Chi tiết layout file và cách tải: `data/raw/mendeley/README.md`.
