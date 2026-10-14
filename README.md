# coco — Phân loại độ chín dừa bằng âm thanh gõ

Gõ vào quả dừa → thu âm → phân loại độ chín: `im` non (premature) / `m` vừa (mature) / `om` già (overmature).
Mục tiêu triển khai on-device: ESP32-S3 (TFLite Micro) hoặc Raspberry Pi Zero 2 W, mic I2S MEMS hoặc piezo, solenoid gõ.

## Cấu trúc

```
data/raw/mendeley/     xlsx gốc Mendeley + README layout (không commit file xlsx)
data/raw/recorded/     wav tự thu theo phiên (src/record.py)
data/processed/        wav 16 kHz đã tách + metadata.csv + features.npz
src/convert_mendeley.py   xlsx -> 381 wav 16 kHz + metadata.csv
src/features.py           onset -> cắt 0,5 s quanh mỗi lần gõ -> MFCC + log-mel
src/train_baseline.py     RF + SVM trên MFCC, CV theo quả, confusion matrix, F1
src/train_cnn.py          CNN nhỏ trên log-mel -> TFLite int8 (+ .cc cho TFLite Micro)
src/record.py             thu âm từ mic, tự tách từng lần gõ, gán nhãn bằng phím, metadata.csv
src/relabel.py            cập nhật nhãn sau khi bổ quả
docs/hardware.md          ESP32-S3 vs Pi Zero 2 W, sơ đồ mic I2S / piezo / solenoid
docs/data_collection_protocol.md   quy trình thu dữ liệu và xác nhận nhãn
models/                   model đã train (.keras / .tflite / .cc / .joblib)
tests/                    pytest (không cần dataset thật)
```

## Cài đặt (Windows, Python 3.10)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests -q
```

## Pipeline

```powershell
# 1. Tải dataset (278 MB) theo data/raw/mendeley/README.md, rồi tách thành wav 16 kHz
python -m src.convert_mendeley

# 2. Trích đặc trưng: mỗi lần gõ trong file -> 1 mẫu (0,5 s, 40 log-mel x 51 khung, 20 MFCC mean/std)
python -m src.features --classes im,m,om          # hoặc --classes m,om

# 3. Baseline RF + SVM, 5-fold StratifiedGroupKFold theo quả
python -m src.train_baseline --classes im,m,om

# 4. CNN + export TFLite int8 (thêm --cv để chạy 5-fold float trước khi train cuối)
python -m src.train_cnn --classes im,m,om --cv

# 5. Thu dữ liệu mới
python -m src.record --list-devices
python -m src.record --device 1 --device-name pi0_inmp441 --session 2026-10-20_lot1
python -m src.relabel --session 2026-10-20_lot1    # sau khi bổ quả, đọc fruits.csv
```

Tham số front-end cố định để khớp thiết bị nhúng: 16 kHz, FFT 512, hop 160 (10 ms), 40 mel (60–8000 Hz),
cửa sổ 20 ms trước + 480 ms sau onset = 8000 mẫu; log-mel dB (ref = max, top 80 dB) chuẩn hoá `(x + 40) / 40`.

## Dataset

Caladcad J.A., Piedad E. Jr. et al., *Acoustic Signal Dataset: Tall Coconut Fruit Species*, Mendeley Data V1 (2023),
DOI [10.17632/hxh8kd3snj.1](https://doi.org/10.17632/hxh8kd3snj.1), CC BY 4.0. Bài báo gốc: Caladcad et al. 2020,
*Determining Philippine coconut maturity level using machine learning algorithms based on acoustic signal*,
Computers and Electronics in Agriculture 172, 105327.

Thực tế trong file (khác bài báo): **127 quả / 381 bản ghi** (thiếu c67, c86); mỗi bản ghi 3 s chứa **~3 lần gõ**
→ sau khi tách: 1 056 mẫu. Phân bố quả: 8 `im` / 36 `m` / 83 `om`. Chi tiết: `data/raw/mendeley/README.md`.

Không có dataset gõ dưa hấu / sầu riêng công khai trên Kaggle hay Hugging Face (dataset dưa hấu Qilin trên
IEEE DataPort cần đăng ký trả phí và nhãn là độ ngọt), nên không dùng.

## Kết quả hiện tại (Mendeley, 3 lớp, chia theo quả)

| Model | Accuracy | Macro-F1 | Đánh giá |
|---|---|---|---|
| RF (MFCC 40 chiều) | 0,76 | 0,51 | 5-fold OOF, 1 056 mẫu |
| SVM RBF (MFCC) | 0,74 | 0,56 | 5-fold OOF |
| CNN log-mel, float32 | 0,76 | 0,63 | 5-fold OOF (`--cv`) |
| CNN float32, train cuối | 0,67 | 0,56 | 1 fold test (213 mẫu, 24 `im`) |
| CNN int8 TFLite | 0,67 | 0,55 | cùng fold; **30 824 bytes**, 23 827 tham số, arena ~55 KB |

- Quantize int8 gần như không mất độ chính xác (−0,5 điểm accuracy trên cùng test set).
- Chênh lệch giữa OOF 5-fold và 1 fold test phản ánh phương sai lớn khi chỉ có 127 quả: kết quả dao động ±10 điểm
  tuỳ fold. Hãy dùng số OOF để so sánh model, và số 1 fold để so float vs int8.
- Lớp `im` chỉ có 8 quả nên F1 lớp này thấp (CNN nhận đúng 28/64 mẫu `im` trong CV; RF/SVM gần như không nhận);
  đây là lý do cần thu thêm dữ liệu theo `docs/data_collection_protocol.md`. Accuracy ~75% phần lớn đến từ
  lớp `om` chiếm 65% dữ liệu.
- Model trên thiết bị: `models/coconut_cnn_int8.tflite` (Pi, `tflite-runtime`) hoặc `models/coconut_cnn_int8.cc`
  (ESP32-S3, TFLite Micro). Đầu vào int8 shape (1, 40, 51, 1); xem `docs/hardware.md`.
