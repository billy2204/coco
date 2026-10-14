# Quy trình thu dữ liệu tiếng gõ dừa

Mục tiêu: dữ liệu tự thu phải (1) dùng được cùng pipeline với dataset Mendeley, (2) có nhãn tin cậy
nhờ bổ quả xác nhận, (3) đủ cân bằng để lớp `im` (non) học được.

## 1. Chuẩn bị

| Hạng mục | Yêu cầu |
|---|---|
| Thiết bị | Đúng mic + gá + solenoid sẽ dùng cho sản phẩm (xem `docs/hardware.md`). Không đổi mic giữa chừng; nếu đổi, ghi `device` khác trong metadata. |
| Khoảng cách mic | 2–5 cm, cố định bằng gá. Ghi giá trị thật vào `--notes` (vd `mic_3cm`). |
| Lực gõ | Solenoid 12 V, xung 30 ms, đầu gõ cách vỏ 5 mm khi nghỉ. Nếu gõ tay: búa nhỏ (~20 g) thả tự do từ cùng độ cao 5 cm, không đè. |
| Vị trí gõ | 3 gờ (ridge) của quả, giữa hai đầu, như Mendeley. Đánh dấu A/B/C bằng bút. |
| Môi trường | Phòng yên tĩnh, tắt quạt. Trước mỗi phiên ghi 5 s nền (script tự đo 1 s nền và in RMS; nếu RMS nền > 0,01 thì đổi chỗ). |
| Quả | Rửa, lau khô, để nhiệt độ phòng ≥ 30 phút (quả lạnh cho âm khác). Ghi nguồn gốc, ngày hái nếu biết. |

## 2. Thu

1. Khởi động: `python -m src.record --device <id> --device-name pi0_inmp441 --session 2026-10-20_lot1 --classes im,m,om --notes "mic_3cm,solenoid_30ms"`
2. Nhấn `0` để nhãn = `unknown` (nhãn thật gán sau khi bổ). Chỉ nhấn `1/2/3` nếu quả đã được bổ trước hoặc nhãn chắc chắn (vd dừa đã nảy mầm = `om`).
3. Mỗi quả: **3 gờ × 3 lần gõ = 9 bản ghi**, thứ tự A1 A2 A3 B1 B2 B3 C1 C2 C3, nghỉ ≥ 1 s giữa hai lần gõ (script có thời gian chết 300 ms, nhưng để tiếng ngân tắt hẳn cần ~0,5 s).
4. Nếu một lần gõ bị lỗi (chạm hai lần, tiếng ồn ngoài), nhấn `u` xoá ngay rồi gõ lại.
5. Nhấn `n` sang quả tiếp theo; dán nhãn giấy `fNNN` lên quả (trùng `fruit_id` trong metadata).
6. Cuối phiên: kiểm tra `data/raw/recorded/<session>/metadata.csv` có đúng số dòng = 9 × số quả.

Mỗi quả mất khoảng 1 phút. Một phiên 30 quả ≈ 30–40 phút thu + thời gian bổ.

## 3. Xác nhận nhãn bằng bổ quả

Bổ **ngay sau khi thu xong phiên** (cùng ngày), từng quả theo đúng `fruit_id`.

| Tiêu chí | `im` non (premature) | `m` vừa (mature) | `om` già (overmature) |
|---|---|---|---|
| Cùi (cơm dừa) | Mỏng < 5 mm, trong/mềm như thạch, cạo được bằng thìa | 6–10 mm, trắng đục, dẻo, còn uốn được | > 10 mm, trắng cứng, giòn, bẻ gãy |
| Nước | Đầy khoang, ngọt nhẹ | Hơi hụt (lắc nghe óc ách), ngọt rõ | Hụt nhiều (< 2/3 khoang), nhạt hoặc hơi chua |
| Gáo (sọ) | Mềm, cắt được bằng dao | Cứng vừa | Rất cứng, nâu sẫm |
| Vỏ ngoài | Xanh bóng, không có vết nâu | Xanh ngả vàng ở cuống | Nâu/khô một phần, có thể nảy mầm |
| Khối lượng riêng (quả nổi/chìm khi thả vào nước) | Chìm | Lưng chừng | Nổi |

Quy tắc chốt nhãn: cùi là tiêu chí chính; nước và gáo để phá hoà. Trường hợp giáp ranh ghi vào `notes`
(vd `borderline_m_om`) và vẫn gán một nhãn; sau này có thể loại các quả giáp ranh khi đánh giá.

Ghi thêm vào bảng riêng (`data/raw/recorded/<session>/fruits.csv`): `fruit_id, cui_mm, nuoc_ml, gao, vo, noi_chim, label_final, notes`.
Đo cùi bằng thước kẹp tại 2 điểm, lấy trung bình; nước đổ ra cốc đong.

Cập nhật nhãn vào metadata bằng `python -m src.relabel --session <session>` (đọc `fruits.csv` trong thư mục phiên,
thay `unknown` bằng `label_final` theo `fruit_id`, tự backup `metadata.csv.bak`; thêm `--dry-run` để xem trước).

## 4. Số lượng mục tiêu

| Lớp | Mendeley hiện có (quả) | Mục tiêu tự thu (quả) | Lý do |
|---|---|---|---|
| `im` | 8 | **≥ 40** | Lớp yếu nhất; baseline hiện F1 ≈ 0,1–0,2 |
| `m` | 36 | ≥ 30 | |
| `om` | 83 | ≥ 30 | Đã nhiều, thu để khớp miền thiết bị mới |

Mua quả theo lô từ 2–3 nguồn khác nhau (vườn/chợ), mỗi lô trộn cả ba độ chín, tránh mọi quả `im` đến từ cùng một buồng.

## 5. Ghi chú về trộn dữ liệu Mendeley và dữ liệu tự thu

- Mendeley: mic ATR2500 trong buồng xốp, cánh tay servo gõ **~3 lần trong 3 s**; `src/features.py` đã tách từng lần gõ. Dữ liệu tự thu có cùng cấu trúc (mỗi file 1 lần gõ) nên gộp được qua `metadata.csv` chung schema.
- Luôn chia train/test **theo quả** (`fruit_id`), và nên đánh giá riêng trên dữ liệu tự thu (`source=recorded`) để biết model hoạt động thế nào với đúng phần cứng triển khai.
- Nếu kết quả trên dữ liệu tự thu kém hơn nhiều so với Mendeley: fine-tune CNN trên dữ liệu tự thu (giữ Mendeley làm pre-train), hoặc huấn luyện chỉ trên dữ liệu tự thu khi đã đủ ≥ 100 quả.
- Lưu thêm 1–2 s nền của mỗi phiên để có thể augment bằng nhiễu thật.

## 6. Checklist mỗi phiên

- [ ] Cùng mic, cùng gá, cùng khoảng cách, ghi vào `--notes`
- [ ] Nền yên tĩnh, RMS nền in ra < 0,01
- [ ] 9 bản ghi/quả, kiểm tra số dòng metadata
- [ ] Nhãn giấy `fNNN` trên quả
- [ ] Bổ và ghi `fruits.csv` cùng ngày
- [ ] Cập nhật nhãn vào `metadata.csv`, kiểm tra không còn `unknown`
- [ ] Backup thư mục phiên (copy sang máy khác hoặc cloud)
