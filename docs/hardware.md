# Phần cứng: ESP32-S3 vs Raspberry Pi Zero 2 W, mic I2S, piezo, solenoid gõ

Model đích: CNN log-mel trong `src/train_cnn.py`, ~24 k tham số, file TFLite int8 ~30 KB,
tensor arena ước ~55 KB, đầu vào 40 mel × 51 khung (0,5 s @ 16 kHz, FFT 512, hop 160).

## 1. So sánh nền tảng

| Tiêu chí | ESP32-S3 (vd. ESP32-S3-DevKitC-1 N8R8) | Raspberry Pi Zero 2 W |
|---|---|---|
| CPU | 2× Xtensa LX7 @ 240 MHz, có lệnh vector (ESP-NN tăng tốc int8) | 4× Cortex-A53 @ 1 GHz |
| RAM | 512 KB SRAM nội + 8 MB PSRAM (N8R8) | 512 MB LPDDR2 |
| Flash | 8–16 MB | thẻ microSD |
| Model 30 KB + arena 55 KB | Vừa trong SRAM nội, không cần PSRAM | Không đáng kể |
| Front-end mel (FFT 512, 51 khung) | ESP-DSP FFT float/int16: ~1–2 ms / khung → ~50–80 ms cho 0,5 s nếu tính sau; ~0 nếu tính streaming theo từng khung 10 ms | < 10 ms với numpy/librosa, < 2 ms với C |
| Suy luận CNN int8 | TFLite Micro + ESP-NN: ước **20–60 ms** cho model này (~1,5 M MAC) | tflite-runtime (XNNPACK): **< 5 ms** |
| Latency tổng gõ → kết quả | ~100–150 ms | ~20–50 ms (sau khi Python đã nạp) |
| Thời gian khởi động | < 1 s | 20–40 s boot Linux |
| Công suất | 0,1–0,3 W khi chạy; deep sleep ~10 µA → chạy pin 18650 nhiều ngày | 0,6–1,2 W; không có sleep thực sự; pin 18650 ~ vài giờ |
| Mic I2S | I2S cứng, DMA, dễ cấu hình (ESP-IDF `i2s_std`) | I2S qua overlay `googlevoicehat-soundcard` / `i2s-mems`, cần cấu hình dtoverlay |
| ADC cho piezo | ADC 12-bit nội (ồn, nhưng đủ cho piezo contact) | **Không có ADC** → cần ADS1115/MCP3008 hoặc ADC I2S ngoài |
| Giải pháp phần mềm | C/C++ (ESP-IDF hoặc Arduino), TFLite Micro | Python đầy đủ: chạy thẳng `src/record.py`, `tflite-runtime` |
| Giá tham khảo (2025–2026) | 150–250 k VND (dev kit) | 400–500 k VND + thẻ SD + nguồn |
| Thiết bị cầm tay sản phẩm | **Phù hợp** | Nặng, nóng, boot chậm |
| Giai đoạn thu dữ liệu / prototype | Phải viết firmware ghi wav ra SD/UART | **Phù hợp**: dùng `record.py`, lưu wav, gán nhãn qua bàn phím/SSH |

**Khuyến nghị**

- **Thu dữ liệu và thử nghiệm**: Pi Zero 2 W + INMP441 (hoặc mic USB), chạy `python -m src.record`. Cùng mic và cùng gá sẽ dùng cho sản phẩm để tránh lệch miền (domain shift).
- **Sản phẩm cầm tay**: ESP32-S3 + INMP441 + solenoid, model `models/coconut_cnn_int8.cc` nhúng vào firmware TFLite Micro. Nếu sau này cần ghi wav để thu thêm dữ liệu trên chính thiết bị, ghi ra thẻ SD qua SPI.
- Nếu muốn một phần cứng duy nhất cho cả hai giai đoạn: ESP32-S3 với firmware có chế độ "ghi wav ra SD + UART log", chấp nhận viết thêm firmware.

## 2. Mic I2S MEMS

### INMP441 (khuyên dùng)

Khung I2S Philips chuẩn, 24-bit trong slot 32-bit, SNR 61 dB, đáp tuyến phẳng 60 Hz–15 kHz. Dải cộng hưởng tiếng gõ dừa (≈ 300 Hz–4 kHz) nằm trọn trong vùng phẳng.

```
INMP441      ESP32-S3 (GPIO tuỳ chọn, ví dụ)      Pi Zero 2 W (header 40 chân)
VDD   -----  3V3                                  3V3 (chân 1)
GND   -----  GND                                  GND (chân 6)
SCK   -----  GPIO 5  (I2S BCLK)                   GPIO 18 / chân 12 (PCM_CLK)
WS    -----  GPIO 6  (I2S WS / LRCLK)             GPIO 19 / chân 35 (PCM_FS)
SD    -----  GPIO 4  (I2S DIN)                    GPIO 20 / chân 38 (PCM_DIN)
L/R   -----  GND (kênh trái)                      GND
```

- ESP-IDF: `i2s_std` mode, slot 32-bit, mono trái, 16 kHz; lấy 24 bit có nghĩa bằng cách `sample >> 8` (hoặc `>> 14` để ra 16-bit có headroom).
- Pi: `dtoverlay=googlevoicehat-soundcard` (Raspberry Pi OS Bookworm) rồi `arecord -D plughw:1 -c1 -r16000 -f S32_LE`; `sounddevice` sẽ thấy card này. Dây tín hiệu ≤ 10 cm.

### SPH0645LM4H

Cũng I2S, nhưng dữ liệu lệch 1 bit so với chuẩn Philips (MSB xuất ngay sau cạnh WS) và chỉ 18 bit có nghĩa. Trên ESP32-S3 cần bật `bit_shift` / dùng cấu hình "Philips với sửa lệch", trên Pi overlay `googlevoicehat-soundcard` vốn được làm cho SPH0645 nên hoạt động ngay. Nếu đã có sẵn thì dùng được; mua mới thì chọn INMP441.

### Gá mic

Mic cách vỏ dừa **2–5 cm**, hướng vào điểm gõ, cố định bằng gá in 3D cùng khung với solenoid để khoảng cách và góc không đổi giữa các quả (xem `docs/data_collection_protocol.md`). Bọc mic bằng xốp mỏng để giảm gió, không che kín.

## 3. Piezo contact sensor (thay thế hoặc bổ sung mic)

Đĩa piezo 20–27 mm dán hoặc ép sát vỏ dừa cạnh điểm gõ, đo rung động cơ học thay vì âm thanh không khí.

| | Mic I2S | Piezo tiếp xúc |
|---|---|---|
| Nhiễu môi trường (chợ, quạt, người nói) | Nhạy | Gần như miễn nhiễm |
| Cộng hưởng khoang nước / cùi | Bắt tốt (đây là đặc trưng chính của độ chín) | Bắt rung động vỏ + một phần khoang; phổ khác mic |
| Lắp đặt | Không chạm quả | Phải ép lên vỏ, lực ép ảnh hưởng tín hiệu |
| Tương thích dataset Mendeley | Cùng loại (mic) | Khác miền, phải thu lại hoàn toàn |

Mạch điều hoà tín hiệu (đầu vào ADC 0–3,3 V):

```
piezo (+) ---+----- R1 1 MΩ -----+---- Vbias (1,65 V, từ chia áp 2× 10 kΩ + tụ 10 µF)
             |                   |
             |            D1, D2 schottky (BAT54S) kẹp về GND / 3V3
             |                   |
             +---- C1 100 nF ----+----> ADC (ESP32-S3 GPIO1..10 ADC1) hoặc ADS1115/MCP3008 trên Pi
piezo (-) --- GND
```

- R1 tạo đường xả, C1 chặn DC, Vbias đẩy tín hiệu về giữa thang ADC, diode kẹp bảo vệ khi gõ mạnh (piezo có thể sinh > 20 V).
- Nếu tín hiệu quá nhỏ: thêm op-amp rail-to-rail (MCP6002) khuếch đại ×5–×10 trước ADC.
- Lấy mẫu 16 kHz bằng ADC liên tục (ESP32-S3: `adc_continuous` + DMA). Pi Zero không có ADC: dùng ADS1115 (chỉ 860 SPS, **quá chậm** cho 16 kHz) → thực tế phải dùng MCP3008 qua SPI (~100 kSPS, 10-bit) hoặc đưa piezo vào mạch mic USB/I2S ADC (PCM1808). Vì vậy piezo hợp với ESP32-S3 hơn.

Có thể chạy **song song mic + piezo** (hai kênh), gộp đặc trưng; nhưng hãy bắt đầu với mic để dùng được dataset Mendeley.

## 4. Solenoid gõ lực cố định

Mendeley dùng cánh tay servo đầu cao su cứng. Solenoid đẩy (push-type) 12 V, hành trình 10 mm, lực 3–5 N là đủ; đầu gõ gắn bi nhựa POM hoặc cao su cứng Ø 10 mm để có xung ngắn, lặp lại.

```
            +12 V (nguồn riêng, 1–2 A, GND chung với MCU)
              |
         [Solenoid]  <--- diode flyback 1N5819/1N4007 song song, cực âm về +12 V
              |
            D (drain)
GPIO ---R 100Ω--- G   Q1: MOSFET logic-level IRLZ44N (hoặc AO3400 nếu < 1 A)
       R 10 kΩ về GND (pull-down giữ tắt khi reset)
            S (source)
              |
             GND
```

- Xung điều khiển **20–50 ms** (thử để đầu gõ chạm và bật ra, không đè lên vỏ). Nghỉ ≥ 500 ms giữa hai lần gõ cho lò xo hồi.
- Lực gõ cố định = điện áp cố định + hành trình cố định + khoảng cách đầu gõ → vỏ cố định (gá đỡ quả hình chữ V để quả luôn tựa cùng vị trí).
- **Đồng bộ với ghi âm**: MCU phát xung solenoid rồi lấy `t0`; cửa sổ phân loại lấy `[t0 - 20 ms, t0 + 480 ms]` (bỏ qua bước dò onset) hoặc vẫn dò onset trong 100 ms đầu để bù độ trễ cơ khí. Tiếng click của solenoid chính nó lẫn vào đầu cửa sổ: giữ nguyên trong cả lúc thu data lẫn lúc chạy để model học cùng điều kiện.
- Nguồn 12 V tách riêng khỏi 3V3 của mic; tụ 470 µF trên đường 12 V để xung không kéo sụt nguồn gây nhiễu vào mic.

## 5. Bộ phần cứng gợi ý

**Bộ thu dữ liệu (prototype)**: Pi Zero 2 W + thẻ 32 GB + INMP441 + gá mic + solenoid 12 V + IRLZ44N + nguồn 12 V 2 A + pin dự phòng USB; điều khiển solenoid qua GPIO của Pi, chạy `src/record.py` bản sửa có trigger solenoid.

**Sản phẩm**: ESP32-S3 N8R8 + INMP441 + solenoid + MOSFET + pin 18650 + mạch sạc TP4056 + boost 12 V (MT3608) cho solenoid + 3 LED (non/vừa/già) hoặc OLED 0,96". Firmware: ESP-IDF, `i2s_std` → vòng đệm 0,5 s → mel (ESP-DSP) → TFLite Micro (`coconut_cnn_int8.cc`) → LED.
