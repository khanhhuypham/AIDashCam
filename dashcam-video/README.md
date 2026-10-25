# AI Dashcam: chạy logic cảnh báo trên video (`dashcam_video`)

Bản Python của logic cảnh báo trong app **iOS** và **Android**: **detect + tracking + cảnh báo va chạm (FCW) + cảnh báo lệch làn (LDW)** chạy trên video dashcam, và xuất **`DashcamConfig.json`** để hai app dùng đúng cùng một bộ ngưỡng.

Project này **không train mô hình**. Mô hình đến từ project **dashcam-train** đặt cạnh (`../dashcam-train`):

| File | Lấy từ dashcam-train | Dùng để |
|---|---|---|
| `outputs/train/yolo11n_bdd100k/weights/best.pt` | lệnh `train` | chạy detect trên video (`inference.weights`) |
| `outputs/coreml/manifest.json` | lệnh `export-coreml` | điền mục `models` trong `DashcamConfig.json` của iOS |
| `outputs/tflite/manifest.json` | lệnh `export-tflite` | điền mục `models` trong `DashcamConfig.json` của Android |

Việc trong file `Đồ_án_AI_Dashcam.xlsm` mà project này làm:

| Việc | Lệnh |
|---|---|
| CV04 Chạy thử trên video | `run` |
| CV13 Logic cảnh báo bằng Python, ngưỡng để trong config | `run` + `config.yaml` |
| App iOS / Android: dữ liệu JSON để test bản Swift / Kotlin | `run` (file `*_events.json`) + `export-app-config` |

---

## 1. Kiến trúc

```
dashcam-video/
├── config.yaml                 # đường dẫn mô hình + TOÀN BỘ ngưỡng cảnh báo
├── pyproject.toml              # thư viện cần cài + cấu hình pytest và ruff (định dạng code)
├── requirements.txt            # chỉ trỏ tới pyproject.toml (pip install -r requirements.txt)
├── dashcam_video/
│   ├── __main__.py             # python -m dashcam_video <command>
│   ├── domain/                 # (1) Không phụ thuộc thư viện ngoài
│   │   ├── enums.py            #     AlertLevel, LaneDepartureSide, EgoSpeedSource, Command...
│   │   └── entities.py         #     BoundingBox, Detection, FcwResult, LdwResult...
│   ├── config/                 # (2) Schema pydantic cho config.yaml (gốc là RootConfig) + loader
│   ├── core/                   # (3) Logic cảnh báo, chỉ dùng numpy, dễ chuyển sang Swift / Kotlin
│   │   ├── tracking.py         #     IouTracker
│   │   ├── geometry.py         #     CameraModel: ước lượng khoảng cách (vùng làn xe: InPathRoi.contains trong config/schema.py)
│   │   ├── fcw.py              #     chọn xe dẫn đầu, TTC, mức cảnh báo
│   │   ├── ldw.py              #     vị trí trong làn từ mặt nạ vạch làn
│   │   ├── debounce.py         #     chống nhấp nháy + thời gian chờ (dùng chung FCW/LDW)
│   │   ├── ego_speed.py        #     tốc độ xe (hằng số / CSV; GPS trên điện thoại)
│   │   └── class_catalog.py    #     tra lớp theo tên/alias
│   ├── perception/             # (4) Mô hình nhìn ảnh (torch)
│   │   ├── yolo_detector.py    #     YOLO -> danh sách Detection
│   │   └── lane_segmenter.py   #     TwinLiteNet -> mặt nạ vùng lái + vạch làn (phương án B)
│   ├── export/                 # (4) DashcamConfig.json cho app iOS và Android
│   ├── pipeline/               # (5) Ghép perception + core, chạy trên video
│   │   ├── dashcam_pipeline.py #     một bước AI trên một khung: detect -> track -> FCW (-> LDW). Phần app làm lại
│   │   ├── video_runner.py     #     đọc video, gọi pipeline theo ai_fps, ghi video debug
│   │   ├── events_writer.py    #     ghi <video>_events.json (dữ liệu test cho Swift / Kotlin)
│   │   ├── renderer.py         #     vẽ box, ROI, HUD, cảnh báo lên khung hình
│   │   └── frame_result.py     #     kết quả một khung AI
│   ├── cli/                    # (6) argparse, mỗi Command một handler
│   └── utils/                  #     logging, file, device
└── tests/                      # pytest cho config + core + pipeline (detector giả, không cần mô hình)
```

Quy tắc phụ thuộc: tầng dưới không import tầng trên.
`domain` ← `config` ← `core` ← `perception / export` ← `pipeline` ← `cli`.
Riêng `core/` không import torch, ultralytics hay OpenCV, nên mỗi file có thể viết lại 1-1 bằng Swift (iOS) và Kotlin (Android).
`DashcamPipeline` nhận detector và lane segmenter từ bên ngoài (bất kỳ thứ gì có hàm `detect` / `segment`): khi chạy thật thì `DashcamPipeline.from_config(...)` tạo YOLO và TwinLiteNet, khi test thì truyền detector giả.

### Bản đồ code

| Lệnh | Bắt đầu đọc từ | Code dùng thêm | Kết quả |
|---|---|---|---|
| `run` | `pipeline/video_runner.py` | `pipeline/dashcam_pipeline.py`, `perception/yolo_detector.py`, toàn bộ `core/`, `pipeline/renderer.py`, `pipeline/events_writer.py`, `perception/lane_segmenter.py` (khi bật LDW) | `outputs/video/` |
| `export-app-config` | `export/app_config.py` | | `outputs/ios/DashcamConfig.json`, `outputs/android/DashcamConfig.json` |

### Thư mục `outputs/` (không đưa vào Git)

```
outputs/
├── video/    # <video>_debug.mp4 + <video>_events.json
├── ios/      # DashcamConfig.json cho app iOS
└── android/  # DashcamConfig.json cho app Android
```

---

## 2. Cài đặt

Yêu cầu Python 3.11–3.12. Đặt project này **cạnh** dashcam-train (cùng thư mục cha), vì `config.yaml` trỏ tới `../dashcam-train/...`.

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. macOS/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

Lệnh này cài project ở chế độ sửa trực tiếp (editable) cùng các thư viện trong `pyproject.toml` và công cụ dev (`pytest`, `ruff`).

Kiểm tra cài đặt:

```bash
python -m pytest -q
```

Trước khi commit, cho `ruff` tự định dạng code và kiểm tra lỗi (cấu hình trong `pyproject.toml`, dòng tối đa 120 ký tự):

```bash
python -m ruff format .
```

```bash
python -m ruff check . --fix
```

Mọi lệnh đều nhận `--config <file>` (mặc định là `config.yaml`). Đường dẫn tương đối trong config được tính **từ thư mục chứa file config**.

---

## 3. Chạy logic cảnh báo trên video (CV04, CV13)

Đặt video vào `data/samples/dashcam.mp4` (hoặc sửa `video.source`), rồi chạy:

```bash
python -m dashcam_video run
```

- AI chạy ở `video.ai_fps` (mặc định 10 FPS) để giả lập điện thoại yếu (iPhone 8, Android tầm thấp).
- Tốc độ xe lấy từ `ego_speed:`. Có thể dùng hằng số, hoặc một file CSV `timestamp_s,speed_kmh` xuất từ GPS.
- Kết quả trong `outputs/video/`:
  - `<video>_debug.mp4`: box, ID track, vùng làn xe (hình thang), HUD khoảng cách/TTC/headway, banner cảnh báo.
  - `<video>_events.json`: đầu vào và đầu ra của logic ở từng khung AI, kèm toàn bộ ngưỡng. File này là **dữ liệu test cho bản Swift và Kotlin** (khóa `meta.ios_config` giữ tên cũ; ngưỡng trong đó dùng chung cho cả hai app).

**Logic FCW** (`core/fcw.py`):

1. Tracker IoU gán ID cho từng đối tượng. Track chỉ được dùng khi đã xuất hiện `min_hits` lần.
2. Xe dẫn đầu là đối tượng gần nhất có điểm giữa cạnh dưới box nằm trong hình thang `in_path_roi`.
3. **TTC = s / (ds/dt)**, với `s` là bề rộng (hoặc chiều cao) box, lấy bằng hồi quy tuyến tính trên `ttc_window_s` giây gần nhất. Cách này không cần hiệu chỉnh camera. Sau đó làm mượt bằng EMA.
4. Khoảng cách tính theo mô hình pinhole từ `reference_size_m` và `camera.horizontal_fov_deg`. Chỉ gần đúng. Headway = khoảng cách / tốc độ xe.
5. Mức cảnh báo: `CRITICAL` nếu TTC ≤ `ttc_critical_s`, `CAUTION` nếu TTC ≤ `ttc_caution_s` hoặc headway ≤ `headway_caution_s`. Dưới `min_ego_speed_kmh` thì không cảnh báo.
6. Chống nhấp nháy: cần `min_consecutive_frames` khung liên tiếp mới bật cảnh báo. Âm thanh chỉ phát lại sau `cooldown_s`, trừ khi mức cảnh báo tăng lên.

**Logic LDW** (`core/ldw.py`): ở mỗi dòng pixel gần đáy ảnh, lấy mép trong của vạch làn bên trái và bên phải tâm xe để suy ra tâm làn và bề rộng làn. Lấy trung vị qua các dòng. Độ lệch được tính là `(tâm xe − tâm làn) / (nửa bề rộng làn)`; nếu vượt `departure_threshold` liên tục thì cảnh báo trái/phải. Trên điện thoại, người lái tự xác nhận xi-nhan qua tham số `turn_signal_active`.

### Phương án B: bật LDW

LDW cần TwinLiteNet. Project dùng chung repo đã clone trong dashcam-train (`../dashcam-train/third_party/TwinLiteNet`, xem README của dashcam-train mục 4.2). Kiểm tra `twinlitenet.weights` trỏ đúng tới file `.pth`, đặt `ldw.enabled: true` rồi chạy `run` để xem mặt nạ vùng lái (xanh lá), vạch làn (đỏ) và cảnh báo lệch làn.

---

## 4. Config cho app iOS và Android

```bash
python -m dashcam_video export-app-config
```

Lệnh tạo một file `DashcamConfig.json` cho mỗi nền tảng trong `app_config_export.targets`:

| File | Mục `models` lấy từ |
|---|---|
| `outputs/ios/DashcamConfig.json` | `../dashcam-train/outputs/coreml/manifest.json` (Core ML) |
| `outputs/android/DashcamConfig.json` | `../dashcam-train/outputs/tflite/manifest.json` (TFLite) |

Hai file có **cùng cấu trúc và cùng ngưỡng** (camera, lớp, tracker, fcw, ldw), chỉ khác danh sách mô hình. Chưa chạy `export-tflite` thì `models` của Android để trống. Key để dạng `snake_case`: Swift decode bằng `JSONDecoder().keyDecodingStrategy = .convertFromSnakeCase`, Kotlin dùng naming strategy snake_case (kotlinx.serialization `JsonNamingStrategy.SnakeCase` hoặc `@SerialName`).

**Không đổi cấu trúc file này**: app iOS đã phát hành đang đọc nó. Test `tests/test_config.py` kiểm tra danh sách key của cả hai file.

---

## 5. Tích hợp vào app

### 5.1 iOS

1. Kéo `.mlpackage` (từ `../dashcam-train/outputs/coreml/`) và `DashcamConfig.json` vào Xcode target (deployment target iOS 16).
2. Chạy mô hình bằng Vision trên một **queue riêng**, tách khỏi luồng ghi video:
   ```swift
   let config = MLModelConfiguration()
   config.computeUnits = .all            // A11: GPU/CPU, A12+: ANE
   let model = try VNCoreMLModel(for: yolo11n_416_fp16(configuration: config).model)
   let request = VNCoreMLRequest(model: model) { request, _ in
       let observations = request.results as? [VNRecognizedObjectObservation] ?? []
       // boundingBox của Vision có gốc DƯỚI-trái: y_top = 1 - box.maxY
   }
   request.imageCropAndScaleOption = .scaleFit   // gần với letterbox lúc train
   ```
3. Viết lại `core/` bằng Swift theo đúng cấu trúc:

   | Python | Swift gợi ý |
   |---|---|
   | `domain/enums.py` | `enum AlertLevel: Int`, `enum LaneDepartureSide: String` |
   | `domain/entities.py` | `struct BoundingBox`, `struct Detection` |
   | `core/tracking.py` | `final class IouTracker` |
   | `core/debounce.py` | `final class AlertDebouncer<State>` |
   | `core/fcw.py`, `core/ldw.py` | `ForwardCollisionWarner`, `LaneDepartureWarner` |

4. Test Swift bằng `*_events.json`: đưa `detections` của từng frame vào `IouTracker` + `ForwardCollisionWarner` của Swift, rồi so `tracks` và `fcw` với giá trị Python đã ghi. Tracker xử lý hòa điểm theo thứ tự chỉ số, nên cả hai phía phải ra **cùng track ID**.

### 5.2 Android

1. Chép `.tflite` (từ `../dashcam-train/outputs/tflite/`) và `outputs/android/DashcamConfig.json` vào `assets/` của app.
2. Chạy mô hình bằng LiteRT (TFLite) với **GPU delegate** trên một luồng riêng, tách khỏi luồng ghi video (CameraX `ImageAnalysis` với `STRATEGY_KEEP_ONLY_LATEST`):
   - Input: ảnh RGB, letterbox về `input_width × input_height` (viền 114), chia 255. Đúng thứ tự trục ghi trong `output_layout` của `manifest.json`.
   - Output **chưa có NMS**: mỗi anchor gồm `cx, cy, w, h` chuẩn hóa 0..1 rồi điểm từng lớp. App tự lọc theo `inference.conf`, gộp box trùng theo `inference.iou`, giữ tối đa `inference.max_det` (đều có trong `DashcamConfig.json`).
   - Đổi tọa độ từ ảnh letterbox về khung hình gốc, **gốc trên-trái**, chuẩn hóa 0..1 như trong `core/`.
3. Viết lại `core/` bằng Kotlin theo đúng cấu trúc:

   | Python | Kotlin gợi ý |
   |---|---|
   | `domain/enums.py` | `enum class AlertLevel`, `enum class LaneDepartureSide` |
   | `domain/entities.py` | `data class BoundingBox`, `data class Detection` |
   | `core/tracking.py` | `class IouTracker` |
   | `core/debounce.py` | `class AlertDebouncer<S : Enum<S>>` |
   | `core/fcw.py`, `core/ldw.py` | `ForwardCollisionWarner`, `LaneDepartureWarner` |

4. Góc nhìn camera: máy Android khác nhau nhiều. Nên tính FOV ngang thật từ `CameraCharacteristics` (`LENS_INFO_AVAILABLE_FOCAL_LENGTHS` + `SENSOR_INFO_PHYSICAL_SIZE`) và chỉ dùng `camera.horizontal_fov_deg` khi không đọc được.
5. Test Kotlin bằng `*_events.json` giống bản Swift: cùng `detections` vào phải ra cùng `tracks` và `fcw`.

---

## 6. Cấu hình (`config.yaml`)

| Nhóm | Nội dung |
|---|---|
| `project` | `device` (`auto`/`cpu`/`0`/`mps`), `log_level` |
| `classes` | tên lớp + alias, nhóm (`vehicle`/`two_wheeler`/`pedestrian`), kích thước thật |
| `camera`, `ego_speed` | FOV ngang của camera, nguồn tốc độ xe (`constant`/`csv`/`none`) |
| `inference`, `video` | mô hình (từ dashcam-train), video đầu vào/đầu ra |
| `tracker`, `fcw` | tracking + cảnh báo va chạm |
| `twinlitenet`, `ldw` | phương án B |
| `app_config_export` | mỗi nền tảng: `manifest.json` của dashcam-train + nơi ghi `DashcamConfig.json` |

Mọi trường đều được kiểm tra bởi `dashcam_video/config/schema.py`: sai kiểu, sai giá trị enum, key thừa, hay `ttc_critical_s ≥ ttc_caution_s` đều bị báo lỗi ngay lúc đọc.

---

## 7. Giới hạn và lưu ý

- Khoảng cách từ một camera **chỉ gần đúng**. Phải hiệu chỉnh `camera.horizontal_fov_deg` và `in_path_roi` theo góc gắn điện thoại thực tế. TTC dựa trên tốc độ phóng to của box nên ít bị ảnh hưởng hơn.
- App chỉ **hỗ trợ cảnh báo**, không thay thế sự chú ý của người lái. Trên iOS không chạy nền được; iPhone 8 tối đa iOS 16, nên không dùng API iOS 17+. Trên Android, chạy nền cần foreground service và bị giới hạn khác nhau theo hãng.
- Các ngưỡng mặc định trong `config.yaml` là điểm khởi đầu hợp lý, **chưa được hiệu chỉnh trên đường Việt Nam**. Cần chỉnh sau khi chạy thử trên xe và đếm cảnh báo sai mỗi giờ.
- Không đưa `data/`, `outputs/` vào Git (đã có trong `.gitignore`).
