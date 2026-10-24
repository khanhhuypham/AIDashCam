# Kế hoạch: tách `run-video` ra project riêng

> **Trạng thái: ĐÃ LÀM giai đoạn 1–3 (23/10/2026).** Project mới nằm ở `../dashcam-video`. Giai đoạn 0 (hoàn thiện mô hình) vẫn đang tiếp tục; khi `best.pt` mới ra, project Video tự dùng nó vì đọc thẳng đường dẫn. Xem mục 7 để biết chỗ nào làm khác kế hoạch.

---

## 1. Mục tiêu

Hiện tại `AIDashCam` vừa train mô hình, vừa chạy logic cảnh báo trên video. Sau khi tách:

| Project | Làm gì | Không làm gì |
|---|---|---|
| **AIDashCam** (giữ nguyên) | dữ liệu → train → đánh giá → xuất Core ML | không chạy video, không chứa logic cảnh báo |
| **AIDashCam-Video** (mới) | đọc mô hình đã train, chạy detect + tracking + FCW (+ LDW) trên video, xuất `DashcamConfig.json` cho app | không train, không xử lý dataset |

Lợi ích:
- Mỗi project nhỏ hơn, đọc một project chỉ cần hiểu một việc.
- Project mới chính là "bản Python của app iOS": cùng logic, cùng ngưỡng, cùng file JSON. Dễ dùng để test bản Swift.

---

## 2. Ranh giới giữa hai project

```
            AIDashCam (ML)                                   AIDashCam-Video
  ┌───────────────────────────────┐                ┌───────────────────────────────────┐
  │ data/ → train → evaluate      │                │ YoloDetector (đọc best.pt)        │
  │                               │   best.pt      │ core/: tracking, FCW, LDW         │
  │ outputs/train/.../best.pt ────┼───────────────▶│ pipeline/: chạy video, vẽ overlay │
  │                               │                │                                   │
  │ export-coreml                 │  manifest.json │ export-ios-config                 │
  │ outputs/coreml/*.mlpackage    │───────────────▶│   → DashcamConfig.json            │
  │ outputs/coreml/manifest.json  │                │ outputs/video/*_debug.mp4         │
  └───────────────────────────────┘                │ outputs/video/*_events.json       │
                  │                                └───────────────────────────────────┘
                  │ .mlpackage                                      │ DashcamConfig.json
                  ▼                                                 ▼
                              App iOS (Swift, đã có trên App Store)
```

Hai project chỉ nói chuyện với nhau qua **file**, không import code của nhau:

| File | Do ai tạo | Ai đọc |
|---|---|---|
| `best.pt` | AIDashCam (`train`) | AIDashCam-Video (chạy video trên máy tính) |
| `manifest.json` | AIDashCam (`export-coreml`) | AIDashCam-Video (điền mục `models` trong `DashcamConfig.json`) |
| `*.mlpackage` | AIDashCam (`export-coreml`) | App iOS |
| `DashcamConfig.json` | AIDashCam-Video (`export-ios-config`) | App iOS |
| `*_events.json` | AIDashCam-Video (`run`) | test của bản Swift |

**Ai giữ ngưỡng cảnh báo?** Project Video. Các ngưỡng `tracker`, `fcw`, `ldw`, `camera`, `ego_speed` và kích thước thật của từng lớp là logic cảnh báo, không liên quan đến train. Đặt chúng cùng chỗ với code dùng chúng thì chỉnh ngưỡng xong chạy video thử ngay, không phải sửa project ML.

---

## 3. Code nào đi đâu

### 3.1 Chuyển sang AIDashCam-Video

| Hiện tại (`dashcam_ml/`) | Số dòng | Ghi chú |
|---|---|---|
| `pipeline/video_runner.py`, `renderer.py`, `frame_result.py` | ~410 | |
| `core/tracking.py`, `fcw.py`, `ldw.py`, `debounce.py`, `geometry.py`, `ego_speed.py` | ~570 | |
| `models/yolo_detector.py` → chỉ class `YoloDetector` | ~50 | hai hàm `to_ultralytics_imgsz`, `ultralytics_precision_kwargs` ở lại ML (xem 3.3) |
| `models/twinlitenet.py` → phần `TwinLiteNetSegmenter` | ~50 | chỉ khi làm phương án B (giai đoạn 4) |
| `export/ios_config.py` | ~40 | |
| `tests/test_fcw.py`, `test_ldw.py`, `test_tracking.py` | | |
| Phần config: `camera`, `ego_speed`, `inference`, `video`, `tracker`, `fcw`, `ldw`, `twinlitenet` (phần chạy), `ios_config_export`, `classes` (kích thước thật) | | thành `config.yaml` của project mới |

Tổng cộng khoảng **1.100 dòng** code chuyển đi. Project ML còn khoảng **2.500 dòng**.

### 3.2 Chép sang (cả hai project đều cần)

| Code | Lý do |
|---|---|
| `domain/entities.py`, `domain/enums.py` | chỉ chép phần project mới dùng (`BoundingBox`, `Detection`, `FcwResult`, `LdwResult`, `AlertLevel`...) |
| `core/class_catalog.py` | ML dùng để đổi nhãn COCO khi đánh giá; Video dùng để tra kích thước thật |
| `config/loader.py`, phần `StrictModel` / `ProjectPath` trong `schema.py` | mỗi project có schema riêng cho `config.yaml` của nó |
| `utils/` | logging, file, chọn device |

Không làm thư viện dùng chung (shared package): code chung chỉ vài trăm dòng và hầu như không đổi, chép sang đơn giản hơn nhiều cho dự án một người làm.

### 3.3 Ở lại AIDashCam và cần sửa

| Việc | Lý do |
|---|---|
| Chuyển `to_ultralytics_imgsz`, `ultralytics_precision_kwargs` sang `models/ultralytics_utils.py` | `evaluate` và `export-coreml` vẫn dùng |
| `classes` trong `config.yaml` chỉ còn `name` + `aliases` | `evaluate` cần alias để so mô hình COCO; kích thước thật đã chuyển sang Video |
| Bỏ lệnh `run-video`, `export-ios-config` khỏi `Command` và `cli/commands.py` | |
| `export-twinlitenet` vẫn ở lại (là xuất Core ML), giữ phần load mô hình trong `models/twinlitenet.py` | |
| Sửa README: bỏ mục 3.5, 4.4, phần FCW/LDW ở mục 5, thêm link sang project Video | |

---

## 4. Project mới trông thế nào

```
AIDashCam-Video/
├── config.yaml                  # đường dẫn mô hình + toàn bộ ngưỡng cảnh báo
├── requirements.txt             # ultralytics, opencv, numpy, pydantic (không cần coremltools)
├── dashcam_video/
│   ├── __main__.py              # python -m dashcam_video <lệnh>
│   ├── domain/                  # entities, enums (chép từ ML)
│   ├── config/                  # schema + loader
│   ├── core/                    # tracking, fcw, ldw, debounce, geometry, ego_speed, class_catalog
│   ├── detector/                # YoloDetector (+ TwinLiteNetSegmenter khi làm phương án B)
│   ├── pipeline/                # video_runner, renderer, frame_result
│   ├── export/                  # ios_config
│   └── cli/
├── tests/                       # test_fcw, test_ldw, test_tracking, test_config
└── outputs/video/               # không đưa vào Git
```

Hai lệnh:

```bash
python -m dashcam_video run                 # = run-video hiện tại
python -m dashcam_video export-ios-config   # = export-ios-config hiện tại
```

Đường dẫn tới mô hình trong `config.yaml` của project mới:

```yaml
inference:
    weights: ../dashcam-train/outputs/train/yolo11n_bdd100k/weights/best.pt
ios_config_export:
    models_manifest: ../dashcam-train/outputs/coreml/manifest.json
```

Hai project đặt cạnh nhau trong `D:\huy\project\AI\`, nên chỉ cần đường dẫn tương đối, không phải chép mô hình qua lại.

---

## 5. Các bước thực hiện

### Giai đoạn 0: hoàn thiện mô hình (đang làm)

Không động vào code video. Xong khi:
- Đã chọn được `best.pt` cuối cùng (đã train đủ, điểm `evaluate` không tăng thêm).
- Đã chọn kích thước ảnh / profile Core ML sẽ dùng trên iPhone.

### Giai đoạn 1: tạo project Video, chạy song song với bản cũ

1. **Lưu kết quả chuẩn trước khi tách:** chạy `run-video` và `export-ios-config` trong AIDashCam, chép `outputs/video/dashcam_events.json` và `outputs/coreml/DashcamConfig.json` ra làm "đáp án".
2. Tạo repo `AIDashCam-Video`, chép code theo mục 3.1 và 3.2, đổi tên import `dashcam_ml` → `dashcam_video`.
3. Viết `config.yaml` mới từ các phần config tương ứng.
4. Chạy test: `python -m pytest -q`.

Xong khi: chạy `python -m dashcam_video run` trên cùng video cho ra `dashcam_events.json` **giống hệt đáp án** (cùng track ID, cùng TTC, cùng mức cảnh báo).

### Giai đoạn 2: chuyển `export-ios-config`

1. Chạy `python -m dashcam_video export-ios-config`.
2. So với `DashcamConfig.json` đáp án.

Xong khi: hai file JSON giống nhau. **Không được đổi cấu trúc file này**, vì app iOS đã phát hành đang đọc nó.

### Giai đoạn 3: dọn project ML

Làm theo mục 3.3, rồi xoá code đã chuyển (mục 3.1) khỏi `dashcam_ml/`.

Xong khi: `python -m pytest -q` qua, và `train`, `evaluate`, `export-coreml` vẫn chạy như trước.

### Giai đoạn 4 (khi cần): LDW / TwinLiteNet

Chỉ làm khi quyết định dùng phương án B. Chuyển `TwinLiteNetSegmenter` sang `detector/` của project Video, bật `ldw.enabled`.

---

## 6. Rủi ro và cách tránh

| Rủi ro | Cách tránh |
|---|---|
| Kết quả cảnh báo bị đổi khi chép code | So `dashcam_events.json` với đáp án ở giai đoạn 1 |
| App iOS đọc sai `DashcamConfig.json` | Giữ nguyên cấu trúc, so với đáp án ở giai đoạn 2 |
| Thứ tự / tên lớp lệch nhau giữa hai project | Project Video lấy tên lớp từ chính `best.pt` (`model.names`), chỉ tra kích thước thật theo tên |
| Đổi chỗ thư mục làm hỏng đường dẫn tới `best.pt` | Đường dẫn để trong `config.yaml`, project Video báo lỗi rõ khi không tìm thấy file |
| Code chung (`domain/`, `class_catalog`) bị sửa ở một bên mà quên bên kia | Code này gần như không đổi; nếu phải sửa thì sửa cả hai và ghi vào commit |

---

## 7. Kết quả thực hiện (23/10/2026)

| Kiểm tra | Kết quả |
|---|---|
| `dashcam_events.json` từ `python -m dashcam_video run` so với bản cũ (`run-video`) | giống hệt: 351 khung AI, cùng track ID, cùng 2 cảnh báo FCW |
| `DashcamConfig.json` từ `python -m dashcam_video export-ios-config` so với bản cũ | giống hệt, kể cả 4 mô hình đọc từ `manifest.json` |
| Test AIDashCam-Video | 19 test qua |
| Test AIDashCam sau khi dọn | 14 test qua; `--help` chỉ còn lệnh dữ liệu, train, đánh giá, Core ML |

Chỗ làm khác kế hoạch:

- **TwinLiteNet đã chuyển luôn** (không đợi giai đoạn 4): `detector/twinlitenet.py` của project Video giữ phần chạy trên video, còn `models/twinlitenet.py` của AIDashCam chỉ giữ phần load + đầu ra xác suất để xuất Core ML. LDW vẫn tắt (`ldw.enabled: false`). Project Video dùng chung repo TwinLiteNet clone trong `../dashcam-train/third_party/`.
- Tên key trong config: `inference.weights` (giữ tên cũ) và `ios_config_export.models_manifest` thay cho `model.weights` / `model.manifest`.
- `DashcamConfig.json` giờ ghi ở `AIDashCam-Video/outputs/ios/`. File cũ `AIDashCam/outputs/coreml/DashcamConfig.json` (do Colab tạo) không còn được cập nhật.
- Video mẫu `data/samples/dashcam.mp4` và kết quả `outputs/video/` đã chuyển sang project Video.
- Colab notebook không còn gọi `export-ios-config`.
