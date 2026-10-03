# AI Dashcam iOS: phần Python (`dashcam_ml`)

Code Python để **chuẩn bị dữ liệu, train/đánh giá, thử logic cảnh báo trên video và xuất mô hình Core ML** cho app dashcam iOS (máy tối thiểu iPhone 8, iOS 16).

Nội dung dựa trên file `Đồ_án_AI_Dashcam.xlsm`:

| Phương án (sheet *So sánh đề tài*) | Trạng thái trong code |
|---|---|
| **A. Cảnh báo va chạm (FCW)**: YOLO11n | Đầy đủ: data, train, evaluate, tracking + TTC, xuất Core ML, kiểm tra |
| **B. Cảnh báo lệch làn (LDW)**: TwinLiteNet | Logic LDW + chạy PyTorch + xuất Core ML (cần clone repo tác giả) |
| C. Đa nhiệm: YOLOP | Chưa làm (nặng nhất, rủi ro cao trên iPhone 8) |

Các việc trong sheet *Công việc cần làm* mà code này làm được:

| Việc | Lệnh |
|---|---|
| CV02 Tải và xem BDD100K | `prepare-bdd100k` |
| CV04 Chạy thử trên video | `run-video` |
| CV08 Trích khung hình, lọc ảnh mờ/trùng | `extract-frames` |
| CV09 Chia dữ liệu theo từng video | `split-dataset` |
| CV10/CV11 Đánh giá trọng số có sẵn, so kích thước ảnh | `evaluate` |
| CV12 Fine-tune | `train` |
| CV13 Logic cảnh báo bằng Python, ngưỡng để trong config | `run-video` + `config.yaml` |
| CV14 Xuất Core ML và so với PyTorch | `export-coreml`, `export-twinlitenet`, `verify-coreml` |
| App iOS (sau pipeline): dữ liệu JSON để test bản Swift | `run-video` (file `*_events.json`) + `export-ios-config` |

---

## 1. Kiến trúc

```
AIDashCam/
├── config.yaml                 # TOÀN BỘ cấu hình (đường dẫn, ngưỡng, profile export)
├── requirements.txt
├── dashcam_ml/
│   ├── __main__.py             # python -m dashcam_ml <command>
│   ├── domain/                 # (1) Không phụ thuộc thư viện ngoài
│   │   ├── enums.py            #     mọi Enum: AlertLevel, LaneDepartureSide, CopyMode, WeightPrecision, Command...
│   │   └── entities.py         #     BoundingBox, Detection, FcwResult, LdwResult...
│   ├── config/                 # (2) Schema pydantic cho config.yaml + loader
│   ├── core/                   # (3) Logic cảnh báo, chỉ dùng numpy, dễ chuyển sang Swift
│   │   ├── tracking.py         #     IouTracker
│   │   ├── geometry.py         #     CameraModel, ước lượng khoảng cách, ROI làn xe
│   │   ├── fcw.py              #     chọn xe dẫn đầu, TTC, mức cảnh báo
│   │   ├── ldw.py              #     vị trí trong làn từ mặt nạ vạch làn
│   │   ├── debounce.py         #     chống nhấp nháy + thời gian chờ (dùng chung FCW/LDW)
│   │   ├── ego_speed.py        #     tốc độ xe (hằng số / CSV; GPS trên iPhone)
│   │   └── class_catalog.py    #     tra lớp theo tên/alias
│   ├── models/                 # (4) Adapter: Ultralytics YOLO, TwinLiteNet (torch)
│   ├── data/                   # (4) BDD100K -> YOLO, trích khung hình, chia tập, remap lớp
│   ├── training/               # (4) train, evaluate
│   ├── export/                 # (4) Core ML, manifest, kiểm tra, JSON cho iOS
│   ├── pipeline/               # (5) Ghép models + core chạy trên video, vẽ overlay
│   ├── cli/                    # (6) argparse, mỗi Command một handler
│   └── utils/                  #     logging, file, device/platform
└── tests/                      # pytest cho config + core + data
```

Quy tắc phụ thuộc: tầng dưới không import tầng trên.
`domain` ← `config` ← `core` ← `models / data / training / export` ← `pipeline` ← `cli`.
Riêng `core/` không import torch, ultralytics hay OpenCV, nên mỗi file có thể viết lại 1-1 bằng Swift.

### Quy ước code

- Mọi hàm, biến, thuộc tính đều có **kiểu dữ liệu**. Hàm của project đặt `*` trong chữ ký để **bắt buộc gọi bằng named argument**, ví dụ `load_config(config_path=...)`.
- Mọi tập giá trị cố định đều là **`Enum`** (`StrEnum`/`IntEnum`) trong `domain/enums.py`. Pydantic chuyển chuỗi trong YAML sang enum, nên giá trị sai bị báo lỗi ngay lúc đọc config.
- Không viết cứng ngưỡng hay đường dẫn trong code. Tất cả nằm trong `config.yaml`.
- Tọa độ box luôn **chuẩn hóa 0..1, gốc trên-trái**, không phụ thuộc độ phân giải, giống cách dùng với Vision trên iOS.

---

## 2. Cài đặt

Yêu cầu Python 3.10–3.12.

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. macOS/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

- **GPU NVIDIA**: cài `torch` bản CUDA theo https://pytorch.org/get-started/locally/ trước, sau đó mới chạy lệnh trên.
- **Core ML** (`coremltools`) **không có bản Windows**. Các lệnh `export-*` phải chạy trên macOS, Linux, WSL2 hoặc Google Colab. `verify-coreml` chỉ chạy được trên **macOS**, vì chỉ máy Apple mới chạy được mô hình Core ML.
- Trọng số chính thức (`yolo11n.pt`) được Ultralytics **tự tải về** thư mục `weights/` khi dùng lần đầu.

Kiểm tra cài đặt:

```bash
python -m pytest -q
```

```bash
python -m dashcam_ml --help
```

Mọi lệnh đều nhận `--config <file>` (mặc định là `config.yaml`). Đường dẫn tương đối trong config được tính **từ thư mục chứa file config**.

---

## 3. Quy trình làm việc

### 3.1 Dữ liệu BDD100K (CV02)

Dự án dùng bản **Dataset Ninja** (định dạng Supervisely), tải tại https://datasetninja.com/bdd100k#download. Giải nén rồi đổi tên thư mục thành `data/bdd100k/`:

```
data/bdd100k/
├── meta.json
├── train/img/*.jpg   train/ann/*.jpg.json
├── val/img/*.jpg     val/ann/*.jpg.json
└── test/...                  # không có nhãn, không được chuyển
```

Bản mẫu (~800 ảnh có nhãn) và bản đầy đủ (70.000 train / 10.000 val / 20.000 test) có **cùng cấu trúc**, nên cả hai đều đặt ở `data/bdd100k/`. Muốn đổi bộ dữ liệu thì chỉ cần **thay thư mục** `data/bdd100k/`, không phải sửa config. Hiện `data/bdd100k/` đang chứa bản mẫu.

Cấu hình nằm trong phần `bdd100k:` của `config.yaml`:

| Key | Ý nghĩa |
|---|---|
| `dataset_root` | thư mục dữ liệu Dataset Ninja (mặc định `data/bdd100k`) |
| `output_dir` | nơi ghi dataset YOLO (mặc định `data/yolo_bdd100k`) |
| `class_names` | danh sách lớp, **thứ tự = class id** trong nhãn YOLO |
| `category_aliases` | đổi tên lớp của Dataset Ninja sang `class_names`: `person → pedestrian`, `bike → bicycle`, `motor → motorcycle` |
| `copy_mode` | `hardlink` (không tốn ổ, cần cùng ổ đĩa) hoặc `copy` |
| `max_images_per_split` | số file nhãn tối đa đọc mỗi split; `null` = dùng hết |
| `condition_splits` | các tập val theo tag ảnh, ví dụ `night: {timeofday: night}` |

- Chỉ lấy object dạng `rectangle`; `polygon` (vùng lái xe) và `line` (vạch làn) bị bỏ qua.
- Box có tên lớp không nằm trong `class_names` (sau khi đổi alias) bị bỏ và được đếm vào `unknown-category boxes` trong log.
- Kích thước ảnh lấy từ trường `size` của từng file nhãn (bản BDD100K là 1280×720).
- Tag ảnh (`timeofday`, `weather`, `scene`) được dùng cho các tập theo điều kiện.
- Thiếu thư mục `img/` hoặc `ann/` của `train`/`val` thì lệnh báo lỗi ngay, không tạo dataset rỗng.
- Bản mẫu chỉ đủ để chạy thử pipeline (tập `val_rainy` chỉ có 6 ảnh).
- Dữ liệu dùng theo điều khoản trong `LICENSE.md` đi kèm: được dùng cho giáo dục, nghiên cứu, phi lợi nhuận.

```bash
python -m dashcam_ml prepare-bdd100k
```

Kết quả nằm trong `data/yolo_bdd100k/` (ảnh được hardlink nên không tốn thêm dung lượng ổ) cùng file `dataset.yaml`. Chạy lại lệnh sẽ ghi đè ảnh và nhãn cùng tên; khi thay dữ liệu trong `data/bdd100k/` thì nên xóa `data/yolo_bdd100k/` trước để không lẫn file cũ. Lệnh cũng tạo thêm các tập `val_night`, `val_rainy`, `val_daytime` để thử điều kiện khó (CV15). Muốn train thử nhanh trên bản đầy đủ thì đặt `max_images_per_split: 500`: chỉ 500 file nhãn đầu tiên (theo tên) của mỗi split được đọc, nên không phải chờ đọc hết 70.000 file.

### 3.2 Dữ liệu tự quay ở Việt Nam (CV07–CV09)

1. Chép video vào `data/vn_videos/`.
2. Trích khung hình. Lệnh bỏ ảnh mờ (`blur_threshold`) và ảnh gần trùng nhau (`duplicate_threshold`):
   ```bash
   python -m dashcam_ml extract-frames
   ```
   Ảnh được đặt tên `<tên_video>__f000123.jpg`.
3. Gán nhãn định dạng YOLO (CVAT, Label Studio, Roboflow...) dùng **đúng thứ tự lớp** trong `split.class_names`. Đặt ảnh vào `data/vn_labeled/images/`, nhãn `.txt` vào `data/vn_labeled/labels/`.
4. Chia tập **theo từng video**, để khung hình của cùng một video không nằm ở hai tập:
   ```bash
   python -m dashcam_ml split-dataset
   ```
   Lệnh tạo `data/vn_split/dataset.yaml` và `split_report.json` (video nào thuộc tập nào).

### 3.3 Đánh giá mô hình nền (CV10, CV11)

Khai báo các trọng số và kích thước ảnh cần so trong `evaluate:`, rồi chạy:

```bash
python -m dashcam_ml evaluate
```

- Trọng số COCO (`person`, `motorcycle`...) được **tự remap** sang nhãn BDD/VN (`pedestrian`, ...) qua `aliases` trong `classes:`, nên có thể so COCO với bản fine-tune trên cùng tập test.
- Kết quả (P, R, mAP50, mAP50-95, riêng từng lớp, ms/ảnh) được ghi vào `outputs/eval/evaluation_report.json`. Lớp trong `focus_classes` (mặc định `motorcycle`) được in riêng.

### 3.4 Fine-tune (CV12)

```bash
python -m dashcam_ml train
```

Trọng số tốt nhất nằm ở `outputs/train/<run_name>/weights/best.pt`. Thêm đường dẫn này vào `evaluate.weights` để so với mô hình nền.

**Train trên Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khanhhuypham/AIDashCam/blob/main/colab_train.ipynb), rồi làm theo ô đầu tiên của notebook. Notebook làm luôn bước đánh giá và xuất Core ML, kết quả lưu ở `MyDrive/AIDashCam/outputs/`.

### 3.5 Chạy logic cảnh báo trên video (CV04, CV13)

```bash
python -m dashcam_ml run-video
```

- Đầu vào lấy từ `video.source`. AI chạy ở `video.ai_fps` (mặc định 10 FPS) để giả lập iPhone 8.
- Tốc độ xe lấy từ `ego_speed:`. Có thể dùng hằng số, hoặc một file CSV `timestamp_s,speed_kmh` xuất từ GPS.
- Kết quả trong `outputs/video/`:
  - `<video>_debug.mp4`: box, ID track, vùng làn xe (hình thang), HUD khoảng cách/TTC/headway, banner cảnh báo.
  - `<video>_events.json`: đầu vào và đầu ra của logic ở từng khung AI, kèm toàn bộ ngưỡng. File này là **dữ liệu test cho bản Swift** [App iOS (sau pipeline)].

**Logic FCW** (`core/fcw.py`):

1. Tracker IoU gán ID cho từng đối tượng. Track chỉ được dùng khi đã xuất hiện `min_hits` lần.
2. Xe dẫn đầu là đối tượng gần nhất có điểm giữa cạnh dưới box nằm trong hình thang `in_path_roi`.
3. **TTC = s / (ds/dt)**, với `s` là bề rộng (hoặc chiều cao) box, lấy bằng hồi quy tuyến tính trên `ttc_window_s` giây gần nhất. Cách này không cần hiệu chỉnh camera. Sau đó làm mượt bằng EMA.
4. Khoảng cách tính theo mô hình pinhole từ `reference_size_m` và `camera.horizontal_fov_deg`. Chỉ gần đúng. Headway = khoảng cách / tốc độ xe.
5. Mức cảnh báo: `CRITICAL` nếu TTC ≤ `ttc_critical_s`, `CAUTION` nếu TTC ≤ `ttc_caution_s` hoặc headway ≤ `headway_caution_s`. Dưới `min_ego_speed_kmh` thì không cảnh báo.
6. Chống nhấp nháy: cần `min_consecutive_frames` khung liên tiếp mới bật cảnh báo. Âm thanh chỉ phát lại sau `cooldown_s`, trừ khi mức cảnh báo tăng lên.

**Logic LDW** (`core/ldw.py`): ở mỗi dòng pixel gần đáy ảnh, lấy mép trong của vạch làn bên trái và bên phải tâm xe để suy ra tâm làn và bề rộng làn. Lấy trung vị qua các dòng. Độ lệch được tính là `(tâm xe − tâm làn) / (nửa bề rộng làn)`; nếu vượt `departure_threshold` liên tục thì cảnh báo trái/phải. Trên iPhone, người lái tự xác nhận xi-nhan qua tham số `turn_signal_active`.

### 3.6 Phương án B: TwinLiteNet

Repo của tác giả (cần kiểm tra lại link và trọng số):

```bash
git clone https://github.com/chequanghuy/TwinLiteNet third_party/TwinLiteNet
```

Kiểm tra `twinlitenet.weights` có trỏ đúng tới file `.pth` không, sau đó đặt `ldw.enabled: true` và chạy `run-video` để xem mặt nạ vùng lái (xanh lá), vạch làn (đỏ) và cảnh báo lệch làn.

---

## 4. Xuất Core ML (CV14), chạy trên macOS/Linux

### 4.1 YOLO

Mỗi profile trong `coreml_export.profiles` tạo ra một `.mlpackage`:

| Profile | imgsz | precision | Dành cho |
|---|---|---|---|
| `yolo11n_320_fp16` | 320 | fp16 | iPhone 8 (A11, chạy GPU) |
| `yolo11n_416_fp16` | 416 | fp16 | iPhone 8 |
| `yolo11n_416_int8` | 416 | int8 | iPhone 8, thử nén trọng số |
| `yolo11n_640_fp16` | 640 | fp16 | A12+ (XR/XS trở lên, chạy ANE) |

```bash
python -m dashcam_ml export-coreml
```

- `precision` là enum `fp32 | fp16 | int8`. Code tự dùng tham số `quantize` với Ultralytics ≥ 8.4, và `half`/`int8` với bản cũ hơn.
- `imgsz` có thể là số (ảnh vuông) hoặc `[cao, rộng]` (bội số của 32), ví dụ `[256, 416]` để hợp với khung 16:9.
- `nms: true` gộp NMS vào mô hình. Khi đó Vision trả về thẳng `VNRecognizedObjectObservation`.
- Kết quả trong `outputs/coreml/`: các `.mlpackage` và `manifest.json` (kích thước input, precision, tên lớp, tên input/output).

### 4.2 TwinLiteNet

```bash
python -m dashcam_ml export-twinlitenet
```

Input là ảnh RGB 640×360. Hai output `drivable_prob` và `lane_prob` có dạng `(1, 360, 640)`, là xác suất lớp tiền cảnh. Trong Swift, so với `mask_threshold` để ra mặt nạ.

### 4.3 So Core ML với PyTorch (chỉ macOS)

Chép vài ảnh dashcam vào `data/samples/images/`, rồi chạy:

```bash
python -m dashcam_ml verify-coreml
```

Hai phía nhận **cùng một ảnh đã letterbox**. Báo cáo `outputs/coreml/verify_report.json` gồm tỉ lệ box khớp, IoU trung bình, |Δconfidence| (YOLO), và IoU mặt nạ (TwinLiteNet). Muốn biết mô hình chạy trên CPU, GPU hay ANE thì xem thêm **Core ML Performance Report** trong Xcode.

### 4.4 Config cho app iOS [App iOS (sau pipeline)]

```bash
python -m dashcam_ml export-ios-config
```

Lệnh tạo `outputs/coreml/DashcamConfig.json` gồm camera, lớp, tracker, fcw, ldw và danh sách mô hình, để Swift dùng **chung bộ ngưỡng** với Python. Key để dạng `snake_case`; trong Swift decode bằng `JSONDecoder().keyDecodingStrategy = .convertFromSnakeCase`.

---

## 5. Tích hợp vào app iOS

1. Kéo `.mlpackage` và `DashcamConfig.json` vào Xcode target (deployment target iOS 16).
2. Chạy mô hình bằng Vision trên một **queue riêng**, tách khỏi luồng ghi video [App iOS (sau pipeline)]:
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

---

## 6. Cấu hình (`config.yaml`)

| Nhóm | Nội dung |
|---|---|
| `project` | thư mục output, seed, `device` (`auto`/`cpu`/`0`/`mps`), `log_level` |
| `classes` | tên lớp + alias, nhóm (`vehicle`/`two_wheeler`/`pedestrian`), kích thước thật |
| `camera`, `ego_speed` | FOV ngang của camera, nguồn tốc độ xe (`constant`/`csv`/`none`) |
| `bdd100k`, `frame_extraction`, `split` | dữ liệu |
| `train`, `evaluate` | huấn luyện, đánh giá |
| `inference`, `video`, `tracker`, `fcw` | chạy video + FCW |
| `twinlitenet`, `ldw` | phương án B |
| `coreml_export`, `twinlitenet_export`, `coreml_verify`, `ios_config_export` | Core ML |

Mọi trường đều được kiểm tra bởi `dashcam_ml/config/schema.py`: sai kiểu, sai giá trị enum, key thừa, hay `ttc_critical_s ≥ ttc_caution_s` đều bị báo lỗi ngay lúc đọc.

---

## 7. Giới hạn và lưu ý

- Khoảng cách từ một camera **chỉ gần đúng**. Phải hiệu chỉnh `camera.horizontal_fov_deg` và `in_path_roi` theo góc gắn điện thoại thực tế. TTC dựa trên tốc độ phóng to của box nên ít bị ảnh hưởng hơn.
- App chỉ **hỗ trợ cảnh báo**, không thay thế sự chú ý của người lái. Không chạy nền (giới hạn iOS). iPhone 8 tối đa iOS 16, nên không dùng API iOS 17+.
- Các ngưỡng mặc định trong `config.yaml` là điểm khởi đầu hợp lý, **chưa được hiệu chỉnh trên đường Việt Nam**. Cần chỉnh sau khi chạy thử trên xe và đếm cảnh báo sai mỗi giờ [App iOS (sau pipeline)].
- Không đưa `data/`, `weights/`, `outputs/`, `third_party/` vào Git (đã có trong `.gitignore`).
