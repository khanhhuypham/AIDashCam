# AI Dashcam: phần Python train mô hình (`dashcam_ml`)

Code Python để **chuẩn bị dữ liệu, train/đánh giá và xuất mô hình** cho app dashcam **iOS** (Core ML, máy tối thiểu iPhone 8, iOS 16) và **Android** (TFLite).

Phần **chạy logic cảnh báo trên video** (tracking, FCW, LDW) và **`DashcamConfig.json` cho app** nằm ở project riêng **dashcam-video**, đặt cạnh project này (`../dashcam-video`). Hai project chỉ trao đổi qua file:

| File | Project này tạo | Ai dùng |
|---|---|---|
| `outputs/train/yolo11n_bdd100k/weights/best.pt` | `train` | dashcam-video (chạy video trên máy tính) |
| `outputs/coreml/*.mlpackage` | `export-coreml` | app iOS |
| `outputs/coreml/manifest.json` | `export-coreml` | dashcam-video (điền danh sách mô hình vào `DashcamConfig.json` của iOS) |
| `outputs/tflite/*.tflite` | `export-tflite` | app Android |
| `outputs/tflite/manifest.json` | `export-tflite` | dashcam-video (điền danh sách mô hình vào `DashcamConfig.json` của Android) |

Nội dung dựa trên file `Đồ_án_AI_Dashcam.xlsm`:

| Phương án (sheet *So sánh đề tài*) | Trạng thái trong code |
|---|---|
| **A. Cảnh báo va chạm (FCW)**: YOLO11n | Đầy đủ: data, train, evaluate, xuất Core ML + TFLite, kiểm tra. Logic FCW ở dashcam-video |
| **B. Cảnh báo lệch làn (LDW)**: TwinLiteNet | Xuất Core ML (cần clone repo tác giả); chưa có bản TFLite. Logic LDW ở dashcam-video |
| C. Đa nhiệm: YOLOP | Chưa làm (nặng nhất, rủi ro cao trên máy yếu) |

Các việc trong sheet *Công việc cần làm* mà code này làm được:

| Việc | Lệnh |
|---|---|
| CV02 Tải và xem BDD100K | `prepare-bdd100k` |
| CV08 Trích khung hình, lọc ảnh mờ/trùng | `extract-frames` |
| CV09 Chia dữ liệu theo từng video | `split-dataset` |
| CV10/CV11 Đánh giá trọng số có sẵn, so kích thước ảnh | `evaluate` |
| CV12 Fine-tune | `train` |
| CV14 Xuất Core ML (iOS) và so với PyTorch | `export-coreml`, `export-twinlitenet`, `verify-coreml` |
| Xuất TFLite (Android) và so với PyTorch | `export-tflite`, `verify-tflite` |
| CV04, CV13, dữ liệu test cho bản Swift / Kotlin | ở project dashcam-video |

---

## 1. Kiến trúc

```
dashcam-train/
├── config.yaml                 # TOÀN BỘ cấu hình (đường dẫn, tham số train, profile export)
├── requirements.txt
├── dashcam_ml/
│   ├── __main__.py             # python -m dashcam_ml <command>
│   ├── domain/                 # (1) Không phụ thuộc thư viện ngoài
│   │   ├── enums.py            #     mọi Enum: CopyMode, WeightPrecision, ModelKind, Command...
│   │   └── entities.py         #     BoundingBox, Detection (dùng khi so mô hình xuất ra với PyTorch)
│   ├── config/                 # (2) Schema pydantic cho config.yaml + loader
│   ├── core/                   # (3) class_catalog.py: tra lớp theo tên/alias
│   ├── models/                 # (4) Ultralytics (tham số dùng chung), TwinLiteNet (torch)
│   ├── data/                   # (4) BDD100K -> YOLO, trích khung hình, chia tập, remap lớp
│   ├── training/               # (4) train, evaluate
│   ├── export/                 # (4) Core ML (iOS), TFLite (Android), manifest, kiểm tra
│   ├── cli/                    # (5) argparse, mỗi Command một handler
│   └── utils/                  #     logging, file, device/platform
├── docs/                       # kế hoạch, ghi chú
└── tests/                      # pytest cho config + data
```

Quy tắc phụ thuộc: tầng dưới không import tầng trên.
`domain` ← `config` ← `core` ← `models / data / training / export` ← `cli`.

### Bản đồ code: lệnh nào dùng code nào

Mỗi lệnh có một handler trong `cli/commands.py`. Handler gọi đúng một hàm chính, nên muốn đọc lệnh nào thì mở file ở cột "Bắt đầu đọc từ".

| Lệnh | Bắt đầu đọc từ | Code dùng thêm | Kết quả |
|---|---|---|---|
| `prepare-bdd100k` | `data/bdd100k.py` | `data/yolo_dataset.py` | `data/yolo_bdd100k/` |
| `extract-frames` | `data/frame_extraction.py` | | `data/vn_frames/` |
| `split-dataset` | `data/split.py` | `data/yolo_dataset.py` | `data/vn_split/` |
| `train` | `training/trainer.py` | | `outputs/train/<run_name>/` |
| `evaluate` | `training/evaluator.py` | `data/class_remap.py`, `core/class_catalog.py`, `models/ultralytics_utils.py` | `outputs/eval/` |
| `export-coreml` | `export/yolo_coreml.py` | `export/manifest.py`, `models/ultralytics_utils.py` | `outputs/coreml/*.mlpackage`, `manifest.json` |
| `export-twinlitenet` | `export/twinlitenet_coreml.py` | `models/twinlitenet.py`, `export/manifest.py` | `outputs/coreml/twinlitenet_*.mlpackage` |
| `verify-coreml` | `export/verify.py` | `export/comparison.py` | `outputs/coreml/verify_report.json` |
| `export-tflite` | `export/yolo_tflite.py` | `export/manifest.py`, `models/ultralytics_utils.py` | `outputs/tflite/*.tflite`, `manifest.json` |
| `verify-tflite` | `export/verify.py` | `export/comparison.py` | `outputs/tflite/verify_report.json` |

`domain/`, `config/` và `utils/` được mọi lệnh dùng chung.

### Thư mục `outputs/`

Mọi thứ do các lệnh tạo ra đều nằm ở đây (không đưa vào Git):

```
outputs/
├── pretrained_coco/yolo11n.pt              # mô hình gốc học trên COCO, điểm bắt đầu để train
├── train/yolo11n_bdd100k/weights/best.pt   # mô hình đã train trên BDD100K (dùng cho video, Core ML, TFLite)
├── eval/evaluation_report.json             # điểm số trước và sau khi train
├── coreml/                                 # .mlpackage + manifest.json cho app iOS
└── tflite/                                 # .tflite + manifest.json cho app Android
```

### Quy ước code

- Mọi hàm, biến, thuộc tính đều có **kiểu dữ liệu**. Hàm của project đặt `*` trong chữ ký để **bắt buộc gọi bằng named argument**, ví dụ `load_config(config_path=...)`.
- Mọi tập giá trị cố định đều là **`Enum`** (`StrEnum`/`IntEnum`) trong `domain/enums.py`. Pydantic chuyển chuỗi trong YAML sang enum, nên giá trị sai bị báo lỗi ngay lúc đọc config.
- Không viết cứng ngưỡng hay đường dẫn trong code. Tất cả nằm trong `config.yaml`.
- Tọa độ box luôn **chuẩn hóa 0..1, gốc trên-trái**, không phụ thuộc độ phân giải, giống cách dùng với Vision trên iOS và với đầu ra TFLite trên Android.

---

## 2. Cài đặt

Yêu cầu Python 3.11–3.12.

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. macOS/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

- **GPU NVIDIA**: cài `torch` bản CUDA theo https://pytorch.org/get-started/locally/ trước, sau đó mới chạy lệnh trên.
- **Core ML** (`coremltools`) **không có bản Windows**. Các lệnh `export-*` phải chạy trên macOS, Linux, WSL2 hoặc Google Colab. `verify-coreml` chỉ chạy được trên **macOS**, vì chỉ máy Apple mới chạy được mô hình Core ML.
- **TFLite** (LiteRT: `litert-torch`, `ai-edge-litert`) cũng **không có bản Windows**: `export-tflite` và `verify-tflite` chạy trên Linux x86-64 hoặc macOS, thường là Google Colab. Notebook tự cài các gói này ở bước xuất.
- Trọng số chính thức (`yolo11n.pt`) được Ultralytics **tự tải về** thư mục `outputs/pretrained_coco/` khi dùng lần đầu. Đây là mô hình gốc học trên COCO; mô hình đã train trên BDD100K nằm ở `outputs/train/yolo11n_bdd100k/weights/best.pt`.

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

Trọng số tốt nhất nằm ở `outputs/train/<run_name>/weights/best.pt`. Đường dẫn này đã có trong `evaluate.weights` (để so với mô hình nền), `coreml_export.weights`, `coreml_verify.pytorch_weights`, `tflite_export.weights` và `tflite_verify.pytorch_weights`. Project dashcam-video cũng đọc thẳng file này để chạy video.

**Train trên Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khanhhuypham/AIDashCam/blob/main/dashcam-train/colab_train.ipynb), rồi làm theo ô đầu tiên của notebook. Notebook làm luôn bước đánh giá và xuất Core ML + TFLite, kết quả lưu ở `MyDrive/AIDashCam/outputs/`.

### 3.5 Thử mô hình trên video

Xem project **dashcam-video** (`../dashcam-video/README.md`): `python -m dashcam_video run`.

---

## 4. Xuất Core ML cho iOS (CV14), chạy trên macOS/Linux

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

Repo của tác giả (cần kiểm tra lại link và trọng số). dashcam-video cũng dùng repo clone ở đây khi bật LDW:

```bash
git clone https://github.com/chequanghuy/TwinLiteNet third_party/TwinLiteNet
```

```bash
python -m dashcam_ml export-twinlitenet
```

Input là ảnh RGB 640×360. Hai output `drivable_prob` và `lane_prob` có dạng `(1, 360, 640)`, là xác suất lớp tiền cảnh. Trong app, so với `mask_threshold` để ra mặt nạ.

### 4.3 So Core ML với PyTorch (chỉ macOS)

Chép vài ảnh dashcam vào `data/samples/images/`, rồi chạy:

```bash
python -m dashcam_ml verify-coreml
```

Hai phía nhận **cùng một ảnh đã letterbox**. Báo cáo `outputs/coreml/verify_report.json` gồm tỉ lệ box khớp, IoU trung bình, |Δconfidence| (YOLO), và IoU mặt nạ (TwinLiteNet). Muốn biết mô hình chạy trên CPU, GPU hay ANE thì xem thêm **Core ML Performance Report** trong Xcode.

### 4.4 Giao cho app iOS

Kéo các `.mlpackage` vào Xcode target (deployment target iOS 16). `DashcamConfig.json` (ngưỡng cảnh báo + danh sách mô hình) do dashcam-video tạo: `python -m dashcam_video export-app-config` (file `outputs/ios/DashcamConfig.json`).

---

## 5. Xuất TFLite cho Android, chạy trên Linux/macOS (Colab)

### 5.1 YOLO

Ultralytics gọi định dạng này là **LiteRT** (tên mới của TFLite), file vẫn là `.tflite`. Mỗi profile trong `tflite_export.profiles` tạo ra một file:

| Profile | imgsz | precision | Dành cho |
|---|---|---|---|
| `yolo11n_320_fp32` | 320 | fp32 | Android tầm thấp (GPU delegate) |
| `yolo11n_416_fp32` | 416 | fp32 | Android tầm trung (GPU delegate) |
| `yolo11n_416_int8` | 416 | int8 | máy không có GPU tốt, thử INT8 |

```bash
python -m dashcam_ml export-tflite
```

- `precision` chỉ nhận `fp32` hoặc `int8`. Mô hình fp32 tự chạy ở fp16 trên GPU Android (GPU delegate), nên không có profile fp16 riêng.
- `int8` cần ảnh để hiệu chỉnh: lấy ảnh `val` của `calibration_data`, dùng `calibration_fraction` (mặc định 5%).
- **Mô hình không có NMS bên trong** (LiteRT không hỗ trợ). App Android tự lọc theo ngưỡng `conf` và gộp box trùng (NMS) theo `iou` trong `DashcamConfig.json`. Cách đọc input/output ghi ở trường `output_layout` của từng mô hình trong `manifest.json`: input `float32` RGB 0..1 đã letterbox (viền 114); output mỗi anchor gồm `cx, cy, w, h` chuẩn hóa 0..1 rồi điểm của từng lớp theo thứ tự `class_names`.
- `minimum_android_sdk` (mặc định 26 = Android 8.0) được ghi vào `manifest.json`. **Giá trị tạm**, đổi khi chốt máy Android yếu nhất.
- Kết quả trong `outputs/tflite/`: các `.tflite` và `manifest.json`. Bản manifest của iOS (`outputs/coreml/`) không đổi.

### 5.2 So TFLite với PyTorch

```bash
python -m dashcam_ml verify-tflite
```

Giống `verify-coreml` (cùng ảnh đã letterbox, cùng chỉ số), báo cáo ở `outputs/tflite/verify_report.json`. Trên Colab notebook tự trỏ `tflite_verify.images_dir` sang ảnh val.

### 5.3 Giao cho app Android

Chép các `.tflite` vào `assets/` của app Android. `DashcamConfig.json` cho Android do dashcam-video tạo cùng lệnh `python -m dashcam_video export-app-config` (file `outputs/android/DashcamConfig.json`).

---

## 6. Cấu hình (`config.yaml`)

| Nhóm | Nội dung |
|---|---|
| `project` | thư mục output, seed, `device` (`auto`/`cpu`/`0`/`mps`), `log_level` |
| `classes` | tên lớp + alias (để so mô hình COCO với nhãn BDD/VN) |
| `bdd100k`, `frame_extraction`, `split` | dữ liệu |
| `train`, `evaluate` | huấn luyện, đánh giá |
| `twinlitenet` | phương án B: repo + trọng số TwinLiteNet |
| `coreml_export`, `twinlitenet_export`, `coreml_verify` | Core ML (iOS) |
| `tflite_export`, `tflite_verify` | TFLite (Android) |

Mọi trường đều được kiểm tra bởi `dashcam_ml/config/schema.py`: sai kiểu, sai giá trị enum, key thừa, hay tỉ lệ chia tập không cộng đủ 1 đều bị báo lỗi ngay lúc đọc.

---

## 7. Lưu ý

- iPhone 8 tối đa iOS 16, nên mô hình xuất ra phải chạy được trên iOS 16 (`minimum_ios_version` trong `manifest.json` của Core ML).
- Máy Android yếu nhất chưa chốt: `minimum_android_sdk` và các profile TFLite hiện là giá trị tạm.
- Không đưa `data/`, `outputs/`, `third_party/` vào Git (đã có trong `.gitignore`).
