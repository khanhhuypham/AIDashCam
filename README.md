# AI Dashcam (iOS + Android): phần Python

Repo gồm hai project Python cho app dashcam **iOS** (máy tối thiểu iPhone 8, iOS 16) và **Android**. Mỗi project có `config.yaml`, `README.md` và `requirements.txt` riêng.

| Thư mục | Làm gì | Lệnh |
|---|---|---|
| [`dashcam-train/`](dashcam-train/README.md) | dữ liệu BDD100K → train YOLO11n → đánh giá → xuất Core ML (iOS) + TFLite (Android) | `python -m dashcam_ml <lệnh>` |
| [`dashcam-video/`](dashcam-video/README.md) | chạy detect + tracking + cảnh báo va chạm (FCW) / lệch làn (LDW) trên video, xuất `DashcamConfig.json` cho app iOS và Android | `python -m dashcam_video <lệnh>` |

Hai project chỉ trao đổi qua file: `dashcam-video` đọc `../dashcam-train/outputs/train/yolo11n_bdd100k/weights/best.pt` và `manifest.json` trong `../dashcam-train/outputs/coreml/` (iOS) và `../dashcam-train/outputs/tflite/` (Android). Vì vậy giữ hai thư mục cạnh nhau như trong repo.

**Train trên Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khanhhuypham/AIDashCam/blob/main/dashcam-train/colab_train.ipynb)

Chạy lệnh của project nào thì mở terminal (hoặc PyCharm) ở thư mục của project đó.
