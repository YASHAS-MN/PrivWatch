# VI_PrivWatch Repository Structure

```
VI_PrivWatch/
├── Root Configuration & Scripts
│   ├── app.py
│   ├── setup.py
│   ├── requirements.txt
│   ├── selective_downloader.py
│   ├── download_collapse.py
│   ├── download_hockey.py
│   └── download_intrusion.py
│
├── archive_experiments/
│   ├── augment_dataset.py
│   ├── extract_ur_fall.py
│   └── train_baseline.py
│
├── data/
│   ├── features/
│   │   ├── normal/
│   │   └── violence/
│   ├── frames/
│   │   ├── normal/
│   │   └── violence/
│   ├── keypoints/
│   │   ├── collapse/          (50 fall_*.npy + synthetic augmented files)
│   │   ├── fight/
│   │   ├── intrusion/
│   │   └── normal/
│   └── raw_videos/
│       ├── collapse/
│       ├── fight/
│       ├── intrusion/
│       └── normal/
│
├── models/
│   ├── pose_landmarker_lite.task
│   └── saved_weights/
│       ├── pose_lstm_3class.pth
│       ├── pose_lstm_model.pth
│       └── raw_lstm_model.pth
│
└── src/
    ├── data_prep/
    │   ├── augment_math.py
    │   ├── extract_cnn_features.py
    │   ├── extract_frames.py
    │   ├── extract_keypoints.py
    │   ├── master_extractor.py
    │   └── data/
    │
    ├── models/
    │   ├── realtime_inference.py
    │   ├── realtime_pose_inference.py
    │   ├── train_pose_model.py
    │   └── train_raw_model.py
    │
    └── ui/
        └── app.py
```

## Project Overview

- **Root Scripts**: Entry points for data downloading and downloading orchestration
- **archive_experiments/**: Historical experiments and baseline training code
- **data/**: Organized dataset with features, frames, keypoints, and raw videos across 4 categories (collapse, fight, intrusion, normal)
- **models/**: Pre-trained models and saved weights (pose landmarker + LSTM models)
- **src/**: Main source code organized into three modules:
  - `data_prep/`: Data extraction and augmentation pipeline
  - `models/`: Model training and inference
  - `ui/`: Streamlit application interface
