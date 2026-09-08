# PrivWatch

A Privacy-Preserving, Non-Persistent Video Anomaly Detection Framework

PrivWatch is an intelligent video surveillance research prototype designed to detect safety-critical activities while minimizing the persistence and unnecessary retention of sensitive visual data.

The core idea is simple:

Analyze video when it is necessary, retain the minimum information required, and avoid creating a permanent archive of surveillance footage by default.

PrivWatch combines deep-learning-based video activity recognition with a transient, RAM-oriented processing workflow. The system currently focuses on three primary activity classes:

- Normal Activity
- Fight
- Collapse

The project also contains a Sterile Room execution layer that wraps the inference pipeline with volatile buffering, short-lived worker processes, memory sanitization, and event-only alert handling.

This repository is an academic/research prototype and should not be interpreted as a production-grade secure enclave or as a guarantee against operating-system-level forensic recovery.

## 1. Project Vision

Conventional CCTV systems are generally optimized for surveillance continuity and evidence retention. A typical deployment continuously records video and stores it on local disks, NVRs, servers, or cl[...]

That model creates a fundamental privacy problem:

Most surveillance footage contains normal activity, yet the system permanently retains it.

PrivWatch explores an alternative workflow:

```text
Camera / Video Stream
        │
        ▼
Transient Frame Acquisition
        │
        ▼
Volatile RAM Buffer
        │
        ▼
AI Video Inference
        │
        ▼
Event Classification
        │
        ├── Normal ──────────────┐
        │                        │
        └── Anomaly ─────────────┤
                                 ▼
                           Event Metadata
                                 │
                                 ▼
                         Frame Evaporation
```

The objective is not to eliminate surveillance.

The objective is to reduce the persistence surface of surveillance data while preserving the ability to detect important events.

## 2. Problem Statement

Traditional surveillance systems create several privacy and security risks:

- Continuous recording of individuals.
- Long-term retention of personally identifiable visual information.
- Large surveillance archives becoming attractive attack targets.
- Unauthorized access to historical footage.
- Insider misuse of stored recordings.
- Unnecessary retention of normal activity.
- Dependence on permanent storage infrastructure.
- Difficulty proving that footage has actually been removed after use.

PrivWatch investigates whether real-time AI-based event detection can be performed while substantially reducing persistent storage of raw video.

The project therefore treats data lifetime as an important engineering variable in addition to model accuracy.

## 3. Core Objectives

### Primary objectives

- Detect safety-critical activities from video.
- Maintain useful anomaly-detection recall.
- Process video transiently.
- Avoid intentional permanent storage of raw surveillance frames during sterile inference.
- Minimize the lifetime of sensitive visual data in memory.
- Isolate inference execution where practical.
- Preserve only minimal event metadata when required.
- Provide an auditable demonstration of the privacy-preserving workflow.

### Secondary objectives

- Compare conventional/raw surveillance processing with transient processing.
- Evaluate multiple video-learning architectures.
- Measure accuracy, precision, recall, F1-score, confusion matrix, and inference latency.
- Study false positives and false negatives using unseen videos.
- Demonstrate the practical trade-off between surveillance accuracy and privacy persistence.

## 4. Current System Architecture

The current PrivWatch prototype is composed of two major layers.

### Layer A — AI inference

The current primary prototype uses:

```text
MobileNetV2
     │
     ▼
Spatial feature extraction
     │
     ▼
Temporal sequence construction
     │
     ▼
LSTM
     │
     ▼
3-class classifier
     │
     ├── Normal
     ├── Fight
     └── Collapse
```

MobileNetV2 provides spatial feature extraction from individual frames.

LSTM models temporal relationships between extracted frame-level features.

The resulting classifier produces class probabilities for the three supported activities.

### Layer B — Sterile Room

The inference engine can be wrapped by the Sterile Room execution layer:

```text
Input Video
    │
    ▼
RAM-oriented acquisition
    │
    ▼
Volatile circular buffer
    │
    ▼
Short-lived inference worker
    │
    ▼
MobileNetV2 + LSTM
    │
    ▼
Event result
    │
    ▼
Memory cleanup / buffer overwrite
    │
    ▼
Worker termination
    │
    ▼
Event metadata only
```

The Sterile Room does not replace the AI model.

It changes how the model is executed and how visual data is handled around it.

## 5. Privacy Philosophy

PrivWatch currently follows a Privacy by Procedure philosophy.

The system does not claim that the underlying computer becomes mathematically incapable of retaining information.

Instead, the implementation attempts to establish a controlled processing procedure:

- Receive video.
- Decode it transiently.
- Maintain only a small working buffer.
- Perform inference.
- Generate an event result.
- Overwrite/clear accessible buffers.
- Release runtime resources.
- Destroy the short-lived worker process.
- Retain only non-visual event information where logging is enabled.

This is intentionally a more conservative claim than:

"The video can never exist on disk."

The project acknowledges operating-system, runtime, driver, and hardware-level limitations.

## 6. Sterile Room

The Sterile Room is the privacy-oriented execution wrapper around the existing inference system.

### Components

| Component | Responsibility |
| --- | --- |
| sterile_session.py | Controls transient inference sessions and video acquisition |
| volatile_buffer.py | Maintains a bounded circular frame buffer |
| inference_worker.py | Executes inference in a short-lived spawned process |
| memory_sanitizer.py | Performs best-effort memory cleanup and CUDA cleanup |
| alert_registry.py | Stores event metadata without storing frames |
| streamlit_integration.py | Provides integration support for Streamlit-based interfaces |
| README.md | Documents Sterile Room architecture and limitations |
| THREAT_MODEL.md | Documents threats, mitigations, and unavoidable risks |

### Sterile Room design goals

#### RAM-first processing

Frames are intended to remain in volatile memory during normal inference operation rather than being intentionally written to a permanent video archive.

#### Bounded buffering

The circular buffer limits how many frames are simultaneously retained by the application.

#### Sliding evaporation

When a buffer slot is reused, the previous frame is actively cleared before the slot is overwritten where practical.

#### Short-lived inference workers

Inference is executed in a spawned worker process.

The goal is to limit the lifetime of:

- Python allocator state
- Torch tensors
- CUDA context
- worker-local objects
- temporary runtime state

When the worker exits, the operating system destroys the process address space.

#### Best-effort memory sanitization

The implementation attempts to clear accessible NumPy/Torch buffers before releasing them.

CUDA cache cleanup is also performed where supported.

#### Event-only persistence

The design does not require saving:

- raw frames
- frame crops
- face images
- pose skeletons
- visual embeddings

An alert may contain only metadata such as:

- timestamp
- event label
- confidence
- processing status

## 7. Threat Model

PrivWatch does not claim to be an impenetrable security boundary.

The current threat model explicitly recognizes the following limitations.

| Threat | Severity | Position |
| --- | --- | --- |
| Permanent video archive | High | Mitigated by transient processing |
| Persistent Python runtime | Medium | Reduced using worker isolation |
| CUDA allocator persistence | Low–Medium | Best-effort cleanup + worker destruction |
| NumPy allocator retention | Low | Best-effort wiping |
| OpenCV/native decoder buffers | Low | Partially controllable |
| Multiprocessing serialization copies | Medium | Short-lived but unavoidable in current design |
| Windows pagefile/swap | Medium | Not fully controllable from Python |
| CUDA driver internals | Medium | Outside application control |
| Kernel-level forensic acquisition | High | Not mitigated |
| Compromised host OS | Critical | Out of scope |
| Malicious hardware/firmware | Critical | Out of scope |

The project's claims should therefore remain precise:

PrivWatch minimizes intentional persistence of reconstructable surveillance footage.

It does not claim:

PrivWatch guarantees that no recoverable representation can ever exist anywhere in system memory or storage.

## 8. AI Model

### Current primary model

The primary implemented video classifier is:

```text
Input video
     │
     ▼
Frame sampling
     │
     ▼
MobileNetV2
     │
     ▼
Frame-level feature vectors
     │
     ▼
Temporal sequence
     │
     ▼
LSTM
     │
     ▼
Fully connected classifier
     │
     ▼
Normal / Fight / Collapse
```

### Why MobileNetV2?

MobileNetV2 was selected because the project is intended to operate under constrained edge-computing conditions.

Advantages:

- Lightweight CNN architecture.
- Relatively low computational cost.
- Suitable for GPU and CPU inference.
- Strong transfer-learning ecosystem.
- Practical for student/research hardware.
- Significantly lighter than many large video-transformer architectures.

### Temporal modeling

MobileNetV2 operates primarily on spatial frame information.

The LSTM adds temporal reasoning by processing a sequence of frame-level feature vectors.

This combination provides a practical compromise:

- MobileNetV2 → spatial understanding
- LSTM        → temporal understanding

## 9. Experimental Model Comparison

PrivWatch is also being evaluated against alternative computer-vision architectures.

The experimental model set includes:

- MobileNetV2 + LSTM
- YOLOv8-based approaches
- Video Swin Transformer / Swin Video Transformer
- Other lightweight video architectures where appropriate

These models should be treated as experimental comparison models, not automatically as the production model.

The final model should be selected using measured evidence rather than architecture popularity.

## 10. Model Evaluation

The project prioritizes anomaly detection quality over raw overall accuracy.

For safety-critical surveillance, a false negative can be more serious than a false positive.

Therefore, the evaluation should include:

### Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1

### Safety-oriented metrics

Especially important:

- Fight Recall
- Collapse Recall
- Anomaly Recall
- False Negative Rate
- False Positive Rate

### Confusion matrix

A confusion matrix should be produced for every final model.

Example:

```text
                 Predicted
              N       F       C
Actual N      TN      FP      FP
       F      FN      TP      FN
       C      FN      FN      TP
```

### Inference performance

Measure:

- Mean latency
- Median latency
- P95 latency
- Frames processed per second
- GPU utilization
- RAM usage
- VRAM usage

## 11. Dataset Organization

The project separates raw data, split data, processed clips, and calibration videos.

A typical layout is:

```text
data/
├── raw/
│   ├── Fight/
│   ├── Normal/
│   └── Collapse/
│
├── split/
│   ├── train/
│   ├── val/
│   └── test/
│
└── clips/
    ├── train/
    │   ├── Fight/
    │   ├── Normal/
    │   └── Collapse/
    │
    ├── val/
    │   ├── Fight/
    │   ├── Normal/
    │   └── Collapse/
    │
    └── test/
        ├── Fight/
        ├── Normal/
        └── Collapse/
```

Calibration and genuinely unseen evaluation data are kept separately:

```text
calibration/
├── normal/
├── fight/
└── collapse/
```

## 12. Data Processing Pipeline

The preprocessing pipeline converts source videos into model-compatible clips.

Typical workflow:

```text
Raw video
    │
    ▼
Video decoding
    │
    ▼
Frame sampling
    │
    ▼
Clip segmentation
    │
    ▼
Resize / normalization
    │
    ▼
Processed clip
    │
    ▼
Training dataset
```

The project deliberately keeps preprocessing separate from sterile inference.

Why?

Because training requires a reusable dataset, while privacy-preserving deployment should minimize persistent visual data.

## 13. Training Workflow

Training is performed from the prepared dataset.

Typical command:

```bash
python scripts/train_raw_model.py
```

The training pipeline should:

- Load the training clips.
- Construct batches.
- Perform forward propagation.
- Calculate classification loss.
- Backpropagate.
- Update model parameters.
- Evaluate on validation data.
- Save the best-performing checkpoint.
- Record reproducibility metadata.

The checkpoint is:

```text
models/best_model.pth
```

The project uses validation performance to determine the best checkpoint rather than assuming the final epoch is always the best model.

## 14. Model Lineage and Forensics

Model provenance is treated as an important engineering requirement.

A valid deployment should be able to answer:

- Which model is running?
- Which training run produced it?
- Which dataset configuration produced it?
- Which epoch was selected?
- What was its validation performance?
- When was it produced?
- What is its file hash?

The model forensic metadata should therefore include:

- model path
- model architecture
- training run ID
- epoch
- validation accuracy
- dataset summary
- training timestamp
- checkpoint SHA-256

This prevents uncertainty about whether inference is accidentally using an old model.

## 15. Inference

The standard inference path can be invoked through the Python module interface.

Example:

```bash
python -m privwatch.privacy_inference "video.mp4"
```

The inference system should return:

- Model identity
- Predicted class
- Class confidence
- Latency

For example:

```text
FINAL RESULT: ALERT - Fight detected
Latency: 2.31 seconds
```

For forensic/debugging purposes, the model identity can additionally be displayed.

## 16. Calibration and Unseen Testing

A critical part of PrivWatch development is evaluation on videos that were not used for training.

The calibration workflow tests:

- 10 Normal
- 10 Fight
- 10 Collapse

or a larger equivalent dataset.

The purpose is to identify:

- False positives
- False negatives
- Class confusion
- Domain shift
- Background bias
- Crowd-related errors
- Motion-related false alarms
- Camera/viewpoint sensitivity

A model that performs extremely well on training/validation data but fails on unseen videos should not be considered deployment-ready.

## 17. Hard Negative Mining

Normal activity is particularly important.

Examples of challenging normal scenes include:

- Crowds
- Clapping
- People running for harmless reasons
- Sports
- Dancing
- Gaming
- People sitting or lying down
- Multiple people interacting
- Sudden camera movement
- Busy public spaces
- Strong background motion

These examples can be added to the normal training set as hard negatives.

The objective is not simply to increase dataset size.

The objective is to teach the classifier:

"This visually unusual activity is still normal."

## 18. High-Recall Safety Strategy

For PrivWatch, the preferred operational objective is:

Minimize dangerous false negatives

A false positive can trigger an unnecessary alert.

A false negative can cause a real event to be missed.

Therefore, the project should tune:

```text
Anomaly Recall
        ↓
False Negative Rate
        ↓
Decision thresholds
        ↓
Temporal smoothing
        ↓
Alert confirmation logic
```

Threshold tuning should be performed on a validation/calibration set and should never be optimized using the final test set.

## 19. RAW vs Sterile Processing

PrivWatch maintains a conceptual comparison between two pipelines.

### RAW surveillance pipeline

```text
Camera
  ↓
Frame acquisition
  ↓
AI inference
  ↓
Event detection
  ↓
Permanent video storage
```

Characteristics:

- Persistent visual archive.
- Easier historical investigation.
- Larger storage requirements.
- Larger privacy exposure surface.

### Sterile Room pipeline

```text
Camera
  ↓
RAM buffer
  ↓
AI inference
  ↓
Event detection
  ↓
Alert metadata
  ↓
Frame eviction / memory cleanup
  ↓
Worker termination
```

Characteristics:

- No intentional permanent raw-video archive.
- Small volatile working set.
- Short-lived inference worker.
- Event-oriented output.
- Reduced persistence surface.

The AI model can remain the same.

The difference is primarily data lifecycle and execution procedure.

## 20. Technology Stack

### Machine Learning

| Technology | Role |
| --- | --- |
| Python | Primary development language |
| PyTorch | Deep learning framework |
| TorchVision | Vision models and transformations |
| MobileNetV2 | Spatial feature extractor |
| LSTM | Temporal sequence modeling |
| YOLOv8 | Experimental object/video detection comparison |
| Video Swin Transformer | Experimental video-model comparison |
| NumPy | Numerical processing |
| OpenCV | Video/frame processing where required |
| PyAV | RAM-oriented video decoding in Sterile Room |

### Backend / Runtime

| Technology | Role |
| --- | --- |
| Python | Core runtime |
| multiprocessing | Worker isolation |
| ctypes | Best-effort memory wiping |
| garbage collector | Object cleanup |
| CUDA | GPU acceleration |
| CUDA memory APIs | GPU cache cleanup |
| FastAPI | Forensic/demo backend |
| Uvicorn | ASGI server |

### Frontend / Demonstration Layer

The repository contains a separate forensic demonstration interface.

| Technology | Role |
| --- | --- |
| React | UI framework |
| TypeScript | Frontend language |
| Vite | Frontend build/dev tooling |
| CSS | UI styling |
| WebSocket | Runtime event streaming |
| FastAPI | Local backend API |
| Uvicorn | Local development server |

The forensic UI is intended to visualize system behavior rather than replace the actual inference engine.

### Development

| Tool | Role |
| --- | --- |
| Git | Version control |
| GitHub | Repository hosting/collaboration |
| VS Code | Primary development environment |
| PowerShell | Windows command-line workflow |
| Virtual environment (env) | Python dependency isolation |
| Roboflow | Optional dataset collection/management/training experimentation |

[...]