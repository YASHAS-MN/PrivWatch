PrivWatch

A Privacy-Preserving, Non-Persistent Video Anomaly Detection Framework

PrivWatch is an intelligent video surveillance research prototype designed to detect safety-critical activities while minimizing the persistence and unnecessary retention of sensitive visual data.

The core idea is simple:

Analyze video when it is necessary, retain the minimum information required, and avoid creating a permanent archive of surveillance footage by default.

PrivWatch combines deep-learning-based video activity recognition with a transient, RAM-oriented processing workflow. The system currently focuses on three primary activity classes:

Normal Activity

Fight

Collapse

The project also contains a Sterile Room execution layer that wraps the inference pipeline with volatile buffering, short-lived worker processes, memory sanitization, and event-only alert handling.

This repository is an academic/research prototype and should not be interpreted as a production-grade secure enclave or as a guarantee against operating-system-level forensic recovery.

1. Project Vision

Conventional CCTV systems are generally optimized for surveillance continuity and evidence retention. A typical deployment continuously records video and stores it on local disks, NVRs, servers, or cloud infrastructure.

That model creates a fundamental privacy problem:

Most surveillance footage contains normal activity, yet the system permanently retains it.

PrivWatch explores an alternative workflow:

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

The objective is not to eliminate surveillance.

The objective is to reduce the persistence surface of surveillance data while preserving the ability to detect important events.

2. Problem Statement

Traditional surveillance systems create several privacy and security risks:

Continuous recording of individuals.

Long-term retention of personally identifiable visual information.

Large surveillance archives becoming attractive attack targets.

Unauthorized access to historical footage.

Insider misuse of stored recordings.

Unnecessary retention of normal activity.

Dependence on permanent storage infrastructure.

Difficulty proving that footage has actually been removed after use.

PrivWatch investigates whether real-time AI-based event detection can be performed while substantially reducing persistent storage of raw video.

The project therefore treats data lifetime as an important engineering variable in addition to model accuracy.

3. Core Objectives

Primary objectives

Detect safety-critical activities from video.

Maintain useful anomaly-detection recall.

Process video transiently.

Avoid intentional permanent storage of raw surveillance frames during sterile inference.

Minimize the lifetime of sensitive visual data in memory.

Isolate inference execution where practical.

Preserve only minimal event metadata when required.

Provide an auditable demonstration of the privacy-preserving workflow.

Secondary objectives

Compare conventional/raw surveillance processing with transient processing.

Evaluate multiple video-learning architectures.

Measure accuracy, precision, recall, F1-score, confusion matrix, and inference latency.

Study false positives and false negatives using unseen videos.

Demonstrate the practical trade-off between surveillance accuracy and privacy persistence.

4. Current System Architecture

The current PrivWatch prototype is composed of two major layers.

Layer A — AI inference

The current primary prototype uses:

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

MobileNetV2 provides spatial feature extraction from individual frames.

LSTM models temporal relationships between extracted frame-level features.

The resulting classifier produces class probabilities for the three supported activities.

Layer B — Sterile Room

The inference engine can be wrapped by the Sterile Room execution layer:

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

The Sterile Room does not replace the AI model.

It changes how the model is executed and how visual data is handled around it.

5. Privacy Philosophy

PrivWatch currently follows a Privacy by Procedure philosophy.

The system does not claim that the underlying computer becomes mathematically incapable of retaining information.

Instead, the implementation attempts to establish a controlled processing procedure:

Receive video.

Decode it transiently.

Maintain only a small working buffer.

Perform inference.

Generate an event result.

Overwrite/clear accessible buffers.

Release runtime resources.

Destroy the short-lived worker process.

Retain only non-visual event information where logging is enabled.

This is intentionally a more conservative claim than:

"The video can never exist on disk."

The project acknowledges operating-system, runtime, driver, and hardware-level limitations.

6. Sterile Room

The Sterile Room is the privacy-oriented execution wrapper around the existing inference system.

Components

Component

Responsibility

sterile_session.py

Controls transient inference sessions and video acquisition

volatile_buffer.py

Maintains a bounded circular frame buffer

inference_worker.py

Executes inference in a short-lived spawned process

memory_sanitizer.py

Performs best-effort memory cleanup and CUDA cleanup

alert_registry.py

Stores event metadata without storing frames

streamlit_integration.py

Provides integration support for Streamlit-based interfaces

README.md

Documents Sterile Room architecture and limitations

THREAT_MODEL.md

Documents threats, mitigations, and unavoidable risks

Sterile Room design goals

RAM-first processing

Frames are intended to remain in volatile memory during normal inference operation rather than being intentionally written to a permanent video archive.

Bounded buffering

The circular buffer limits how many frames are simultaneously retained by the application.

Sliding evaporation

When a buffer slot is reused, the previous frame is actively cleared before the slot is overwritten where practical.

Short-lived inference workers

Inference is executed in a spawned worker process.

The goal is to limit the lifetime of:

Python allocator state

Torch tensors

CUDA context

worker-local objects

temporary runtime state

When the worker exits, the operating system destroys the process address space.

Best-effort memory sanitization

The implementation attempts to clear accessible NumPy/Torch buffers before releasing them.

CUDA cache cleanup is also performed where supported.

Event-only persistence

The design does not require saving:

raw frames

frame crops

face images

pose skeletons

visual embeddings

An alert may contain only metadata such as:

timestamp
event label
confidence
processing status

7. Threat Model

PrivWatch does not claim to be an impenetrable security boundary.

The current threat model explicitly recognizes the following limitations.

Threat

Severity

Position

Permanent video archive

High

Mitigated by transient processing

Persistent Python runtime

Medium

Reduced using worker isolation

CUDA allocator persistence

Low–Medium

Best-effort cleanup + worker destruction

NumPy allocator retention

Low

Best-effort wiping

OpenCV/native decoder buffers

Low

Partially controllable

Multiprocessing serialization copies

Medium

Short-lived but unavoidable in current design

Windows pagefile/swap

Medium

Not fully controllable from Python

CUDA driver internals

Medium

Outside application control

Kernel-level forensic acquisition

High

Not mitigated

Compromised host OS

Critical

Out of scope

Malicious hardware/firmware

Critical

Out of scope

The project's claims should therefore remain precise:

PrivWatch minimizes intentional persistence of reconstructable surveillance footage.

It does not claim:

PrivWatch guarantees that no recoverable representation can ever exist anywhere in system memory or storage.

8. AI Model

Current primary model

The primary implemented video classifier is:

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

Why MobileNetV2?

MobileNetV2 was selected because the project is intended to operate under constrained edge-computing conditions.

Advantages:

Lightweight CNN architecture.

Relatively low computational cost.

Suitable for GPU and CPU inference.

Strong transfer-learning ecosystem.

Practical for student/research hardware.

Significantly lighter than many large video-transformer architectures.

Temporal modeling

MobileNetV2 operates primarily on spatial frame information.

The LSTM adds temporal reasoning by processing a sequence of frame-level feature vectors.

This combination provides a practical compromise:

MobileNetV2 → spatial understanding
LSTM        → temporal understanding

9. Experimental Model Comparison

PrivWatch is also being evaluated against alternative computer-vision architectures.

The experimental model set includes:

MobileNetV2 + LSTM

YOLOv8-based approaches

Video Swin Transformer / Swin Video Transformer

Other lightweight video architectures where appropriate

These models should be treated as experimental comparison models, not automatically as the production model.

The final model should be selected using measured evidence rather than architecture popularity.

10. Model Evaluation

The project prioritizes anomaly detection quality over raw overall accuracy.

For safety-critical surveillance, a false negative can be more serious than a false positive.

Therefore, the evaluation should include:

Classification metrics

Accuracy

Precision

Recall

F1-score

Macro F1

Per-class precision

Per-class recall

Per-class F1

Safety-oriented metrics

Especially important:

Fight Recall
Collapse Recall
Anomaly Recall
False Negative Rate
False Positive Rate

Confusion matrix

A confusion matrix should be produced for every final model.

Example:

                 Predicted
              N       F       C
Actual N      TN      FP      FP
       F      FN      TP      FN
       C      FN      FN      TP

Inference performance

Measure:

Mean latency

Median latency

P95 latency

Frames processed per second

GPU utilization

RAM usage

VRAM usage

11. Dataset Organization

The project separates raw data, split data, processed clips, and calibration videos.

A typical layout is:

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

Calibration and genuinely unseen evaluation data are kept separately:

calibration/
├── normal/
├── fight/
└── collapse/

12. Data Processing Pipeline

The preprocessing pipeline converts source videos into model-compatible clips.

Typical workflow:

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

The project deliberately keeps preprocessing separate from sterile inference.

Why?

Because training requires a reusable dataset, while privacy-preserving deployment should minimize persistent visual data.

13. Training Workflow

Training is performed from the prepared dataset.

Typical command:

python scripts/train_raw_model.py

The training pipeline should:

Load the training clips.

Construct batches.

Perform forward propagation.

Calculate classification loss.

Backpropagate.

Update model parameters.

Evaluate on validation data.

Save the best-performing checkpoint.

Record reproducibility metadata.

The checkpoint is:

models/best_model.pth

The project uses validation performance to determine the best checkpoint rather than assuming the final epoch is always the best model.

14. Model Lineage and Forensics

Model provenance is treated as an important engineering requirement.

A valid deployment should be able to answer:

Which model is running?
Which training run produced it?
Which dataset configuration produced it?
Which epoch was selected?
What was its validation performance?
When was it produced?
What is its file hash?

The model forensic metadata should therefore include:

model path
model architecture
training run ID
epoch
validation accuracy
dataset summary
training timestamp
checkpoint SHA-256

This prevents uncertainty about whether inference is accidentally using an old model.

15. Inference

The standard inference path can be invoked through the Python module interface.

Example:

python -m privwatch.privacy_inference "video.mp4"

The inference system should return:

Model identity
Predicted class
Class confidence
Latency

For example:

FINAL RESULT: ALERT - Fight detected
Latency: 2.31 seconds

For forensic/debugging purposes, the model identity can additionally be displayed.

16. Calibration and Unseen Testing

A critical part of PrivWatch development is evaluation on videos that were not used for training.

The calibration workflow tests:

10 Normal
10 Fight
10 Collapse

or a larger equivalent dataset.

The purpose is to identify:

False positives

False negatives

Class confusion

Domain shift

Background bias

Crowd-related errors

Motion-related false alarms

Camera/viewpoint sensitivity

A model that performs extremely well on training/validation data but fails on unseen videos should not be considered deployment-ready.

17. Hard Negative Mining

Normal activity is particularly important.

Examples of challenging normal scenes include:

Crowds

Clapping

People running for harmless reasons

Sports

Dancing

Gaming

People sitting or lying down

Multiple people interacting

Sudden camera movement

Busy public spaces

Strong background motion

These examples can be added to the normal training set as hard negatives.

The objective is not simply to increase dataset size.

The objective is to teach the classifier:

"This visually unusual activity is still normal."

18. High-Recall Safety Strategy

For PrivWatch, the preferred operational objective is:

Minimize dangerous false negatives

A false positive can trigger an unnecessary alert.

A false negative can cause a real event to be missed.

Therefore, the project should tune:

Anomaly Recall
        ↓
False Negative Rate
        ↓
Decision thresholds
        ↓
Temporal smoothing
        ↓
Alert confirmation logic

Threshold tuning should be performed on a validation/calibration set and should never be optimized using the final test set.

19. RAW vs Sterile Processing

PrivWatch maintains a conceptual comparison between two pipelines.

RAW surveillance pipeline

Camera
  ↓
Frame acquisition
  ↓
AI inference
  ↓
Event detection
  ↓
Permanent video storage

Characteristics:

Persistent visual archive.

Easier historical investigation.

Larger storage requirements.

Larger privacy exposure surface.

Sterile Room pipeline

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

Characteristics:

No intentional permanent raw-video archive.

Small volatile working set.

Short-lived inference worker.

Event-oriented output.

Reduced persistence surface.

The AI model can remain the same.

The difference is primarily data lifecycle and execution procedure.

20. Technology Stack

Machine Learning

Technology

Role

Python

Primary development language

PyTorch

Deep learning framework

TorchVision

Vision models and transformations

MobileNetV2

Spatial feature extractor

LSTM

Temporal sequence modeling

YOLOv8

Experimental object/video detection comparison

Video Swin Transformer

Experimental video-model comparison

NumPy

Numerical processing

OpenCV

Video/frame processing where required

PyAV

RAM-oriented video decoding in Sterile Room

Backend / Runtime

Technology

Role

Python

Core runtime

multiprocessing

Worker isolation

ctypes

Best-effort memory wiping

garbage collector

Object cleanup

CUDA

GPU acceleration

CUDA memory APIs

GPU cache cleanup

FastAPI

Forensic/demo backend

Uvicorn

ASGI server

Frontend / Demonstration Layer

The repository contains a separate forensic demonstration interface.

Technology

Role

React

UI framework

TypeScript

Frontend language

Vite

Frontend build/dev tooling

CSS

UI styling

WebSocket

Runtime event streaming

FastAPI

Local backend API

Uvicorn

Local development server

The forensic UI is intended to visualize system behavior rather than replace the actual inference engine.

Development

Tool

Role

Git

Version control

GitHub

Repository hosting/collaboration

VS Code

Primary development environment

PowerShell

Windows command-line workflow

Virtual environment (env)

Python dependency isolation

Roboflow

Optional dataset collection/management/training experimentation

21. Hardware Environment

The prototype is designed to run on a constrained development laptop.

Representative hardware:

CPU:
AMD Ryzen 5 5600H

GPU:
NVIDIA GeForce RTX 3050 Laptop GPU
4 GB VRAM

RAM:
8 GB

OS:
Windows 11

Because of these constraints, model selection prioritizes:

Inference feasibility

Memory consumption

Training time

Model accuracy

Practical deployment latency

This is one reason lightweight architectures such as MobileNetV2 are attractive for the current prototype.

Large video transformers may provide stronger accuracy but can impose significantly greater computational and memory requirements.

22. Repository Structure

A simplified current repository structure is:

PrivWatch/
│
├── app.py
│
├── privwatch/
│   ├── dataset_loader.py
│   ├── inference.py
│   ├── model.py
│   ├── paths.py
│   ├── privacy_engine.py
│   └── raw_engine.py
│
├── sterile_room/
│   ├── __init__.py
│   ├── alert_registry.py
│   ├── inference_worker.py
│   ├── memory_sanitizer.py
│   ├── sterile_session.py
│   ├── streamlit_integration.py
│   ├── volatile_buffer.py
│   ├── README.md
│   └── THREAT_MODEL.md
│
├── scripts/
│   ├── preprocess.py
│   ├── split_new_data.py
│   ├── train_raw_model.py
│   └── calibration_test.py
│
├── models/
│   └── best_model.pth
│
├── data/
│   ├── raw/
│   ├── split/
│   └── clips/
│
├── calibration/
│   ├── normal/
│   ├── fight/
│   └── collapse/
│
├── PrivWatch-Forensic/
│   ├── backend/
│   └── frontend/
│
└── README.md

Large datasets, virtual environments, generated artifacts, and dependency folders should not be committed to Git.

23. Installation

Clone the repository

git clone <repository-url>
cd PrivWatch

Create a virtual environment

python -m venv env

Activate

.\env\Scripts\Activate.ps1

Install core dependencies

The exact dependency versions should be pinned in the project's requirements file.

Typical packages include:

pip install torch torchvision
pip install numpy opencv-python
pip install av
pip install fastapi uvicorn
pip install streamlit

For CUDA-enabled PyTorch, install the build appropriate for the installed NVIDIA driver and CUDA compatibility.

Do not blindly install a CUDA build from an unrelated tutorial.

Verify GPU availability:

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

24. Preprocessing

After the dataset is correctly organized:

python scripts/preprocess.py

The preprocessing pipeline should produce the expected clip hierarchy under:

data/clips/

Before training, verify:

Class counts.

Train/validation/test separation.

No duplicate clips across splits.

No corrupted source videos.

No accidental mixing of test data into training.

Normal hard negatives are present where intended.

25. Training

Run:

python scripts/train_raw_model.py

The model should be trained from scratch when conducting a clean experiment.

Verify startup output such as:

Training from scratch: no checkpoint loaded

At the end of training, confirm:

models/best_model.pth

was produced or updated.

The best checkpoint should be selected using validation performance.

26. Standard Inference

Example:

python -m privwatch.privacy_inference "example.mp4"

Expected structure:

Privacy model running...

=== MODEL FORENSIC INFO ===
...

FINAL VIDEO CONFIDENCE
Fight: ...
Normal: ...
Collapse: ...

FINAL RESULT: ...
Latency: ... seconds

27. Sterile Room Inference

The Sterile Room should be used when the experiment is specifically testing transient/privacy-oriented execution.

Typical conceptual flow:

Input
 ↓
SterileSession
 ↓
VolatileBuffer
 ↓
Spawn worker
 ↓
ActionModel
 ↓
Result
 ↓
Sanitize
 ↓
Destroy worker

The exact application integration should use the repository's current Sterile Room interfaces rather than duplicating model-loading logic.

28. Demonstration Strategy

For evaluation, the most useful demonstration is not merely showing a prediction.

Show the difference in data lifecycle.

RAW mode

Demonstrate:

Video
  ↓
Inference
  ↓
Persistent recording / file presence

Sterile mode

Demonstrate:

Video
  ↓
RAM buffer
  ↓
Inference worker
  ↓
Alert
  ↓
Buffer overwrite
  ↓
Worker termination

Useful evidence includes:

Process IDs.

Worker creation/destruction.

Buffer occupancy.

Buffer overwrite events.

Memory cleanup events.

Model identity.

Checkpoint hash.

No intentional output video file.

Event-only alert logs.

The visualization should be backed by real runtime telemetry rather than fabricated animations whenever possible.

29. Forensic Demonstration Principle

The project should distinguish between:

Actual runtime evidence

Generated directly by the running system:

worker_spawned
buffer_overwrite
inference_completed
memory_wipe
worker_destroyed

Presentation visualization

A frontend representation of those events.

The frontend should never be treated as proof by itself.

The stronger demonstration is:

Runtime event
      ↓
Backend telemetry
      ↓
Frontend visualization
      ↓
Timestamp / PID / event ID

This makes the visualization auditable.

30. Privacy Guarantees and Limitations

PrivWatch intentionally makes limited claims.

The system attempts to guarantee

No intentional permanent raw-video storage in sterile inference.

Bounded application-level frame buffering.

Best-effort clearing of accessible buffers.

Short-lived worker processes.

Event-oriented persistence.

Reduced surveillance-data lifetime.

The system does NOT guarantee

Protection from a compromised operating system.

Protection from kernel-level memory acquisition.

Guaranteed elimination of Windows pagefile copies.

Guaranteed wiping of every native decoder allocation.

Guaranteed elimination of CUDA-driver memory copies.

Protection against malicious hardware.

Cryptographic destruction of all transient representations.

This distinction is essential for technically honest research.

31. Security Model

The privacy layer is not intended to replace:

OS security

disk encryption

access control

endpoint security

secure boot

hardware security modules

trusted execution environments

Instead, it reduces the amount and lifetime of surveillance information handled by the application.

This follows the principle:

If information does not need to be retained, do not create a persistent copy of it.

32. Research Methodology

The project should be evaluated using controlled experiments.

Experiment A — Model performance

Compare:

MobileNetV2 + LSTM
YOLO-based approach
Video Swin / transformer approach

Measure:

Accuracy

Precision

Recall

F1

Confusion matrix

Latency

RAM

VRAM

Experiment B — Generalization

Use genuinely unseen videos.

Measure:

False positives
False negatives
Anomaly recall
Normal precision

Experiment C — Privacy workflow

Compare:

RAW pipeline
vs
Sterile Room pipeline

Measure:

Persistent video files.

Runtime process lifetime.

Buffer size.

Worker lifetime.

Runtime memory.

Event persistence.

Processing latency.

33. Reproducibility

Every significant training experiment should record:

Experiment ID
Dataset version
Train/validation/test counts
Model architecture
Hyperparameters
Epoch count
Best epoch
Validation metrics
Test metrics
Training timestamp
Checkpoint SHA-256
Software versions
Hardware

This prevents "ghost model" problems where an old checkpoint is accidentally used during inference.

34. Model Lineage Checklist

Before reporting results:

[ ] Dataset version identified
[ ] Dataset split verified
[ ] Training started from scratch where intended
[ ] Checkpoint path verified
[ ] Checkpoint timestamp recorded
[ ] Checkpoint SHA-256 recorded
[ ] Training epoch recorded
[ ] Validation metric recorded
[ ] Inference loads the expected checkpoint
[ ] Inference preprocessing matches training
[ ] Class order verified
[ ] Model architecture verified
[ ] Unseen test set verified

35. Common Failure Modes

Old checkpoint accidentally loaded

Cause:

best_model.pth

was not replaced or training resumed unexpectedly.

Mitigation:

Explicit scratch-training mode.

Checkpoint metadata.

SHA-256 hashing.

Training-run ID.

Forensic startup output.

Dataset leakage

Cause:

The same source video or near-duplicate appears in multiple splits.

Mitigation:

Split at source-video level.

Deduplicate.

Keep test data isolated.

Normal activity classified as anomaly

Possible causes:

Background motion.

Crowds.

Camera movement.

Dataset bias.

Temporal artifacts.

Class imbalance.

Model overconfidence.

Mitigation:

Hard-negative mining.

Diverse normal data.

Threshold calibration.

Temporal smoothing.

Better model architecture.

Scene diversity.

Anomaly classified as normal

This is a false negative.

For safety-oriented deployment this is usually more serious than a false positive.

Mitigation:

Optimize anomaly recall.

Review thresholds.

Increase anomaly diversity.

Evaluate unseen data.

Consider temporal confirmation logic.

36. Ethical Considerations

PrivWatch is intended to explore responsible intelligent surveillance.

Deployment should consider:

Consent.

Legal authorization.

Data protection regulations.

Purpose limitation.

Data minimization.

Retention policies.

Access control.

Human oversight.

False-alarm handling.

Bias and fairness.

Transparency to monitored individuals.

The existence of a privacy-preserving pipeline does not automatically make surveillance lawful or ethical.

37. Responsible Use

PrivWatch should be used for:

Academic research.

Controlled experiments.

Privacy-preserving AI research.

Safety-monitoring prototypes.

Responsible surveillance-system evaluation.

It should not be deployed as an autonomous law-enforcement decision system without appropriate validation, governance, legal review, and human oversight.

38. Future Work

Potential future extensions include:

Better video models

X3D

Video Swin Transformer

MobileNet-based temporal architectures

Efficient video transformers

Lightweight 3D CNNs

Better privacy guarantees

Trusted execution environments.

Encrypted memory.

Secure enclaves.

Hardware-backed isolation.

OS-level memory controls.

Better anomaly detection

Open-set recognition.

One-class anomaly detection.

Continual learning.

Context-aware detection.

Multi-camera reasoning.

Better privacy observability

Runtime memory telemetry.

File-system event monitoring.

Process lifecycle tracing.

Reproducible forensic reports.

Automated integrity verification.

Deployment

Edge devices.

Smart-city gateways.

Hospital edge infrastructure.

Campus safety systems.

Local-only inference appliances.

39. Project Status

PrivWatch is an evolving research prototype.

Current major components:

[✓] Video dataset preparation
[✓] Train / validation / test organization
[✓] Video preprocessing
[✓] MobileNetV2 + LSTM classifier
[✓] Model checkpointing
[✓] Inference pipeline
[✓] Calibration / unseen-video evaluation
[✓] Hard-negative experimentation
[✓] Model lineage / forensic metadata
[✓] Sterile Room architecture
[✓] Volatile circular buffering
[✓] Spawned inference worker
[✓] Best-effort memory cleanup
[✓] Event-only alert design
[~] Comparative model evaluation
[~] Production-grade telemetry
[~] Final frontend demonstration
[~] Comprehensive benchmark study

The [~] items are active research/development areas and should not be represented as fully completed production functionality until verified.

40. Citation

If this project is used in academic work, cite the project's associated paper or repository according to the publication's required citation format.

A future formal citation should identify:

Project:
PrivWatch

Title:
A Non-Persistent Edge-Oriented Video Anomaly Detection Framework
for Privacy-Preserving Surveillance

41. License and Intellectual Property

Unless a license is explicitly included in this repository, users should not assume that the code is released under an open-source license.

The project may contain:

Original source code.

Original system architecture.

Research documentation.

Third-party open-source dependencies.

Publicly sourced datasets.

Trained model weights.

Third-party components remain subject to their respective licenses.

Dataset licenses and usage restrictions must be checked independently before redistribution.

For academic copyright registration, patent evaluation, or institutional submission, maintain:

Source-code history.

Git commits.

Dataset provenance.

Author contribution records.

Training logs.

Model hashes.

Architecture documentation.

Experimental results.

42. Authors

PrivWatch is developed as an academic/research project by:

Yashas M N

Sanjay S

Anshika Prashanth

Archana

Affiliation:

Department of Computer Science and Engineering
RV College of Engineering (RVCE)
Bengaluru, Karnataka, India

43. Final Concept

PrivWatch is fundamentally an experiment in changing the surveillance data lifecycle.

Traditional model:

Observe
  ↓
Record
  ↓
Store
  ↓
Analyze
  ↓
Keep

PrivWatch's intended sterile workflow:

Observe
  ↓
Temporarily buffer
  ↓
Analyze
  ↓
Decide
  ↓
Alert
  ↓
Evaporate visual data
  ↓
Retain only what is necessary

The central engineering principle is:

Surveillance intelligence should not require surveillance permanence.

The project's success should therefore be evaluated on two independent axes:

Detection capability
        +
Data-persistence minimization

A strong PrivWatch implementation is one that can demonstrate both with measurable evidence.
