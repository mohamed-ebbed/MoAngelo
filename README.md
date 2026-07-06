# [3DV 2026] MoAngelo: Motion-Aware Neural Surface Reconstruction for Dynamic Scenes

Official implementation of **MoAngelo: Motion-Aware Neural Surface Reconstruction for Dynamic Scenes**, presented at 3DV 2026.

This codebase is built on top of [NeuralAngelo](https://github.com/nvlabs/neuralangelo).

## Abstract

Dynamic scene reconstruction from multi-view videos remains a fundamental challenge in computer vision. While recent neural surface reconstruction methods have achieved remarkable results in static 3D reconstruction, extending these approaches with comparable quality for dynamic scenes introduces significant computational and representational challenges. Existing dynamic methods focus on novel-view synthesis, therefore, their extracted meshes tend to be noisy. Even approaches aiming for geometric fidelity often result in too smooth meshes due to the ill-posedness of the problem.

We present a novel framework for highly detailed dynamic reconstruction that extends the static 3D reconstruction method NeuralAngelo to work in dynamic settings. To that end, we start with a high-quality template scene reconstruction from the initial frame using NeuralAngelo, and then jointly optimize deformation fields that track the template and refine it based on the temporal sequence. This flexible template allows updating the geometry to include changes that cannot be modeled with the deformation field, for instance occluded parts or the changes in the topology. We show superior reconstruction accuracy in comparison to previous state-of-the-art methods on the ActorsHQ dataset. ([arxiv.org](https://arxiv.org/abs/2509.15892))

---

## TODO

- [x] Release code and instructions for ActorsHQ
- [ ] Add dataset support for [DNA-Rendering](https://dna-rendering.github.io/)
- [ ] Add dataset support for [NHR](https://wuminye.github.io/NHR/)

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Canonical Optimization](#canonical-optimization)
4. [Dynamic Optimization](#dynamic-optimization)
5. [Mesh Extraction](#mesh-extraction)

---

## Installation

### Requirements

- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA 11.8+ (tested on RTX 3090 / A100)
- Python 3.9+
- PyTorch 2.0+

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MoAngelo.git
cd MoAngelo
```

### Step 2 — Create a conda environment

```bash
conda create -n moangelo python=3.10 -y
conda activate moangelo
```

### Step 3 — Install PyTorch

Install PyTorch matching your CUDA version. For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Install tiny-cuda-nn

```bash
pip install setuptools==80.0.0 #necessary to avoid No module named 'pkg_resources' error when compiling tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```
---

## Dataset Preparation

MoAngelo is evaluated on the **[ActorsHQ](https://actorshq.github.io/)** dataset. Download the dataset and place it in a directory referred to as `ACTORSHQ_ROOT`.

The raw dataset is expected to have the following structure:

```
ACTORSHQ_ROOT/
  Actor01/
    Sequence1/
      calibration.csv
      rgbs/
        Cam001/
          Cam001_rgb000001.jpg
          ...
      masks/
        Cam001/
          Cam001_mask000001.png
          ...
    aabbs.csv
  Actor05/
    ...
```

### Convert a single scene

Use the `convert_actorshq_to_json.py` script to convert a raw ActorsHQ sequence into the `transforms.json` format expected by MoAngelo.

```bash
python projects/neuralangelo/scripts/convert_actorshq_to_json.py \
    --dataroot  <ACTORSHQ_ROOT>/<ActorXX>/<SequenceY> \
    --output_dir datasets/<scene_name> \
    --frame_start <START_FRAME> \
    --frame_end   <END_FRAME> \
    --aspect_ratio height_larger
```

The script copies images and masks into `datasets/<scene_name>/images/` and `datasets/<scene_name>/masks/`, and writes a `transforms.json` containing camera calibration, bounding-box geometry, and per-frame metadata.

### Convert all six benchmark scenes (25 frames each)

The six scenes used in our experiments are listed below. Each uses 25 consecutive frames (`frame_end = frame_start + 24`).

| Scene name          | Actor   | Sequence  | Frame start | Frame end |
|---------------------|---------|-----------|-------------|-----------|
| actor1_seq1_140     | Actor01 | Sequence1 | 140         | 164       |
| actor5_seq2_176     | Actor05 | Sequence2 | 176         | 200       |
| actor6_seq2_62      | Actor06 | Sequence2 | 62          | 86        |
| actor6_seq2_180     | Actor06 | Sequence2 | 180         | 204       |
| actor7_seq1_456     | Actor07 | Sequence1 | 456         | 480       |
| actor8_seq2_535     | Actor08 | Sequence2 | 535         | 559       |

Use the provided script to convert all six scenes at once:

```bash
bash convert_datasets.sh --dataroot /path/to/ActorsHQ
```

The script skips any scene whose `datasets/<scene>/transforms.json` already exists. To convert a single scene manually:

```bash
python projects/neuralangelo/scripts/convert_actorshq_to_json.py \
    --dataroot   /path/to/ActorsHQ/Actor01/Sequence1 \
    --output_dir datasets/actor1_seq1_140 \
    --frame_start 140 \
    --frame_end   164 \
    --aspect_ratio height_larger
```

After conversion, `datasets/` should contain one folder per scene, each with:

```
datasets/actor1_seq1_140/
  transforms.json
  images/
    Cam001_rgb000140.png
    ...
  masks/
    Cam001_mask000140.png
    ...
```

---

## Canonical Optimization

The first stage trains a high-quality static reconstruction of the **first frame** of each scene using NeuralAngelo. This produces a canonical template that is used as the starting point for dynamic optimization.

The config template is at [projects/neuralangelo/configs/custom/canonical_experiments.yaml](projects/neuralangelo/configs/custom/canonical_experiments.yaml). It sets `data.canonical: true`, which causes the dataloader to use only frames with `time_step == 1` (i.e., the first frame).

### Run canonical optimization for all scenes

Use the provided script:

```bash
bash run_canonical.sh
```

Or run a single scene manually:

```bash
SCENE=actor1_seq1_140

python train.py \
    --single_gpu \
    --expname=${SCENE}_canonical \
    --logdir=logs/canonical/${SCENE}_canonical \
    --config=projects/neuralangelo/configs/custom/canonical_experiments.yaml \
    --show_pbar \
    --wandb \
    --wandb_name moangelo \
    --data.root=datasets/${SCENE}/
```

Checkpoints and logs are saved under `logs/canonical/<scene>_canonical/`.

### Key config parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data.canonical` | `true` | Use only the first frame for training |
| `data.root` | `datasets/<scene>/` | Path to the converted dataset |
| `model.background.enabled` | `false` | Background model disabled |
| `model.object.sdf.encoding.coarse2fine.init_active_level` | `4` | Coarse-to-fine hash grid start level |

---

## Dynamic Optimization

After canonical optimization, run the iterative dynamic optimization over all frames using the config at [projects/neuralangelo/configs/custom/iterative_exps_bg_25.yaml](projects/neuralangelo/configs/custom/iterative_exps_bg_25.yaml).

You must set `model.object.canonical_dir` to point to the canonical checkpoint:

```bash
SCENE=actor1_seq1_140
CANONICAL_CKPT=logs/canonical/${SCENE}_canonical/<checkpoint>.pt

python train.py \
    --single_gpu \
    --expname=${SCENE}_dynamic \
    --logdir=logs/dynamic/${SCENE}_dynamic \
    --config=projects/neuralangelo/configs/custom/iterative_exps_bg_25.yaml \
    --show_pbar \
    --wandb \
    --wandb_name moangelo \
    --data.root=datasets/${SCENE}/ \
    --model.object.canonical_dir=${CANONICAL_CKPT}
```

The training loop iterates over each frame from `time_step=2` to `data.num_frames`, saving a per-frame checkpoint.

---

## Mesh Extraction

Extract meshes from all per-frame checkpoints of a trained dynamic experiment:

```bash
bash mesh_extractor.sh logs/dynamic/<scene>_dynamic
```

This saves one `.ply` file per frame under `logs/dynamic/<scene>_dynamic/meshes/` and produces a `meshes.zip` archive.

To extract a single mesh manually:

```bash
python extract_mesh.py \
    --config=logs/dynamic/<scene>_dynamic/<frame_folder>/config.yaml \
    --checkpoint=logs/dynamic/<scene>_dynamic/<frame_folder>/<checkpoint>.pt \
    --output_file=mesh.ply \
    --resolution=2048 \
    --block_res=128
```
