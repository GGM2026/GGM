# GGM: NeurIPS submission codebase

This repository contains the code for every experiment reported in the paper:
CNN and Vision Transformer training on CIFAR-10, TinyImageNet, and ImageNet-1k;
the projection and resampling ablations; and the NLP and time-series transfer
experiments.

Every command below states the directory it must be run from. Paths inside each
command are relative to that directory, so the sequence can be copied verbatim
from a fresh checkout.

---

## Experiments

Each row is one experiment: the directory to run it from, the single command
that starts it, and the headline GGM result. Install first (Section 1), then
jump to the section for full options and the complete comparison table.

| # | Experiment | Run from | Command | GGM result |
|---|---|---|---|---:|
| [2.1](#21-cifar-10-resnet-18-resnet-20-vgg-16) | CIFAR-10 CNNs | `CNNs/Cifar10/ResNet18` | `bash run.sh` | 93.58 |
| [2.2](#22-tinyimagenet-resnet-18-and-resnet-34) | TinyImageNet CNNs | `CNNs/Tinyimagenet` | `bash scripts/run_tinyimagenet.sh` | 59.84 |
| [2.3](#23-imagenet-1k-resnet-18-and-resnet-34) | ImageNet CNNs | `CNNs/ImageNet` | `bash scripts/run_imagenet.sh` | 63.54 |
| [3.1](#31-cifar-10-original-research-pipeline) | CIFAR-10 ViT | `ViTs/Cifar10` | `bash run_cifar10.sh` | 86.51 |
| [3.2](#32-imagenet-1k-vit-small) | ImageNet ViT-Small | `ViTs/Imagenet/vit_small` | `bash run.sh` | 56.18 |
| [3.3](#33-imagenet-1k-bhvit) | ImageNet BHViT | `ViTs/Imagenet/bhvit` | `python train_bhvit_imagenet.py` | 65.34 |
| [5.1](#51-glue-with-a-binarized-bert) | GLUE, binarized BERT | `Other Experiments/nlpggm` | `bash run_ggm.sh` | 89.56 |
| [5.2](#52-time-series-classification) | Time series | `Other Experiments/timeseries/adftd_run` | `bash setup_data.sh && bash ggmrun.sh` | 53.72 |

Ablations are in [Section 4](#4-ablations): projection distribution and
expansion ratio, resampling, model size, and weight perturbation.

Datasets are handled per experiment. CIFAR-10 and TinyImageNet download on first
run; ImageNet, GLUE, ADFTD, and PTB-XL are set up by the instructions in their
own sections.

## Reference

| Section | Contents |
|---|---|
| [1. Installation](#1-installation) | Conda or virtual environment, then PyTorch and `requirements.txt` |
| [2. CNN experiments](#2-cnn-experiments) | ResNet-18, ResNet-20, VGG-16 on CIFAR-10; ResNet-18/34 on TinyImageNet and ImageNet |
| [3. Vision Transformer experiments](#3-vision-transformer-experiments) | GGM ViT on CIFAR-10 and ImageNet-1k |
| [4. Ablations](#4-ablations) | Projection distribution, expansion ratio, resampling, model size, perturbation |
| [5. Transfer to other domains](#5-transfer-to-other-domains) | GLUE and time-series classification |

## Repository layout

```text
CNNs/                     CNN training pipelines
  Cifar10/                ResNet18, ResNet20, vgg16 (one self-contained dir each)
  Tinyimagenet/           distributed ResNet training plus a dataset preparation script
  ImageNet/               distributed ResNet-18 and ResNet-34 training
ViTs/                     Vision Transformer training and evaluation
  Cifar10/                CIFAR-10 ViT research pipeline (Section 3.1)
  Imagenet/               ImageNet-1k Vision Transformers, two different models
    vit_small/            ViT-Small (Section 3.2)
    bhvit/                BHViT (Section 3.3)
Other Experiments/        ablations and domain-transfer studies
requirements.txt          all Python dependencies for every experiment
environment.yml           conda interpreter definition
```

---

## 1. Installation

Python 3.10 or newer, and a CUDA-capable GPU for training.

PyTorch must be installed first, from the index matching your GPU, because
`requirements.txt` contains packages that depend on it.

```bash
conda env create -f environment.yml
conda activate GGM_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Without conda, replace the first two lines with a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

`requirements.txt` covers every experiment; there is nothing to install per
directory.

---

## 2. CNN experiments

### 2.1 CIFAR-10: ResNet-18, ResNet-20, VGG-16

Each architecture is a self-contained directory. Run from the directory of the
architecture you want. CIFAR-10 downloads automatically on the first run.

ResNet-18:

```bash
cd CNNs/Cifar10/ResNet18
python main_cifar10.py \
  --data_root ../../../data/cifar10 \
  --size 18 --N_scale 5.0 --seed 6 \
  --batch_size 256 --epochs 300 \
  --base_lr 3e-3 --weight_decay 0.02 --label_smoothing 0.05 \
  --num_workers 8 --amp --no-compile --use_ggm \
  --run_name cifar10_resnet18_N5p0_seed6 \
  --checkpoint_dir ./checkpoints \
  --results_file ./results/nscale_5p0.csv
```

ResNet-20:

```bash
cd CNNs/Cifar10/ResNet20
python main_cifar10.py \
  --data_root ../../../data/cifar10 \
  --size 20 --N_scale 5.0 --seed 6 \
  --batch_size 256 --epochs 300 \
  --base_lr 3e-3 --weight_decay 0.02 --label_smoothing 0.05 \
  --num_workers 8 --amp --no-compile --use_ggm \
  --run_name cifar10_resnet20_N5p0_seed6 \
  --checkpoint_dir ./checkpoints \
  --results_file ./results/nscale_5p0.csv
```

VGG-16:

```bash
cd CNNs/Cifar10/vgg16
python main_cifar10.py \
  --data_root ../../../data/cifar10 \
  --size 16 --N_scale 2.5 --seed 13 \
  --batch_size 256 --epochs 300 \
  --base_lr 3e-3 --weight_decay 0.05 --label_smoothing 0.05 \
  --num_workers 8 --amp --no-compile --use_ggm \
  --run_name cifar10_vgg16_N2.5_seed13 \
  --checkpoint_dir ./checkpoints \
  --results_file ./results/nscale_2.5.csv
```

`--N_scale` is the projection expansion ratio. Add `--no-ggm` for the
full-precision baseline. Per-epoch metrics are appended to `--results_file`.

Each directory also contains `run.sh`, which repeats the same configuration
across consecutive seeds and writes one CSV per expansion ratio. Run it from the
architecture directory:

```bash
cd CNNs/Cifar10/ResNet18
bash run.sh
```

The committed sweeps in `CNNs/Cifar10/*/results/` were produced this way, with
one file per `N_scale` value from 1.0 to 5.0.

**Expected results (paper Table 6).** `--N_scale` is the projection expansion ratio, ν.
GGM rows are averages over 5 runs. VGG-16 baselines were reproduced from the
authors' public implementations, also averaged over 5 runs.

| Architecture | FP (32/32) | Strongest prior 1-bit | GGM (1/1) x 1.0 | GGM (1/1) x 2.5 |
|---|---:|---:|---:|---:|
| ResNet-18 | 94.80 | BiPer 93.75, AdaBin 93.10 | 93.58 | **94.51** |
| ResNet-20 | 92.10 | AdaBin 88.20, RBNN 87.80 | 88.05 | **89.74** |
| VGG-16 | 94.04 | BiPer 91.95, AdaBin 90.26 | 91.47 | **93.12** |



Training settings behind these numbers: 300 epochs, AdamW with a OneCycle
schedule, maximum learning rate 3e-3, batch size 256, weight decay 0.02, label
smoothing 0.05, mixed precision, and a seeded random 10 percent validation split
with the best checkpoint selected by validation accuracy. The stem convolution
and classification head stay full precision; every other hidden layer, including
downsampling layers, is binarized.

### 2.2 TinyImageNet: ResNet-18 and ResNet-34

The run script downloads and restructures TinyImageNet before training, so no
manual dataset setup is needed. It uses every visible GPU by default; override
with `NPROC_PER_NODE`.

```bash
cd CNNs/Tinyimagenet
N_SCALE=1 SIZE=18 bash scripts/run_tinyimagenet.sh
```

`SIZE`, `N_SCALE`, `K_BITS_X`, and `K_BITS_W` select the row of the table below.
The equivalent direct invocation:

```bash
cd CNNs/Tinyimagenet
python scripts/prepare_tinyimagenet.py ./data
torchrun --nproc_per_node=1 train.py \
  --dataset tinyimagenet --data_root ./data \
  --model resnet --size 18 --N_scale 1 \
  --k_bits_x 1 --k_bits_w 1 --scale_policy learnable_mean \
  --img_size 64 --epochs 90 --batch_size 128 --num_workers 8 \
  --base_lr 0.001 --weight_decay 0.05 --label_smoothing 0.1 \
  --mixup 0.8 --cutmix 1.0 --drop_path 0.1 --num_runs 1 --amp \
  --ckpt_dir ./checkpoints --run_name ggm_resnet_18_N1
```

The preparation script downloads `tiny-imagenet-200.zip`, extracts it to
`./data`, and reorganizes the flat validation directory into one folder per
class, which is the layout the loader expects.

**Expected results.**

| Architecture | Precision | ν = 1 | ν = 2 | ν = 3 | FP |
|---|---|---:|---:|---:|---:|
| ResNet-18 | 1/1 | 59.84 | 63.48 | **64.10** | 65.48 |
| ResNet-18 | 1.58/1.58 | 63.75 | | | 65.48 |
| ResNet-34 | 1/1 | 64.46 | 66.54 | **67.08** | 67.85 |
| ResNet-34 | 1.58/1.58 | 66.93 | | | 67.85 |

Set `K_BITS_X=2 K_BITS_W=2` for the 1.58/1.58 rows: the quantizer uses
`Q = 2^(k_bits-1) - 1`, so `k_bits=2` gives the three symmetric levels
`{-s, 0, +s}`, which is log2(3) = 1.58 bits.

Training settings: 90 epochs, per-process batch 128, AdamW at base learning rate
1e-3, weight decay 0.05, label smoothing 0.1, drop-path 0.1, Mixup 0.8,
CutMix 1.0, OneCycle with 10 percent warmup, ReLU, and mixed precision. The 7x7
stride-2 stem is replaced by 3x3 stride-1 with max-pooling removed for the 64x64
input; the stem, classifier, and all batch-normalization layers stay full
precision, and every other convolution including downsampling is binarized.


### 2.3 ImageNet-1k: ResNet-18 and ResNet-34

Place ImageNet in `CNNs/imagenet_download` in `torchvision.datasets.ImageNet`
layout, then run from `CNNs/ImageNet`.

```bash
cd CNNs/ImageNet
bash scripts/run_imagenet.sh
```

To change the data location, model, or process count, edit `DATA_ROOT`,
`MODEL`, `SIZE`, `N_FACTOR`, and `NPROC_PER_NODE` at the top of
`scripts/run_imagenet.sh`. The equivalent direct invocation is:

```bash
cd CNNs/ImageNet
torchrun --nproc_per_node=1 train.py \
  --dataset imagenet --data_root ../imagenet_download \
  --model resnet --size 18 --N_factor 1.0 --prelu \
  --img_size 224 --epochs 200 --batch_size 256 --num_workers 16 \
  --base_lr 3e-3 --weight_decay 0 --label_smoothing 0.1 \
  --optimizer adamw --seed 1337 --num_runs 1 --amp \
  --ckpt_dir ./checkpoints/imagenet --run_name resnet18_ggm_nu1.0
```

`--full_precision` disables GGM layers for the full-precision baseline.

Training logs for the runs behind Table 8 are in this directory as
`train_resnet{18,34}_ggm_nu{1,2,3}.0_base.log`, with separate ResNet-18 test
evaluations in `test_resnet18_ggm_nu{1,2,3}.0_base.log`. Each log ends with a
`[TEST][CHOSEN]` line carrying the reported top-1.

---

**Expected results (paper Table 8).** All GGM rows are 1-bit weights and
activations, `(1/1) x ν`.

| Network | Method | Config | Top-1 |
|---|---|---|---:|
| ResNet-18 | FP | 32/32 | 69.60 |
| | XNOR-Net | (1/1) x 1.0 | 51.20 |
| | IR-Net | (1/1) x 1.0 | 58.10 |
| | RBNN | (1/1) x 1.0 | 59.90 |
| | ReCU | (1/1) x 1.0 | 61.00 |
| | AdaBin | (1/1) x 1.0 | 63.10 |
| | BiPer* | (1/1) x 1.0 | 61.40 |
| | **GGM (ours)** | (1/1) x 1.0 | **63.54** |
| | **GGM (ours)** | (1/1) x 2.0 | **66.39** |
| | **GGM (ours)** | (1/1) x 3.0 | **67.42** |
| | **GGM (ours)** | (1/1) x 5.0 | **66.80** |
| ResNet-34 | FP | 32/32 | 73.30 |
| | IR-Net | (1/1) x 1.0 | 62.90 |
| | RBNN | (1/1) x 1.0 | 63.10 |
| | ReCU | (1/1) x 1.0 | 65.10 |
| | AdaBin | (1/1) x 1.0 | 66.40 |
| | BiPer* | (1/1) x 1.0 | 65.73 |
| | **GGM (ours)** | (1/1) x 1.0 | **68.89** |
| | **GGM (ours)** | (1/1) x 2.0 | **70.92** |
| | **GGM (ours)** | (1/1) x 3.0 | **71.54** |
| | **GGM (ours)** | (1/1) x 5.0 | **71.78** |

`*` BiPer uses a two-stage training procedure.

At ν = 1.0, GGM exceeds the strongest prior on both networks: 63.54 against
AdaBin's 63.10 on ResNet-18, and 68.89 against AdaBin's 66.40 on ResNet-34.

The script above sets `--N_factor 1.0`, reproducing the 63.54 row. Set
`--N_factor 2.0`, `3.0`, or `5.0` for the remaining rows, and `--size 34` for
ResNet-34.

Training settings: 200 epochs, AdamW at maximum learning rate 3e-3, weight decay
0, label smoothing 0.1, mixed precision, seed 1337, batch size 256, PReLU
activations, and a OneCycle cosine schedule stepped every optimizer update. Stem
convolution, downsampling layers, and classifier stay full precision. The runs
behind the table above were single-GPU (`NPROC_PER_NODE=1`); raise it to shard
the same total batch across more devices.

---

## 3. Vision Transformer experiments

### 3.1 CIFAR-10, original research pipeline

The pipeline the package above was distilled from. It must be run from its own
directory, because it resolves its data and results paths from the working
directory.

```bash
cd ViTs/Cifar10
bash run_cifar10.sh
```

`parameters_defaults.txt` records the original hardcoded values for every
argument now exposed on the command line.

**Expected results (paper Tables 1 and 5).** This pipeline uses the controlled
single-stage recipe behind the paper's CIFAR-10 ViT study, so its rows are the
ones to compare against. `--n_factor` is ν and `--k_bits_x` / `--k_bits_w` are
the activation and weight bit-widths.

| Configuration | Expected Top-1 |
|---|---:|
| Full precision (32/32) | 92.49 |
| GGM (1/1) x 1.0 | 86.51 |
| GGM (1/1) x 1.5 | 88.49 |
| GGM (1/1) x 2.0 | 89.46 |
| GGM (1/1) x 2.5 | 90.09 |
| GGM (1/1) x 3.0 | **91.31** |
| GGM (1.58/1.58) x 1.0 | 90.59 |
| GGM (1.58/1.58) x 1.5 | 91.14 |
| GGM (1.58/1.58) x 2.0 | **91.92** |

`run_cifar10.sh` sets `--k_bits_x 1 --k_bits_w 1 --n_factor 1`, so it reproduces
the 86.51 row. Controlled STE baselines under the identical backbone and budget:
STE-AMAX 57.52, STE-AMAX+G 58.64, XNOR+G 78.96, LSCALE+G 78.96, AdaBin+G 79.39,
BiPer-LF+G 77.12. The strongest prior result on this backbone is 82.35 with
two-stage training, or 77.77 under the single-stage budget used here.

Training settings: 450 epochs, AdamW, batch 512, learning rate 5e-4, weight decay
0, OneCycle with 15 percent warmup then cosine annealing, no EMA, random
horizontal flips, random resized crops over scale [0.8, 1.0], RandAugment with 2
operations at magnitude 9, per-channel normalization at mean and standard
deviation 0.5, and MixUp or CutMix at batch level with probability 0.5 and
alpha 1.0. Backbone: patch 4, D = 192, 12 blocks, 3 heads, MLP ratio 4.

### 3.2 ImageNet-1k, ViT-Small

`ViTs/Imagenet/` holds two different models: the ViT-Small in `vit_small/` and
the BHViT in `bhvit/` (Section 3.3). The ViT-Small beats competing approaches at
expansion ratio 1.

Evaluating the released checkpoint requires no training. ImageNet-1k is gated on
the Hugging Face Hub, so authenticate once and accept the dataset terms first.

```bash
cd ViTs/Imagenet/vit_small
hf auth login
python prepare_imagenet.py --output ./imagenet --splits validation
python evaluate.py --data_dir ./imagenet/val
```

Expected: 56.18 percent top-1 and 79.61 percent top-5 over the 50,000 validation
images. `bash run.sh` chains these three steps.

To check that the checkpoint loads and predicts sensibly without touching the
gated dataset, evaluate on the ten-class Imagenette subset, which downloads
without a login:

```bash
cd ViTs/Imagenet/vit_small
python evaluate.py --imagenette
```

This gives 59.92 percent top-1 and 84.43 percent top-5. Imagenette is an easier
ten-class task and is not comparable to the 1000-class result.

To train from scratch:

```bash
cd ViTs/Imagenet/vit_small
python prepare_imagenet.py --output ./imagenet --splits train,validation
python main.py                          # single GPU
torchrun --nproc_per_node=<N> main.py   # multiple GPUs
```

Training runs 300 epochs at batch size 256 with AdamW, a warmup and two-stage
cosine schedule, label smoothing 0.1, mixed precision, and seed 42. Checkpoints
go to `./runs`, and a run resumes automatically from `runs/last.pth` if present.
`bash run.sh train` chains the download and training steps.

**Expected results.** Evaluating the released checkpoint gives 56.18 percent
top-1 and 79.61 percent top-5 over the 50,000 validation images, and 59.92 /
84.43 on the ten-class Imagenette check.

### 3.3 ImageNet-1k, BHViT

The second of the two ImageNet models, alongside the ViT-Small of Section 3.2.
It trains from scratch and needs ImageNet already on disk.

```bash
cd ViTs/Imagenet/bhvit
DATA_DIR=/path/to/imagenet python train_bhvit_imagenet.py
```

Run it from `ViTs/Imagenet/bhvit`; the script resolves `src/` relative to its own
location and expects that directory to be the working directory. `DATA_DIR`
defaults to `ViTs/Imagenet/bhvit/data/imagenet` and expects `train` and `val`
subdirectories with one folder per class.

---

**Expected results (paper Tables 3 and 10).** This is the BHViT-based pipeline
behind the paper's ImageNet Transformer results.

| Method | Configuration | Top-1 |
|---|---|---:|
| BHViT (full precision) | 32/32 | 78.50 |
| BHViT baseline | (1/1) x 1.0 | 64.00 |
| GGM (ours) | (1/1) x 1.0 | 65.34 |
| GGM (ours) | (1.58/1.58) x 4.0 | **75.50** |

Training settings: 200 epochs, AdamW at learning rate 5e-4, betas (0.9, 0.98),
cosine decay with 5 warmup epochs, label smoothing 0.1, weight decay 0, mixed
precision, and distributed data parallel across two H100 GPUs at batch 256 per
GPU (total 512). Augmentation is random resized crop, random horizontal flip,
RandAugment, random erasing, ImageNet normalization, MixUp alpha 0.8, and CutMix
alpha 1.0. Patch stem, downsampling blocks, positional embeddings, normalization,
RPReLU, attention softmax, pooling, residual paths, and classifier stay full
precision.
## 4. Ablations

### 4.1 Projection distribution and expansion ratio

Compares Gaussian against Rademacher projections at matched expansion ratios on
CIFAR-10, for both ResNet-18 and ResNet-20. Four self-contained directories; run
from whichever you want.

```bash
cd "Other Experiments/projection_expansion_ratio_Gaussian_vs_Rademacher/Gaussian_ResNet18"
bash run.sh
```

The same applies to `Gaussian_ResNet20`, `Rademacher_ResNet18`, and
`Rademacher_ResNet20`. To run a single configuration instead of the seed sweep:

```bash
cd "Other Experiments/projection_expansion_ratio_Gaussian_vs_Rademacher/Gaussian_ResNet18"
python main_cifar10.py \
  --data_root ../../../data/cifar10 \
  --size 18 --N_scale 1.0 --seed 6 \
  --batch_size 256 --epochs 300 --base_lr 3e-3 \
  --weight_decay 0.02 --label_smoothing 0.05 \
  --num_workers 8 --amp --no-compile --use_ggm \
  --run_name gaussian_r18_N1p0 --checkpoint_dir ./checkpoints \
  --results_file ./results/nscale_1p0.csv
```

Sweep `--N_scale` over 1.0 to 5.0 to reproduce the expansion-ratio curve.

**Expected results (paper Figure 6 and Figure 9).** Gaussian and Rademacher
embeddings track each other closely across the sweep, which is the point of the
ablation: the cheaper Rademacher projection costs no measurable accuracy.

| Architecture | ν = 1 | ν = 2 | ν = 3 | ν = 4 | ν = 5 | FP |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 93.55 | 94.20 | 94.65 | 94.77 | 94.76 | 94.80 |
| ResNet-20 | 88.04 | 89.36 | 90.04 | 90.45 | 90.67 | 92.10 |

Settings: 300 epochs, base learning rate 3e-3 with OneCycle, weight decay 0.02,
label smoothing 0.05, batch 256, and 10 seeds per expansion ratio. The stem
convolution and classification head stay full precision.

### 4.2 Projection resampling

Compares a fixed projection against redrawing it during training.

```bash
cd "Other Experiments/resampleg"
python main_regular.py     # fixed projection
python main_resample.py    # projection resampled during training
python plot.py             # figure from both runs
```

These scripts hold their configuration in a `Config` class at the top of each
file rather than on the command line, and download CIFAR-10 into
`Other Experiments/resampleg/data`.

### 4.3 Model size and accuracy-memory tradeoff

```bash
cd "Other Experiments/model size"
python main_cifar10.py \
  --data_root ../../data/cifar10 \
  --size 18 --N_scale 1.0 --seed 1 \
  --batch_size 256 --epochs 300 --base_lr 3e-3 \
  --num_workers 8 --amp --no-compile --use_ggm \
  --run_name modelsize_r18 --checkpoint_dir ./checkpoints \
  --results_file ./results/r18.csv
```

Note the two-level `../../data/cifar10`, which resolves to the same shared
repository-root data directory used elsewhere. `Cifar10.ipynb` builds
`accuracy_memory_tradeoff.pdf` from the collected runs.

**Expected results (paper Figure 9).** The tradeoff these runs are meant to
produce:

| Architecture | Configuration | Memory | Top-1 |
|---|---|---:|---:|
| ResNet-20 | FP | 1.09 MB | 92.10 |
| ResNet-20 | GGM ν = 1.0 | 0.05 MB | 88.04 |
| ResNet-20 | GGM ν = 5.0 | 0.18 MB | 90.67 |
| ResNet-18 | FP | 44.71 MB | 94.80 |
| ResNet-18 | GGM ν = 1.0 | 1.48 MB | 93.55 |
| ResNet-18 | GGM ν = 5.0 | ~4 MB | 94.76 |

### 4.4 Weight perturbation

Notebooks comparing perturbation behaviour across binarization methods. Launch
Jupyter from the repository root and open one of:

```text
Other Experiments/perturbation/ggm/ggm.ipynb
Other Experiments/perturbation/adabin/adabin.ipynb
Other Experiments/perturbation/irnet/irnet.ipynb
Other Experiments/perturbation/xnornet/xnornet.ipynb
```

Each notebook is self-contained and includes its own layer definitions.

---

## 5. Transfer to other domains

### 5.1 GLUE with a binarized BERT

Requires two external resources that are not distributed with this repository:
the GLUE task data and the fine-tuned DynaBERT teacher and student checkpoints.
Place them so the paths at the top of `run_ggm.sh` resolve, then:

```bash
cd "Other Experiments/nlpggm"
bash run_ggm.sh
```

`run_ggm.sh` runs SST-2 with one-bit weights, embeddings, and activations and
all five distillation terms enabled. `RATIO` at the top of the script selects
the projection expansion ratio and is passed as `--ggm_ratio`; it ships at 1.0,
so set `RATIO=3.0` to reproduce the strongest row below. Edit `TASK_NAME`,
`TASK_DIR`, `TEACHER_MODEL`, and `STUDENT_MODEL` in the same block to select a
different task. `ggm_eval.sh` evaluates a trained checkpoint.

The per-task scripts under `scripts/` record the exact hyperparameters used for
each GLUE task. They reference absolute paths from the original training
machine, so copy the hyperparameters into `run_ggm.sh` rather than running them
directly.

**Expected results (paper Table 11).** GGM is built on top of the BiBERT
training setup. Configurations read as (weight/activation bits) x projection
expansion ratio, where the bit-widths cover weights, embeddings, and
activations. Set the ratio with `RATIO` at the top of `run_ggm.sh`, which passes
it as `--ggm_ratio`.

| Method | Config | SST-2 | QNLI |
|---|---|---:|---:|
| Full precision | 32/32 | 93.20 | 92.10 |
| BinaryBERT | (1/1) x 1.0 | 53.20 | 51.50 |
| BERT-1bit* | (1/1) x 1.0 | 77.60 | 66.40 |
| BiBERT | (1/1) x 1.0 | 88.70 | 72.60 |
| GGM (ours) | (1/1) x 1.0 | 89.56 | 79.24 |
| GGM (ours) | (1/1) x 2.0 | 90.60 | 86.85 |
| GGM (ours) | (1/1) x 3.0 | **91.39** | **88.54** |

`*` BERT-1bit is an XNOR-style fully binarized BERT.

Training settings: BertAdam with linear warmup then linear decay, learning rate
1e-4, warmup ratio 0.1, weight decay 0.01 on non-bias and non-normalization
parameters and 0 elsewhere, batch size 32, 50 epochs, maximum sequence length 64
for SST-2 and 128 for QNLI. Evaluation runs every 200 steps for SST-2 and every
1000 for QNLI, selecting the best checkpoint by validation performance. The
strongest variant replaces LayerNorm with BinaryNorm and GELU with ReLU, and
keeps the attention products QK^T and AV full precision.

### 5.2 Time-series classification

Two independent experiments, one directory each. Every directory carries its own
`setup_data.sh`, its own preprocessing, and its own run scripts, so the two never
share files or paths.

| Directory | Dataset | Source | Task |
|---|---|---|---|
| `Other Experiments/timeseries/adftd_run` | ADFTD | OpenNeuro `ds004504` | 3-class EEG: 36 Alzheimer's, 23 frontotemporal dementia, 29 control |
| `Other Experiments/timeseries/ptbxlrun` | PTB-XL | PhysioNet 1.0.3 | 12-lead clinical ECG at 500 Hz, about 1.7 GB |

Both datasets are public and need no login.

**PTB-XL.** `setup_data.sh` downloads the archive, extracts it to `./ptb-xl`, and
runs `data_preprocessing/ptbxl_preprocess.py` to write `./PTB-XL/Feature` and
`./PTB-XL/Label/label.npy`, which is where the run scripts point.

```bash
cd "Other Experiments/timeseries/ptbxlrun"
bash setup_data.sh       # download + preprocess, about 1.7 GB
bash ggmrun.sh           # GGM
bash fprun.sh            # full-precision baseline
bash xnorrun.sh          # XNOR-Net
bash adabinrun.sh        # AdaBin
bash dorefarun.sh        # DoReFa
```

**ADFTD.** `setup_data.sh` downloads the raw EEG for all 88 subjects into
`../ADFTD_raw` and prepares `../ADFTD/Feature` and `../ADFTD/Label/label.npy`,
where the run scripts point. Labels are 0 = control, 1 = frontotemporal
dementia, 2 = Alzheimer's.

```bash
cd "Other Experiments/timeseries/adftd_run"
bash setup_data.sh       # download + preprocess
bash ggmrun.sh
bash fprun.sh
bash xnorrun.sh
bash adabinrun.sh
bash dorefarun.sh
```

Both setup scripts are idempotent, skip work already done, and finish by printing
whether each expected path exists. If a dataset is already prepared, point the
run scripts at it instead: ADFTD reads `--root_path ../ADFTD/` and PTB-XL reads
`--root_path ./PTB-XL`.

Both experiments use the Medformer backbone with six encoder layers and identical
patch lengths across methods, so the binarization method is the only variable.
Results land in `results/classification/<model_id>/`, one directory per method.
`experiments.ipynb` and `supplement_experiments.ipynb` in each directory collect
them into the reported tables.

---

**Expected results (paper Table 12).** Both datasets use the Medformer backbone
at ν = 5.0, reported as mean and standard deviation over 5 seeds.

| Method | Config | PTB-XL | ADFTD |
|---|---|---:|---:|
| Medformer (full precision) | 32/32 | 72.87 ± 0.23 | 53.27 ± 1.54 |
| DoReFa-Net | (1/1) x 1.0 | 72.49 ± 0.06 | 52.41 ± 1.65 |
| AdaBin | (1/1) x 1.0 | 72.26 ± 0.18 | 49.26 ± 2.44 |
| XNOR-Net | (1/1) x 1.0 | 71.91 ± 0.12 | 51.17 ± 1.75 |
| GGM (ours) | (1/1) x 5.0 | **73.57 ± 0.42** | **53.72** |

Training settings: 6 encoder blocks, embedding dimension 128, feed-forward
dimension 256, Adam at learning rate 1e-4, cross-entropy loss, stochastic weight
averaging, up to 100 epochs, and early stopping with patience 10. PTB-XL uses
sequence length 250 at batch 256; ADFTD uses the dataset-inferred sequence length
at batch 128. The encoder blocks use BinaryNorm in place of LayerNorm, while the
embedding module, classification head, and training recipe follow the original
Medformer design.

---

## Reported results

Headline GGM numbers at expansion ratio 1.0, one row per experiment in this
repository. Full comparison tables, including every baseline, are in the section
listed on the right.

| Experiment | Metric | GGM (1/1) x 1.0 | Section |
|---|---|---:|---|
| CIFAR-10, ResNet-18 | Top-1 | 93.58 | 3.1 |
| ImageNet-1k, ResNet-18 | Top-1 | 63.54 | 3.3 |
| ImageNet-1k, ResNet-34 | Top-1 | 68.89 | 3.3 |
| CIFAR-10, ViT | Top-1 | 86.51 | 4.1 |
| ImageNet-1k, ViT-Small (released checkpoint) | Top-1 / top-5 | 56.18 / 79.61 | 4.2 |
| ImageNet-1k, BHViT | Top-1 | 65.34 | 4.3 |
| GLUE SST-2, binarized BERT | Accuracy | 89.56 | 6.1 |
| ADFTD, Medformer | Accuracy | 53.72 | 6.2 |
