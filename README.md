# NeurIPS Codebase

This repository contains the official codebase for our NeurIPS submission.

## Installation

It is recommended to use a virtual environment or conda environment. Then, install the required dependencies:

```bash
conda env create -f environment.yml
```



## Workspace Organization
- `src/`: Core shared libraries.
  - `src/layers/`: Contains critical neural network layer implementation (`GGDLinear`, `GGMLinear`), mathematical components, and quantizers.
  - `src/models/`: Contains model construction (e.g. `ViT`).
  - `src/data/`: Handles dataset processing (e.g. CIFAR-10 dataloading).
  - `src/training/`: Training and evaluation loops.
  - `src/utils/`: Configuration and helper functions.
- `CNNs/`: Convolutional Neural Network experiments. Contains subdirectories for CIFAR-10 (ResNet18, ResNet20, VGG16) and ImageNet. 
- `ViTs/`: Vision Transformer experiments on Cifar10 and Imagenet1k
- `Other_Experiments/`: Additional experiment tracks including:
  - `model size/`: Notebooks and scripts for analyzing varied model capacities.
  - `nlp/` & `timeseries/`: Application of our methods to specialized domains outside of standard vision tasks.
  - `perturbation/`: Experiments relating to weight perturbation (e.g. adabin, ggd, irnet, xnornet).
  - `projection_expansion_ratio_Gaussian_vs_Rademacher/`: Mathematical ablations comparing different projection mappings (Gaussian vs Rademacher for ResNets).
  - `resampleg/`: Scripts testing G-matrix resampling.
