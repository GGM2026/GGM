# GGM Vision Transformer (ImageNet-1k)

Vision Transformer (ViT-Small, 16×16 patches, 224×224 input, ~22M parameters)
with GGM binary-projection linear layers (`LinearGGM`). The released checkpoint
stores its weights as packed 1-bit tensors (4.27 MiB) and attains **56.19% top-1
/ 79.61% top-5** on the ImageNet-1k validation set.

## Requirements

Python 3.10 and a CUDA GPU.

```bash
cd ViTs/Imagenet/vit_small
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The repository-level `environment.yml` (conda) installs the same stack; if used,
add `pip install datasets`.

## Layout

- `model.py`, `linearggm.py`, `layer_utils.py` — model and GGM layer definitions.
- `train_utils.py` — data handling and the training/evaluation loops.
- `main.py` — training entry point and configuration.
- `evaluate.py` — evaluate the released checkpoint.
- `prepare_imagenet.py` — download ImageNet-1k into the expected ImageFolder layout.
- `runs/best_seed_G_frozen_Wb_1bit_fp16.pth` — pretrained 1-bit checkpoint.
- `imagenet_checkpoint_eval.ipynb` — notebook equivalent of `evaluate.py`.

## Evaluation

### Full validation set

ImageNet-1k is gated on the Hugging Face Hub. Authenticate once (create a read
token at `huggingface.co/settings/tokens`, accept the terms at
`huggingface.co/datasets/ILSVRC/imagenet-1k`), then:

```bash
hf auth login
python prepare_imagenet.py --splits validation      # -> ./imagenet/val (50k images)
python evaluate.py --data_dir ./imagenet/val
```

This reports 56.19% top-1 / 79.61% top-5.

If ImageNet is already available locally, evaluate it directly and skip the
download:

```bash
python evaluate.py --data_dir /path/to/imagenet/val
```

### Sanity check without ImageNet

For a quick check that the checkpoint loads and predicts correctly, evaluate on
the 10-class Imagenette subset (downloaded automatically, no authentication):

```bash
python evaluate.py --imagenette
```

This yields 59.92% top-1 / 84.43% top-5; the subset is easier than the full task,
so the numbers are higher than the headline result and are not directly comparable.

## Training

```bash
python prepare_imagenet.py --splits train,validation    # -> ./imagenet/{train,val}
python main.py                                           # single GPU
torchrun --nproc_per_node=<N> main.py                    # multi-GPU (DDP)
```

Training runs for 300 epochs (batch size 256, AdamW, cosine schedule, label
smoothing 0.1, mixed precision). Checkpoints and `run_config.json` are written to
`./runs`, and training resumes from `runs/last.pth` when present. The data source
is selected near the top of `main()` in `main.py`: local ImageFolder (default),
`stream_imagenet=True` to stream from the Hub without a local copy, or
`use_fake_data=True` for a dependency-free smoke test.

