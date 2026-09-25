# SPC: Stabilizing Multi-Step Diffusion Distillation with Student Perturbation Consistency 

Training code for SPC: Stabilizing Multi-Step Diffusion Distillation with Student Perturbation Consistency. The method uses Distribution Matching Distillation (DMD) with LoRA: the student is a K-step generator, while the fake score and teacher (Lightning LoRA) provide the matching signal. Training is based on [Accelerate](https://huggingface.co/docs/accelerate), and the data is stored locally in WebDataset tar files.

During training, latent variables `x0` are sampled on the `t=0` side. Flow matching's `add_noise(..., t1=0, t2=τ)` is used to construct states at each timestep. The generator starts from the maximum-noise timestep, consistent with the configured `K_step`.

## Environment

```bash
cd /path/to/SPC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended. Full-resolution 1024 training of Qwen-Image + LoRA typically requires substantial GPU memory. The demo configuration in this repository uses 512 resolution and fewer steps for quick testing.

## Data

By default, the repository uses a small WebDataset sample dataset (4 prompts + 512-resolution images):

```bash
python3 scripts/make_sample_dataset.py   # Skip if data/sample/sample-000000.tar already exists
```

For custom data, organize the dataset as WebDataset tar files. Each sample should contain `{key}.json` (with a `prompt` field) and an optional `{key}.image.png`. Set `dataset_config.data_path` in `configs/train_dmd_qwen_t2i.yaml` to point to the tar directory or a glob pattern.

## Model Configuration

Set the base model weights in the YAML configuration:

```yaml
pretrained_model_name_or_path: Qwen/Qwen-Image   # Or a local directory
```

- **student / fake**: By default, the LoRA adapters are initialized from scratch (the demo uses `r=64` / `r=256`).
- **teacher**: By default, the 4-step [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) LoRA is downloaded from Hugging Face and used only for teacher inference; the backbone is not trained.

## Training

```bash
chmod +x scripts/train.sh
./scripts/train.sh configs/train_dmd_qwen_t2i.yaml
```

Or:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 -m spc.train_dmd_qwen_t2i configs/train_dmd_qwen_t2i.yaml
```

Common hyperparameters in the demo configuration:

| Parameter | Default |
|----|--------|
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `learning_rate` / `learning_rate_fake_score` | 2e-5 / 1e-4 |
| `K_step` | 1 |
| `LIPS_lambda` | 1 |
| `max_train_steps` | 20 (for testing; increase for formal experiments) |

By default, logs are written to TensorBoard (`report_to: tensorboard`), and checkpoints are saved under `outputs/spc_dmd_demo`. The save format is `checkpoint-{step}/`. When `train_lora_only: true`, the checkpoint contains `adapter_student` and `adapter_fake`.

To use Weights & Biases, set `report_to: wandb` and `wandb.enable: true` in the configuration, and set the `WANDB_API_KEY` environment variable.

## Directory Structure

```text
SPC/
  configs/train_dmd_qwen_t2i.yaml
  data/sample/
  scripts/train.sh
  scripts/make_sample_dataset.py
  spc/
    train_dmd_qwen_t2i.py
    data_t2i.py
    model_utils.py
    validation.py
    checkpoint.py
```
