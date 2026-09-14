# SPC — Qwen-Image 文生图 DMD 蒸馏

基于 [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) 的少步文生图（T2I）蒸馏训练代码。使用 Distribution Matching Distillation（DMD）配合 LoRA：student 为 K 步生成器，fake score 与 teacher（Lightning LoRA）提供匹配信号。训练基于 [Accelerate](https://huggingface.co/docs/accelerate)，数据为本地 WebDataset tar。

训练时在 `t=0` 侧采样潜变量 `x0`，通过 flow matching 的 `add_noise(..., t1=0, t2=τ)` 构造各时间步状态；生成器在最大噪声时刻（与配置中的 `K_step` 一致）起算。

## 环境

```bash
cd /path/to/SPC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

建议使用 CUDA GPU。全分辨率 1024 训练 Qwen-Image + LoRA 通常需要较大显存；仓库内 demo 配置使用 512 分辨率与较少步数，便于快速试跑。

## 数据

默认使用仓库内小样例 WebDataset（4 条 prompt + 512 图像）：

```bash
python3 scripts/make_sample_dataset.py   # 若已有 data/sample/sample-000000.tar 可跳过
```

自有数据请整理为 WebDataset tar：每条样本包含 `{key}.json`（字段 `prompt`）与可选 `{key}.image.png`，并在 `configs/train_dmd_qwen_t2i.yaml` 的 `dataset_config.data_path` 中指向 tar 目录或 glob 路径。

## 模型配置

在 yaml 中设置基座权重：

```yaml
pretrained_model_name_or_path: Qwen/Qwen-Image   # 或本地目录
```

- **student / fake**：默认从 scratch 初始化 LoRA（demo 为 `r=64` / `r=256`）。
- **teacher**：默认从 Hugging Face 拉取 [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) 4-step LoRA，仅作教师推理，不训练 backbone。

## 训练

```bash
chmod +x scripts/train.sh
./scripts/train.sh configs/train_dmd_qwen_t2i.yaml
```

或：

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 -m spc.train_dmd_qwen_t2i configs/train_dmd_qwen_t2i.yaml
```

Demo 配置中的常用超参：

| 项 | 默认值 |
|----|--------|
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `learning_rate` / `learning_rate_fake_score` | 2e-5 / 1e-4 |
| `K_step` | 1 |
| `LIPS_lambda` | 1 |
| `max_train_steps` | 20（试跑；正式实验请增大） |

日志默认写入 TensorBoard（`report_to: tensorboard`），权重目录为 `outputs/spc_dmd_demo`。保存格式为 `checkpoint-{step}/`，在 `train_lora_only: true` 时会写入 `adapter_student` 与 `adapter_fake`。

使用 Weights & Biases 时：在配置里设置 `report_to: wandb`、`wandb.enable: true`，并设置环境变量 `WANDB_API_KEY`。

## 目录结构

```
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
