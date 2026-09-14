"""Model init and text encoding for Qwen-Image T2I DMD (local / Hugging Face only)."""
import gc
import os
from typing import Optional

import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
)
from diffusers import QwenImagePipeline
from peft import LoraConfig
from torch.hub import download_url_to_file
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

logger = get_logger(__name__)


def _download_lora_weights(
    lora_path: Optional[str],
    download_url: Optional[str],
    cache_dir: str = "/tmp",
) -> Optional[str]:
    if lora_path and os.path.exists(lora_path):
        logger.info("[LORA] Using existing LoRA path: %s", lora_path)
        return lora_path

    if download_url:
        filename = download_url.split("/")[-1]
        local_path = os.path.join(cache_dir, filename)
        if os.path.exists(local_path):
            logger.info("[LORA] Reusing downloaded LoRA at %s", local_path)
            return local_path
        logger.info("[LORA] Downloading LoRA from %s to %s", download_url, local_path)
        os.makedirs(cache_dir, exist_ok=True)
        download_url_to_file(download_url, local_path)
        return local_path

    logger.warning("[LORA] No LoRA path or download URL provided; skipping LoRA load.")
    return None


def compute_text_embeddings_t2i(prompt, text_encoding_pipeline, max_sequence_length):
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2],
                dtype=torch.long,
                device=prompt_embeds.device,
            )
        if prompt_embeds.shape[1] > 768:
            prompt_embeds = prompt_embeds[:, :768]
            prompt_embeds_mask = prompt_embeds_mask[:, :768]
    return prompt_embeds, prompt_embeds_mask


def _apply_lora_adapter_with_name(
    transformer: QwenImageTransformer2DModel,
    lora_path: str,
    adapter_name: str,
) -> None:
    if not lora_path or not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    logger.info("[LORA] Loading LoRA weights from %s as adapter '%s'", lora_path, adapter_name)

    state_dict = torch.load(lora_path, map_location="cpu") if lora_path.endswith(".bin") else None
    if state_dict is None:
        from safetensors.torch import load_file

        state_dict = load_file(lora_path, device="cpu")

    has_lora_down = any(".lora_down.weight" in k for k in state_dict.keys())
    has_lora_A = any(".lora_A.weight" in k for k in state_dict.keys())

    lora_rank = None
    lora_alpha = None
    for key, value in state_dict.items():
        if ".lora_down.weight" in key or ".lora_A.weight" in key:
            lora_rank = value.shape[0]
            break
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            lora_alpha = value.item() if hasattr(value, "item") else float(value)
            break
    if lora_rank is None:
        raise ValueError(f"[LORA] Could not determine lora_rank from {lora_path}")
    if lora_alpha is None:
        lora_alpha = lora_rank

    converted_state = {}
    if has_lora_A and not has_lora_down:
        for key, value in state_dict.items():
            if ".lora_A.weight" in key:
                new_key = key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight")
                converted_state[new_key] = value
            elif ".lora_B.weight" in key:
                new_key = key.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
                converted_state[new_key] = value
    else:
        for key, value in state_dict.items():
            if key.endswith(".alpha"):
                continue
            if ".lora_down.weight" in key:
                new_key = key.replace(".lora_down.weight", f".lora_A.{adapter_name}.weight")
                converted_state[new_key] = value
            elif ".lora_up.weight" in key:
                new_key = key.replace(".lora_up.weight", f".lora_B.{adapter_name}.weight")
                converted_state[new_key] = value

    del state_dict
    gc.collect()

    if hasattr(transformer, "peft_config") and transformer.peft_config:
        if adapter_name in transformer.peft_config:
            transformer.delete_adapter(adapter_name)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=int(lora_alpha),
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj",
            "to_add_out", "net.0.proj", "net.2",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    transformer.add_adapter(lora_config, adapter_name=adapter_name)

    param_dict = dict(transformer.named_parameters())
    loaded_count = 0
    for key, value in converted_state.items():
        if key in param_dict and param_dict[key].shape == value.shape:
            param_dict[key].data.copy_(value)
            loaded_count += 1

    del converted_state, param_dict
    gc.collect()
    logger.info("[LORA] Loaded %s weights into adapter '%s'", loaded_count, adapter_name)


def _init_lora_adapter(
    transformer,
    init_method: str,
    adapter_name: str,
    lora_cfg=None,
    default_lora_params=None,
    lightning_lora_path=None,
):
    logger.info("[LORA] Initializing '%s' adapter with method: %s", adapter_name, init_method)

    if init_method == "scratch":
        r = getattr(lora_cfg, "lora_r", default_lora_params["lora_r"]) if lora_cfg else default_lora_params["lora_r"]
        alpha = getattr(lora_cfg, "lora_alpha", default_lora_params["lora_alpha"]) if lora_cfg else default_lora_params["lora_alpha"]
        dropout = getattr(lora_cfg, "lora_dropout", default_lora_params["lora_dropout"]) if lora_cfg else default_lora_params["lora_dropout"]
        target_modules = getattr(lora_cfg, "lora_target_modules", default_lora_params["lora_target_modules"]) if lora_cfg else default_lora_params["lora_target_modules"]
        if hasattr(target_modules, "__iter__") and not isinstance(target_modules, (str, list)):
            target_modules = list(target_modules)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
    elif init_method == "lightning":
        if not lightning_lora_path:
            raise ValueError(f"[LORA] {adapter_name} init_method='lightning' but no lightning LoRA path")
        _apply_lora_adapter_with_name(transformer, lightning_lora_path, adapter_name)
    elif os.path.isfile(init_method):
        _apply_lora_adapter_with_name(transformer, init_method, adapter_name)
    else:
        raise ValueError(f"[LORA] Unsupported init_method for release build: {init_method}")


def initialize_QwenImage_t2i_single_backbone_multi_adapter(config):
    """
    Load Qwen-Image T2I from `pretrained_model_name_or_path` (local dir or HF id).
    """
    model_path = config.get("pretrained_model_name_or_path")
    if not model_path:
        raise ValueError("Set `pretrained_model_name_or_path` to a local checkpoint or Hugging Face model id.")

    logger.info("[INFO] Loading Qwen-Image T2I from %s", model_path)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.requires_grad_(False)

    vae = AutoencoderKLQwenImage.from_pretrained(model_path, subfolder="vae")
    vae.requires_grad_(False)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    train_cfg = getattr(config, "train_config", None)
    student_lora_init_method = getattr(train_cfg, "student_lora_init_method", "scratch") if train_cfg else "scratch"
    fake_lora_init_method = getattr(train_cfg, "fake_lora_init_method", "scratch") if train_cfg else "scratch"
    teacher_lora_init_method = getattr(train_cfg, "teacher_lora_init_method", "lightning") if train_cfg else "lightning"
    student_lora_half_rank_reinit = getattr(train_cfg, "student_lora_half_rank_reinit", False) if train_cfg else False

    student_lora_cfg = getattr(train_cfg, "student_lora", None) if train_cfg else None
    fake_lora_cfg = getattr(train_cfg, "fake_lora", None) if train_cfg else None

    default_lora_params = {
        "lora_r": 64,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_target_modules": [
            "to_v", "to_q", "to_out.0", "to_add_out", "net.2", "proj", "to_k",
            "add_q_proj", "add_v_proj", "add_k_proj",
        ],
    }

    distill_cfg = config.get("distillation_config", {})
    if hasattr(distill_cfg, "get"):
        k_step = distill_cfg.get("K_step", 4)
    else:
        k_step = getattr(distill_cfg, "K_step", 4)

    if k_step == 4:
        lightning_url = "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
    elif k_step == 8:
        lightning_url = "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
    else:
        lightning_url = "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"

    default_lora_url = getattr(config, "lora_download_url", lightning_url)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "spc_lora")
    local_lora_path = os.path.join(cache_dir, default_lora_url.split("/")[-1])

    need_lightning = student_lora_init_method == "lightning" or fake_lora_init_method == "lightning"
    lightning_lora_path = _download_lora_weights(
        local_lora_path if os.path.exists(local_lora_path) else None,
        default_lora_url,
        cache_dir=cache_dir,
    ) if need_lightning else None

    if teacher_lora_init_method == "lightning":
        teacher_url = "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
        teacher_lightning_path = _download_lora_weights(None, teacher_url, cache_dir=cache_dir)
    else:
        teacher_lightning_path = None

    _init_lora_adapter(transformer, student_lora_init_method, "student", student_lora_cfg, default_lora_params, lightning_lora_path)

    if student_lora_half_rank_reinit and student_lora_init_method != "scratch":
        reinit_count = 0
        for name, param in transformer.named_parameters():
            if ".lora_A.student." in name and name.endswith(".weight"):
                r = param.shape[0]
                with torch.no_grad():
                    param.data[r // 2 :, :].mul_(0.5)
                reinit_count += 1
        logger.info("[LORA] Half-rank re-init on %s student matrices", reinit_count)

    _init_lora_adapter(transformer, fake_lora_init_method, "fake", fake_lora_cfg, default_lora_params, lightning_lora_path)
    _init_lora_adapter(transformer, teacher_lora_init_method, "teacher", None, default_lora_params, teacher_lightning_path)

    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        model_path,
        vae=vae,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    return (
        vae,
        transformer,
        tokenizer,
        text_encoder,
        noise_scheduler,
        text_encoding_pipeline,
        vae_scale_factor,
    )
