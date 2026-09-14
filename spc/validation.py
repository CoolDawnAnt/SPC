import gc
import json
import math
from collections.abc import Mapping as _ABCMapping
from collections.abc import Sequence as _ABCSequence
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

logger = get_logger(__name__)


def _deserialize_if_needed(value: Any) -> Any:
    if value is None or isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            text = path.read_text()
            return json.loads(text)
        if value.strip().startswith(("{", "[")):
            return json.loads(value)
    return value


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, _ABCSequence)):
        return list(value)
    return [value]


def _normalize_validation_entries(raw_entries: Any) -> List[Dict[str, Any]]:
    entries = _deserialize_if_needed(raw_entries)
    entries = _ensure_list(entries)
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        entry = _deserialize_if_needed(entry)
        if isinstance(entry, str):
            normalized.append({"prompt": entry})
        elif isinstance(entry, dict) or isinstance(entry, _ABCMapping):
            normalized.append(dict(entry))
    return normalized


def _pick_seed(entry: Dict[str, Any], args_seed: Any, default_seed: Any) -> Any:
    if entry.get("seed") is not None:
        return entry["seed"]
    if args_seed is not None:
        return args_seed
    return default_seed


def run_qwenimage_t2i_validation(
    *,
    vae,
    transformer,
    text_encoder,
    accelerator,
    scheduler,
    tokenizer,
    args,
    valid_steps: int = 50,
    log_prefix: str = "t2i_validation",
) -> None:
    validation_entries = _normalize_validation_entries(getattr(args, "validation_prompts", None))
    if not validation_entries:
        logger.info("No validation prompts provided, skipping validation.")
        return

    args_seed = getattr(args, "validation_seed", 2026)
    default_seed = getattr(args, "seed", 2026)
    true_cfg_scale = 1
    default_resolution = getattr(args, "validation_resolution", 1024)

    model_was_training = getattr(transformer, "training", False)
    transformer.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    if hasattr(vae, "eval"):
        vae.eval()

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipeline = QwenImagePipeline(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        tokenizer=tokenizer,
    )
    pipeline = pipeline.to(accelerator.device)

    k_step = args.distillation_config["K_step"]
    if hasattr(args.distillation_config, "get"):
        k_step = args.distillation_config.get("K_step", k_step)

    image_logs: List[Dict[str, Any]] = []
    for idx, entry in enumerate(validation_entries):
        prompt = entry.get("prompt")
        if not prompt:
            continue

        width = entry.get("width", default_resolution)
        height = entry.get("height", default_resolution)
        width = int(width // 32 * 32)
        height = int(height // 32 * 32)

        generator = None
        chosen_seed = _pick_seed(entry, args_seed, default_seed)
        if chosen_seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(int(chosen_seed))

        with torch.no_grad():
            pipeline.set_progress_bar_config(disable=True)
            outputs = pipeline(
                prompt=prompt,
                true_cfg_scale=true_cfg_scale,
                width=width,
                height=height,
                num_inference_steps=k_step,
                generator=generator,
            )

        images = outputs.images if hasattr(outputs, "images") else outputs[0]
        image_logs.append({"images": images, "prompt": prompt})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and image_logs:
        for tracker in getattr(accelerator, "trackers", []):
            if tracker.name != "wandb":
                continue
            payload: Dict[str, List[wandb.Image]] = {}
            for sample_idx, log in enumerate(image_logs):
                formatted = [wandb.Image(image, caption=log["prompt"]) for image in log["images"]]
                if formatted:
                    payload[f"{log_prefix}/sample_{sample_idx}"] = formatted
            if payload:
                tracker.log(payload)

    del pipeline
    if model_was_training:
        transformer.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
