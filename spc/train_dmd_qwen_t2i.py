# ruff: noqa
"""
Qwen-Image Text-to-Image DMD distillation (open-source SPC release).

Differences from internal TDM build:
- No Ray / AWS / S3; local Accelerate training only.
- DMD noise path: sample x0 at t=0 and add noise (no student trajectory / t_anchor replay).
"""
import argparse
import copy
import logging
import math
import os
import random
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils.torch_utils import is_compiled_module
from mmengine import Config
from tqdm.auto import tqdm

from spc.checkpoint import save_checkpoint
from spc.data_t2i import collate_batch_fn_with_unified_size, create_train_dataloader
from spc.model_utils import (
    compute_text_embeddings_t2i,
    initialize_QwenImage_t2i_single_backbone_multi_adapter,
)
from spc.validation import run_qwenimage_t2i_validation

logger = get_logger(__name__)


def flow_timesteps_for_k(k_step: int, device, dtype=torch.float32):
    """Flow-match timesteps with shift=3 (matches distillation training)."""
    shift = 3.0
    t_uniform = torch.linspace(1.0, 0.0, k_step + 1, device=device, dtype=dtype)
    return shift * t_uniform / (1 + (shift - 1) * t_uniform)




def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class Predictor():
    """
    Predictor class for T2I model (no image_latents conditioning).
    """
    def __init__(self, uncond_prompt_embeds, uncond_prompt_embeds_mask, weight_dtype):
        super().__init__()
        self.uncond_prompt_embeds = uncond_prompt_embeds
        self.uncond_prompt_embeds_mask = uncond_prompt_embeds_mask
        self.weight_dtype = weight_dtype

    def _resolve_uncond_inputs(
        self,
        dynamic_prompt_embeds,
        dynamic_prompt_embeds_mask,
    ):
        prompt_embeds = (
            dynamic_prompt_embeds
            if dynamic_prompt_embeds is not None
            else self.uncond_prompt_embeds
        )
        prompt_embeds_mask = (
            dynamic_prompt_embeds_mask
            if dynamic_prompt_embeds_mask is not None
            else self.uncond_prompt_embeds_mask
        )
        if prompt_embeds is None or prompt_embeds_mask is None:
            raise ValueError("Predictor requires unconditional embeddings for classifier-free guidance.")
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        return prompt_embeds, prompt_embeds_mask, txt_seq_lens

    def predict(
        self,
        score_model,
        x_t,
        timestep,
        img_shapes,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        txt_seq_lens,
        cfg=None,
        uncond_prompt_embeds=None,
        uncond_prompt_embeds_mask=None,
        return_DeDMD=False,
        return_double=False,
        return_all=False,
        return_velocity=False,
    ):
        """
        T2I prediction - no image_latents input.
        """
        bsz = x_t.shape[0]
        alpha_t, sigma_t = (1 - timestep).reshape(bsz, 1, 1), timestep.reshape(bsz, 1, 1)

        # T2I: No image_latents concatenation
        latent_model_input = x_t

        score_pred = score_model(
            hidden_states=latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
        )[0]

        if cfg is not None:
            uncond_prompt_embeds, uncond_prompt_embeds_mask, uncond_txt_seq_lens = self._resolve_uncond_inputs(
                uncond_prompt_embeds,
                uncond_prompt_embeds_mask,
            )
            if uncond_prompt_embeds.device != latent_model_input.device:
                uncond_prompt_embeds = uncond_prompt_embeds.to(device=latent_model_input.device)
            if uncond_prompt_embeds.dtype != latent_model_input.dtype:
                uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=latent_model_input.dtype)
            if uncond_prompt_embeds_mask.device != latent_model_input.device:
                uncond_prompt_embeds_mask = uncond_prompt_embeds_mask.to(device=latent_model_input.device)

            score_uncond_pred = score_model(
                hidden_states=latent_model_input,
                encoder_hidden_states=uncond_prompt_embeds,
                encoder_hidden_states_mask=uncond_prompt_embeds_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=uncond_txt_seq_lens,
            )[0]

            comb_pred = score_uncond_pred + cfg * (score_pred - score_uncond_pred)
            noise_norm = torch.norm(score_pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            comb_pred_callibrated = comb_pred * (noise_norm / comb_norm)
            score_pred_cfg = comb_pred_callibrated

            x0_pred_cond = x_t - sigma_t * score_pred
            x0_pred_uncond = x_t - sigma_t * score_uncond_pred
            x0_pred_cfg = x0_pred_uncond + cfg * (x0_pred_cond - x0_pred_uncond)
            velocity_cfg = score_pred_cfg
            x1_pred = x_t + alpha_t * velocity_cfg

            if return_velocity:
                return score_pred, score_uncond_pred
            if return_DeDMD:
                return x0_pred_cond, x0_pred_uncond

            if return_all:
                return x1_pred, x0_pred_cfg, velocity_cfg

            if return_double:
                return x1_pred, x0_pred_cfg

        else:
            x0_pred = x_t - sigma_t * score_pred
            velocity = score_pred
            x1_pred = x_t + alpha_t * velocity
            if return_velocity:
                return score_pred

            if return_all:
                return x1_pred, x0_pred, velocity
            if return_double:
                return x1_pred, x0_pred
        if cfg is not None:
            return x0_pred_cfg
        else:
            return x0_pred

    def predict_multistep(
        self,
        score_model,
        x_t,
        timestep_list,  # sth like [0.5, 0.2, 0], 从大到小
        img_shapes,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        txt_seq_lens,
        cfg=None,
        uncond_prompt_embeds=None,
        uncond_prompt_embeds_mask=None,
        return_DeDMD=False,
    ):
        """
        T2I multistep prediction - no image_latents input.
        """
        if return_DeDMD:
            # 从 x_t 出发，分别用 cond 和 uncond 走各自独立的轨迹
            x_cur_cond = x_t
            x_cur_uncond = x_t

            # 预处理 uncond embeddings
            uncond_embeds, uncond_embeds_mask, uncond_txt_seq_lens = self._resolve_uncond_inputs(
                uncond_prompt_embeds,
                uncond_prompt_embeds_mask,
            )

            for i in range(len(timestep_list) - 1):
                t_cur = timestep_list[i]
                t_next = timestep_list[i + 1]
                bsz = x_cur_cond.shape[0]
                timestep_tensor = t_cur if isinstance(t_cur, torch.Tensor) else torch.tensor([t_cur] * bsz, device=x_cur_cond.device, dtype=x_cur_cond.dtype)

                # T2I: No image_latents concatenation
                latent_cond = x_cur_cond
                latent_uncond = x_cur_uncond

                # 只调用一次 DIT 获取 cond velocity
                v_cond = score_model(
                    hidden_states=latent_cond,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep_tensor,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                )[0]

                # 只调用一次 DIT 获取 uncond velocity
                uncond_embeds_device = uncond_embeds.to(device=latent_uncond.device, dtype=latent_uncond.dtype)
                uncond_embeds_mask_device = uncond_embeds_mask.to(device=latent_uncond.device)
                v_uncond = score_model(
                    hidden_states=latent_uncond,
                    encoder_hidden_states=uncond_embeds_device,
                    encoder_hidden_states_mask=uncond_embeds_mask_device,
                    timestep=timestep_tensor,
                    img_shapes=img_shapes,
                    txt_seq_lens=uncond_txt_seq_lens,
                )[0]

                x_cur_cond = x_cur_cond + (t_next - t_cur) * v_cond  # Euler step, t_next < t_cur
                x_cur_uncond = x_cur_uncond + (t_next - t_cur) * v_uncond  # Euler step, t_next < t_cur
            return x_cur_cond, x_cur_uncond
        else:
            x_cur = x_t
            for i in range(len(timestep_list) - 1):
                t_cur = timestep_list[i]
                t_next = timestep_list[i + 1]
                _, __, v_pred = self.predict(
                    score_model, x_cur, t_cur, img_shapes,
                    encoder_hidden_states, encoder_hidden_states_mask, txt_seq_lens,
                    cfg=cfg,
                    uncond_prompt_embeds=uncond_prompt_embeds,
                    uncond_prompt_embeds_mask=uncond_prompt_embeds_mask,
                    return_all=True
                )
                x_cur = x_cur + (t_next - t_cur) * v_pred  # Euler step, t_next < t_cur
            return x_cur

    def add_noise(self, samples, noise, t1, t2):
        bsz = samples.shape[0]

        alpha_t1, sigma_t1 = (1 - t1).reshape(bsz, 1, 1), t1.reshape(bsz, 1, 1)
        alpha_t2, sigma_t2 = (1 - t2).reshape(bsz, 1, 1), t2.reshape(bsz, 1, 1)

        samples = samples / (alpha_t1 + 1e-6) * alpha_t2

        beta = sigma_t2 ** 2 - (alpha_t2 / (alpha_t1 + 1e-6) * sigma_t1) ** 2
        beta = torch.clamp(beta, min=1e-8) ** 0.5

        samples = samples + beta * noise

        return samples.to(self.weight_dtype)


def calculate_recommended_resolution(
    height: int,
    width: int,
    target_area: int = 1024 * 1024,
    multiple: int = 32,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive to calculate the recommended resolution.")
    ratio = width / height
    target_width = math.sqrt(target_area * ratio)
    target_height = target_width / max(ratio, 1e-8)
    target_width = max(multiple, int(round(target_width / multiple) * multiple))
    target_height = max(multiple, int(round(target_height / multiple) * multiple))
    return target_width, target_height


def make_serializable_for_tracking(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(k): make_serializable_for_tracking(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [make_serializable_for_tracking(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return make_serializable_for_tracking(to_dict())
        except Exception:
            pass
    return str(value)


def train_loop_full(config):

    print('config:', config)

    if isinstance(config, str):
        config = Config.fromfile(config)

    def _get_section(section_name):
        return getattr(config, section_name, None)

    def _get_value(section, key, default=None):
        if section is None:
            return default
        getter = getattr(section, "get", None)
        if callable(getter):
            return getter(key, default)
        if isinstance(section, dict):
            return section.get(key, default)
        return getattr(section, key, default)

    train_cfg = _get_section("train_config")
    distill_cfg = _get_section("distillation_config")
    eval_cfg = _get_section("eval_config")
    wandb_cfg = _get_section("wandb")
    dataloader_cfg = _get_section("dataloader")
    local_debug_cfg = _get_section("local_debug")
    model_cfg = _get_section("model_config")

    config.model_output_dir = getattr(config, "model_output_dir", "./outputs")
    config.logging_dir = getattr(config, "logging_dir", "logs")
    config.report_to = getattr(
        config,
        "report_to",
        "wandb" if _get_value(wandb_cfg, "enable", False) else None,
    )
    config.tracker_name = getattr(
        config,
        "tracker_name",
        _get_value(wandb_cfg, "run_name", getattr(config, "job_name", "training-run")),
    )
    # wandb project name (init_trackers 的第一个参数是 project name)
    config.wandb_project = _get_value(wandb_cfg, "project", config.tracker_name)
    config.mixed_precision = getattr(config, "mixed_precision", "bf16")
    config.allow_tf32 = getattr(config, "allow_tf32", True)
    config.scale_lr = getattr(config, "scale_lr", False)
    config.offload = getattr(config, "offload", False)
    config.bnb_quantization_config_path = getattr(
        config, "bnb_quantization_config_path", None
    )
    config.resume_from_checkpoint = getattr(config, "resume_from_checkpoint", "")
    config.max_sequence_length = getattr(
        config, "max_sequence_length", _get_value(model_cfg, "max_sequence_length", 768)
    )
    config.negative_prompt = getattr(config, "negative_prompt", "")

    config.train_batch_size = _get_value(
        train_cfg, "batch_size", getattr(config, "train_batch_size", 1)
    )
    config.gradient_accumulation_steps = _get_value(
        train_cfg,
        "gradient_accumulation_steps",
        getattr(config, "gradient_accumulation_steps", 8),
    )
    config.learning_rate = float(_get_value(train_cfg, "learning_rate", getattr(config, "learning_rate", 5e-6)))
    config.learning_rate_fake_score = float(_get_value(train_cfg, "learning_rate_fake_score", getattr(config, "learning_rate_fake_score", 5e-6)))
    config.adam_beta1 = float(_get_value(train_cfg, "adam_beta1", getattr(config, "adam_beta1", 0.0)))
    config.adam_beta2 = float(_get_value(train_cfg, "adam_beta2", getattr(config, "adam_beta2", 0.95)))
    default_weight_decay = getattr(
        config, "adam_weight_decay", _get_value(train_cfg, "weight_decay", 0.0)
    )
    config.adam_weight_decay = _get_value(
        train_cfg, "adam_weight_decay", default_weight_decay
    )
    config.adam_epsilon = float(_get_value(
        train_cfg, "adam_epsilon", getattr(config, "adam_epsilon", 1e-8)
    ))
    config.lr_scheduler = _get_value(
        train_cfg, "lr_scheduler", getattr(config, "lr_scheduler", "cosine")
    )
    config.lr_warmup_steps = int(
        _get_value(train_cfg, "warmup_steps", getattr(config, "lr_warmup_steps", 0))
    )
    config.lr_num_cycles = float(
        _get_value(train_cfg, "lr_num_cycles", getattr(config, "lr_num_cycles", 1))
    )
    config.lr_power = float(
        _get_value(train_cfg, "lr_power", getattr(config, "lr_power", 1.0))
    )
    config.max_grad_norm = float(
        _get_value(train_cfg, "max_grad_norm", getattr(config, "max_grad_norm", 1.0))
    )
    config.max_train_steps = _get_value(
        train_cfg, "max_train_steps", getattr(config, "max_train_steps", 1000)
    )
    config.checkpointing_steps = _get_value(
        train_cfg,
        "checkpointing_steps",
        _get_value(train_cfg, "save_steps", getattr(config, "checkpointing_steps", 200)),
    )
    checkpoints_total_limit = _get_value(
        train_cfg, "checkpoints_total_limit", getattr(config, "checkpoints_total_limit", None)
    )
    config.checkpoints_total_limit = (
        int(checkpoints_total_limit) if checkpoints_total_limit is not None else None
    )
    config.prefetch_batches = int(
        _get_value(train_cfg, "prefetch_batches", getattr(config, "prefetch_batches", 3))
    )

    config.dataloader_num_workers = int(
        _get_value(dataloader_cfg, "num_workers", getattr(config, "dataloader_num_workers", 4))
    )
    config.dataloader_shuffle = _get_value(
        dataloader_cfg, "shuffle", getattr(config, "dataloader_shuffle", True)
    )
    config.dataloader_pin_memory = _get_value(
        dataloader_cfg, "pin_memory", getattr(config, "dataloader_pin_memory", True)
    )
    config.dataloader_drop_last = _get_value(
        dataloader_cfg, "drop_last", getattr(config, "dataloader_drop_last", True)
    )
    config.local_shuffle_seed = int(
        _get_value(dataloader_cfg, "local_shuffle_seed", getattr(config, "local_shuffle_seed", 42))
    )
    config.local_shuffle_buffer_multiplier = int(
        _get_value(
            dataloader_cfg,
            "local_shuffle_buffer_multiplier",
            getattr(config, "local_shuffle_buffer_multiplier", 3),
        )
    )

    config.K_step = int(_get_value(distill_cfg, "K_step", getattr(config, "K_step", 8)))
    config.eta = float(_get_value(distill_cfg, "eta", getattr(config, "eta", 1.0)))
    config.reg_lambda = float(
        _get_value(distill_cfg, "reg_lambda", getattr(config, "reg_lambda", 0.0))
    )
    config.max_loss_fake = float(
        _get_value(distill_cfg, "max_loss_fake", getattr(config, "max_loss_fake", 1000.0))
    )
    config.cfg = float(_get_value(distill_cfg, "cfg", getattr(config, "cfg", 4)))

    config.validation_steps = _get_value(
        eval_cfg, "every_steps", getattr(config, "validation_steps", 0)
    )
    config.validation_prompts = _get_value(
        eval_cfg, "validation_prompts", getattr(config, "validation_prompts", None)
    )
    config.validation_func_name = _get_value(
        eval_cfg, "validation_func_name", getattr(config, "validation_func_name", "")
    )
    config.num_validation_steps = int(
        _get_value(eval_cfg, "num_inference_steps", getattr(config, "num_validation_steps", 8))
    )
    config.validation_guidance_scale = float(
        _get_value(eval_cfg, "validation_guidance_scale", getattr(config, "validation_guidance_scale", 1.0))
    )
    config.validation_true_cfg_scale = float(
        _get_value(
            eval_cfg,
            "true_cfg_scale",
            getattr(config, "validation_true_cfg_scale", getattr(config, "cfg", 4.0)),
        )
    )
    config.validation_seed = _get_value(
        eval_cfg,
        "seed",
        getattr(config, "validation_seed", getattr(config, "seed", None)),
    )
    config.validation_reference_images = _get_value(
        eval_cfg,
        "validation_reference_images",
        getattr(config, "validation_reference_images", None),
    )
    config.validation_resolution = int(
        _get_value(eval_cfg, "resolution", getattr(config, "validation_resolution", 1024))
    )
    config.validation_target_area = int(
        _get_value(
            eval_cfg,
            "target_area",
            getattr(config, "validation_target_area", 1024 * 1024),
        )
    )
    config.dataset_config = _get_section("dataset_config") or {}
    config.local_debug = local_debug_cfg or {}
    config.dataset = getattr(
        config, "dataset", _get_value(config.local_debug, "dataset", None)
    )

    config.train_batch_size = int(config.train_batch_size)
    config.gradient_accumulation_steps = int(config.gradient_accumulation_steps)
    config.max_train_steps = int(config.max_train_steps)
    config.checkpointing_steps = max(1, int(config.checkpointing_steps))
    config.prefetch_batches = int(config.prefetch_batches)
    config.validation_steps = (
        int(config.validation_steps) if config.validation_steps else 0
    )

    if config.report_to == "wandb" and config.hub_token is not None: # you need to specify the hub_token in the config
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(config.model_output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.model_output_dir,
        logging_dir=logging_dir
    )
    # For full parameter fine-tuning, we need to be more careful about unused parameters
    # Set to True initially to detect unused parameters, then we can optimize
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # gradient_accumulation_steps=config.gradient_accumulation_steps, 弃用自带的accumulation_steps，使用自定义的accumulation_steps
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        # warning
        logger.warning(f"Webdataset wds.ResampledShards is not compatible with the seed.")
        set_seed(config.seed)

    # Handle the repository creation
    if True:  # set to true forever. Otherwise it will trigger error due to multi-node training
    # if accelerator.is_main_process:
        if config.model_output_dir is not None:
            os.makedirs(config.model_output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Use T2I initialization (no image conditioning)
    (
        vae,
        transformer,  # single backbone with 'student' and 'fake' adapters
        tokenizer,
        text_encoder,
        noise_scheduler,
        text_encoding_pipeline,
        vae_scale_factor,
    ) = initialize_QwenImage_t2i_single_backbone_multi_adapter(config)

    # Check if new_lora mode is enabled (fixed_student + fresh student)
    new_lora_enabled = _get_value(config, "new_lora", False)
    if new_lora_enabled:
        logger.info("[TRAIN] new_lora mode enabled: using ['fixed_student', 'student'] for inference, training only 'student'")
        # In new_lora mode, we use both adapters for inference
        student_adapter_name = ["fixed_student", "student"]
    else:
        student_adapter_name = "student"

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    config.vae_scale_factor = vae_scale_factor

    # move models to the assigned device
    logger.info(f"[INFO] move models to the assigned device")
    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not config.offload else {"dtype": weight_dtype}
    vae.to(**to_kwargs)
    text_encoder.to(**to_kwargs)

    transformer_to_kwargs = (
        {"device": accelerator.device}
        if config.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )
    transformer.to(**transformer_to_kwargs)

    # Backbone is frozen, adapters will be selectively enabled during training
    transformer.train()

    # Load DreamSim for perceptual diversity loss (frozen, eval-only)
    # Rank 0 downloads first to avoid multi-process file race conditions,
    # then other ranks load from the cached files.
    dreamsim_model = None
    _div_lambda_check = float(getattr(config.distillation_config, "diversity_lambda", 0) if not isinstance(config.distillation_config, dict) else config.distillation_config.get("diversity_lambda", 0))
    if _div_lambda_check > 0:
        from dreamsim import dreamsim as _load_dreamsim
        # Use a fixed absolute cache dir so weights persist across runs/workers
        _dreamsim_cache = os.path.join(os.path.expanduser("~"), ".cache", "dreamsim")
        os.makedirs(_dreamsim_cache, exist_ok=True)
        # Also fix torch.hub cache for DINO/CLIP backbone downloads
        torch.hub.set_dir(_dreamsim_cache)
        if accelerator.is_main_process:
            dreamsim_model, _ = _load_dreamsim(pretrained=True, cache_dir=_dreamsim_cache)
            logger.info("[INFO] DreamSim downloaded by main process")
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            dreamsim_model, _ = _load_dreamsim(pretrained=True, cache_dir=_dreamsim_cache)
        dreamsim_model = dreamsim_model.to(device=accelerator.device, dtype=torch.float32)
        dreamsim_model.eval()
        dreamsim_model.requires_grad_(False)
        logger.info("[INFO] DreamSim loaded for perceptual diversity loss (fp32)")

    # Debug: Check which parameters are trainable
    trainable_param_names = []
    non_trainable_param_names = []
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            trainable_param_names.append(name)
        else:
            non_trainable_param_names.append(name)

    logger.info(f"Trainable parameters: {len(trainable_param_names)}")
    logger.info(f"Non-trainable parameters: {len(non_trainable_param_names)}")
    if non_trainable_param_names:
        logger.warning(f"Non-trainable parameters found: {non_trainable_param_names[:10]}...")  # Show first 10

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make sure the trainable params are in float32.
    if config.mixed_precision == "fp16":
        models = [transformer]
        # upcast trainable parameters into fp32
        cast_training_params(models, dtype=torch.float32)

    # T2I: Use simpler collate function for text-only data
    train_k_step_generator_every_n_steps = getattr(config.distillation_config, "train_k_step_generator_every_n_steps", 5)
    macro_batch_size = (
        config.train_batch_size
        * config.gradient_accumulation_steps
        * train_k_step_generator_every_n_steps
    )
    train_dataloader = create_train_dataloader(
        config.dataset_config,
        batch_size=macro_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle_seed=config.local_shuffle_seed,
    )

    del text_encoder, tokenizer
    free_memory()

    # No FSDP - single backbone with multiple adapters is memory efficient enough

    def _collect_lora_params_by_adapter(model, adapter_name):
        """Collect LoRA parameters for a specific adapter."""
        from peft.tuners.tuners_utils import BaseTunerLayer
        params = []
        for name, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                for param_name, param in module.named_parameters():
                    if adapter_name in param_name and param.requires_grad:
                        params.append(param)
        return params

    # Enable student adapter and collect its parameters
    transformer.set_adapter(student_adapter_name)
    transformer.enable_adapters()
    # Only collect 'student' adapter parameters for training (not fixed_student)
    transformer_student_parameters = _collect_lora_params_by_adapter(transformer, "student")

    transformer_parameters_with_lr = {"params": transformer_student_parameters, "lr": config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Debug: Print parameter count
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer_student_parameters)
    logger.info(f"Total parameters transformer: {total_params:,}")
    logger.info(f"Trainable parameters (student adapter): {trainable_params:,}")
    logger.info(f"Trainable parameter ratio: {trainable_params/total_params:.2%}")

    # Collect fake adapter parameters
    transformer.set_adapter("fake")
    transformer_fake_parameters = _collect_lora_params_by_adapter(transformer, "fake")
    transformer_fake_parameters_with_lr = {"params": transformer_fake_parameters, "lr": config.learning_rate_fake_score}
    params_to_optimize_fake = [transformer_fake_parameters_with_lr]

    trainable_params_fake = sum(p.numel() for p in transformer_fake_parameters)
    logger.info(f"Trainable parameters (fake adapter): {trainable_params_fake:,}")

    # Reset to student adapter as default
    transformer.set_adapter(student_adapter_name)


    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    optimizer_fake = torch.optim.AdamW(
        params_to_optimize_fake,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )


    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    lr_scheduler_fake = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer_fake,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    train_dataloader, optimizer, lr_scheduler, optimizer_fake, lr_scheduler_fake = \
        accelerator.prepare(train_dataloader, optimizer, lr_scheduler, optimizer_fake, lr_scheduler_fake)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = config.tracker_name
        tracker_config = make_serializable_for_tracking(
            {k: v for k, v in vars(config).items() if not k.startswith("_")}
        )
        accelerator.init_trackers(
            config.wandb_project,  # project name
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": getattr(config, "run_name", tracker_name),  # run name
                }
            } if config.report_to == "wandb" else None,
        )

    accelerator.wait_for_everyone()
    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    # logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    # logger.info(f"  Train iteration function: {config.train_iteration_func_name}")
    logger.info(f"  Validation function : {config.validation_func_name}")
    global_step = 0
    first_epoch = 0
    timestep_shift = getattr(config.train_config, "timestep_shift", 1)
    reg_lambda = getattr(config.distillation_config, "reg_lambda", 0)
    LIPS_lambda = getattr(config.distillation_config, "LIPS_lambda", 0)
    LIPS_sample = getattr(config.distillation_config, "LIPS_sample", "Gaussian")
    LIPS_eps = getattr(config.distillation_config, "LIPS_eps", 0.1)
    CM_lambda = getattr(config.distillation_config, "CM_lambda", 0)
    HLIPS_lambda = getattr(config.distillation_config, "HLIPS_lambda", 0)
    hlips_step = getattr(config.distillation_config, "hlips_step", 1)
    diversity_lambda = float(getattr(config.distillation_config, "diversity_lambda", 0) if not isinstance(config.distillation_config, dict) else config.distillation_config.get("diversity_lambda", 0))
    diversity_eps = float(getattr(config.distillation_config, "diversity_eps", 1e-6) if not isinstance(config.distillation_config, dict) else config.distillation_config.get("diversity_eps", 1e-6))
    diversity_ode_steps = int(getattr(config.distillation_config, "diversity_ode_steps", 1) if not isinstance(config.distillation_config, dict) else config.distillation_config.get("diversity_ode_steps", 1))
    rcgm_step = getattr(config.distillation_config, "rcgm_step", 1)
    ucgm_step = getattr(config.distillation_config, "ucgm_step", 1)
    train_k_step_generator_every_n_steps = getattr(config.distillation_config, "train_k_step_generator_every_n_steps", 5)
    stable_fake = getattr(config.distillation_config, "stable_fake", False)
    print('Using timestep_shift as ', timestep_shift)
    print('Using train_k_step_generator_every_n_steps as ', train_k_step_generator_every_n_steps)
    print('Using LIPS_sample as ', LIPS_sample)
    print('Using LIPS_eps as ', LIPS_eps)
    print('Using CM_lambda as ', CM_lambda)
    print('Using HLIPS_lambda as ', HLIPS_lambda)
    print('Using hlips_step as ', hlips_step)
    print('Using diversity_lambda as ', diversity_lambda)
    print('Using diversity_eps as ', diversity_eps)
    print('Using diversity_ode_steps as ', diversity_ode_steps)
    print('Using reg_lambda as ', reg_lambda)
    print("Using rcgm_step as ", rcgm_step)
    print("Using ucgm_step as ", ucgm_step)

    def periodic_log_cwd_and_wandb_dir():
        while True:
            # regularly log the current working directory, wandb online link, and config
            logger.info(f"[PERIODIC LOG] Current working directory: {os.getcwd()}")
            # if (wandb is not None) and (wandb.run is not None):
            #     logger.info(f"[PERIODIC LOG] wandb online link: {wandb.run.url}")
            time.sleep(300)  # 5 minutes
    if accelerator.is_main_process:
        t = threading.Thread(target=periodic_log_cwd_and_wandb_dir, daemon=True)
        t.start()

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:

        if True:  # local checkpoint resume
            if config.resume_from_checkpoint != "latest":
                path = os.path.basename(config.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(config.model_output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                config.resume_from_checkpoint = None
                initial_global_step = 0
                global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(config.model_output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                # first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
        global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar on the global main process (not per-node)
        disable=not accelerator.is_main_process,
    )

    # T2I: Compute negative prompt embeddings (no condition images)
    neg_prompt_embeds, neg_prompt_embeds_mask = compute_text_embeddings_t2i(
        " ", text_encoding_pipeline, 768
    )
    neg_txt_seq_lens = neg_prompt_embeds_mask.sum(dim=1).tolist()
    predictor = Predictor(neg_prompt_embeds, neg_prompt_embeds_mask, weight_dtype)

    # T2I validation prompts (no reference images needed)
    validation_prompts = [
        {"prompt": "On a smooth, beige desktop, four ballpoint pens with blue, black, silver, and red barrels are meticulously arranged at right angles to each other, creating a rectangular outline. In the center of this rectangle, five wooden pencils with freshly sharpened tips are placed with their erasers touching, forming a precise circle. The stark contrast between the rigid geometry of the pens and the soft curve of the pencils is evident upon the uniform background of the desk's surface."},
        {"prompt": "A majestic parrot with vibrant green, red, and blue feathers glides effortlessly across the bright blue sky. Its impressive wingspan is fully outstretched, catching the warm sunlight as it soars high above the tree line. Below, the landscape features rolling hills and patches of dense forest." },
        {"prompt": "Inside a warm room with a large window showcasing a picturesque winter landscape, three gleaming ruby red necklaces are elegantly laid out on the plush surface of a deep purple velvet jewelry box. The gentle glow from the overhead light accentuates the rich color and intricate design of the necklaces. Just beyond the glass pane, snowflakes can be seen gently falling to coat the ground outside in a blanket of white."},
        # {"prompt": "A vintage-style kitchen featuring an integrated dishwasher that's finished with a panel matching the surrounding warm wood cabinetry. Above the dishwasher, a muted stone countertop is adorned with an assortment of vintage kitchen tools and fresh green herbs in terracotta pots. The room exudes a rustic charm, with exposed ceiling beams and a classic farmhouse sink completing the homely setting."},
        {"prompt": "An engaging presentation slide featuring a clean white background with a diagonal light gray overlay, creating a minimalist yet informative design. The title in large teal font says ""Employee Satisfaction Survey Results"", followed by a subtitle ""Q1 2025 Overview"". Below the subtitle, a paragraph summarizes key findings: ""82% of employees report high job satisfaction, remote work flexibility is identified as the top-rated benefit, communication with management has improved by 15%, while the main concern remains the lack of career development opportunities."" To the right, a colorful pie chart titled ""Factors Influencing Satisfaction"" displays five segments labeled ""Work-Life Balance"", ""Compensation"", ""Growth Opportunities"", ""Management Support"", and ""Work Environment"", each with percentage values. Beneath the chart, a brief explanation reads ""While overall satisfaction remains high, targeted initiatives are needed to address professional growth concerns."" Three minimalist icons—a smiley face, a briefcase, and a bar graph—are placed at the bottom right for visual balance."},
        # {"prompt": "A sleek, black rectangular keyboard lies comfortably on the luxurious beige carpet of a quiet home office, bathed in the gentle sunlight of early afternoon. The keys of the keyboard show signs of frequent use, and it's positioned diagonally across the plush carpet, which is textured with subtle patterns. Nearby, a rolling office chair with a high back and adjustable armrests sits invitingly, hinting at a quick break taken by its usual occupant."},
        {"prompt": "Imagine a PPT slide where Global Tourism Recovery takes center stage against a clean white space accented by soft geometric shapes. The title reads ""Global Tourism Recovery"" and is complemented by a paragraph stating ""Post-pandemic travel is witnessing a surge, driven by digital nomadism, eco-tourism, and flexible booking options."". A visual chart labeled ""Tourist Arrivals by Continent"" includes categories like ""Europe"", ""Asia"", and ""Americas"". Decorative icons such as an airplane, a suitcase, and a globe add context. The slide concludes with a footer note: ""World Tourism Organization, 2025""."},
        {"prompt": "Spongebob depicted in the style of Dragon Ball Z."},
        {"prompt": "Lionel Messi portrayed as a sitcom character."},
        {"prompt": "Five red hockey sticks, each with a slender shape and a worn texture from frequent use, are propped against the frosty rink's white boards. The sky above casts a dim, featureless gray light over the area, accentuating the early morning stillness. On the icy surface of the rink, illuminated by the ambient outdoor lighting, the sticks' shadows form elongated silhouettes, showcasing their readiness for the day's practice."},
        {"prompt": "An array of freshly baked goods is presented on a rectangular silver tray with a reflective surface. To one side, vibrant sun-yellow lemon tarts, their delicate, flaky pastry crusts cradling a glistening citrus filling, are arranged neatly in a row. Adjacent to them, slightly purple-blueberry muffins, their tops golden brown and dusted with a fine layer of sugar, exhibit a contrasting texture. The pastries are placed against a backdrop of a marbled white countertop, with soft natural light enhancing their appetizing colors."},
        {"prompt": "The image depicts Rika Furude from the video game Yume Nikki."},
        {"prompt": "1girl, 2boys, multiple boys, anime, blonde hair, purple hair, sunglasses, logo"},
        {"prompt": "Show a racing car blazing along a neon-lit track, sparks flying as speed takes over. The poster bursts with energy, featuring ""Velocity Grand Prix 2024"" in bold, dynamic typography. A tagline near the bottom exclaims ""Feel the speed. Live the thrill. Witness history in motion."" Race day info completes the design: ""June 22 — Thunder Circuit — Visit www.velocitygp.com for tickets."""},
        {"prompt": "A sleek and modern PPT slide designed with a dark blue background and a subtle grid pattern to emphasize structure and professionalism. The large, bold title at the top-left reads ""Global Supply Chain Disruptions"". Directly beneath, a detailed paragraph states ""The COVID-19 pandemic exposed vulnerabilities in global logistics networks, leading to unprecedented delays and increased costs. Companies are now focusing on diversification of suppliers, regional manufacturing, and investment in digital tracking systems to enhance resilience."" On the right side, a bar chart titled ""Average Shipping Delays (2020-2024)"" compares delays across five continents, with axis labels such as ""North America"", ""Europe"", and ""Asia"", and numerical values clearly marked. Three icons—a cargo ship, a factory, and a location pin—are aligned beneath the chart. A footer note in smaller font reads ""Source: International Trade Association, 2024""."},
        {"prompt": "A visually striking PPT slide characterized by a gradient background transitioning from deep purple to light violet, enhancing the futuristic theme. The bold white title centered at the top reads ""Blockchain Applications Beyond Cryptocurrency"". A paragraph below elaborates ""Blockchain technology is revolutionizing industries such as healthcare, supply chain management, and voting systems by providing transparency, security, and decentralization. Smart contracts automate transactions, reducing the need for intermediaries."" On the left side, additional text highlights ""Key applications include secure medical records, transparent supply chains, and tamper-proof voting systems."" On the right, a flowchart titled ""Blockchain Workflow"" illustrates blocks connected in sequence, with arrows labeled ""Data Input"", ""Validation"", ""Block Creation"", and ""Ledger Update"". Small icons representing a hospital, a truck, and a ballot box are placed near the descriptive text. A watermark-style graphic of interconnected hexagons fills the background subtly."}
    ]
    config.validation_prompts = validation_prompts

    # Run initial validation before training starts
    if config.validation_prompts:
        torch.cuda.empty_cache()
        # Set transformer to student adapter before validation
        transformer.enable_adapters()
        transformer.set_adapter(student_adapter_name)
        run_qwenimage_t2i_validation(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoding_pipeline.text_encoder,
            accelerator=accelerator,
            scheduler=noise_scheduler,
            tokenizer=text_encoding_pipeline.tokenizer,
            args=config,
            valid_steps=config.num_validation_steps,
        )
        # Restore transformer to training state after validation
        transformer.train()
        transformer.enable_adapters()
        transformer.set_adapter(student_adapter_name)
        torch.cuda.empty_cache()

    fake_acc = 0
    gen_acc = 0
    acc_loss_fake = 0.0
    acc_loss_gen = 0.0
    first_epoch = 0
    resolutions = list(config.train_config.resolutions) if hasattr(config.train_config, "resolutions") else [(1382, 1382), (1664, 928), (928, 1664), (1472, 1104), (1104, 1472), (1584, 1056), (1056, 1584)]
    config.num_train_epochs = 99999 # only relies on the number of steps
    for epoch in range(first_epoch, config.num_train_epochs):
        # print('start epoch:', epoch)
        transformer.train()

        train_loss = 0.0

        # current_batch_dataloader = copy.deepcopy(train_dataloader)
        current_batch_dataloader = train_dataloader

        for step, macro_batch in enumerate(current_batch_dataloader): # pixels (1, 3, 1, 1024, 1024); prompts ...
            # print('fxfx step: ', step)

            num_samples_in_macro_batch = len(macro_batch["prompts"])  # 实际样本数量，而不是 dict 的 key 数量
            for micro_sample_idx in range(num_samples_in_macro_batch + config.gradient_accumulation_steps): # 把batch变成 (GA * (1 + K)) 个sample， 前 num_samples_in_macro_batch 个用于计算fake，后config.gradient_accumulation_steps个是用来算gen的sample
                should_update_step = False
                if micro_sample_idx >= num_samples_in_macro_batch:
                    true_idx = micro_sample_idx - config.gradient_accumulation_steps
                else:
                    true_idx = micro_sample_idx

                batch = {"prompts": [macro_batch["prompts"][true_idx]]}
                ##########################################################
                #       Part 1 数据准备与轨迹采样 (Sampling Phase)          #
                ##########################################################
                transformer.eval()
                with torch.no_grad():
                    ##########################################################
                    #              Part 1.1 text token preparation           #
                    ##########################################################
                    bsz = len(batch["prompts"])
                    device = accelerator.device
                    prompts = batch["prompts"]

                    # T2I: No condition images, pure text encoding
                    prompt_embeds, prompt_embeds_mask = compute_text_embeddings_t2i(
                        prompts, text_encoding_pipeline, config.max_sequence_length
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                    ##########################################################
                    #              Part 1.2 image latent preparation         #
                    ##########################################################
                    # T2I: Use fixed resolution or get from config
                    target_width, target_height = resolutions[random.randint(0, len(resolutions) - 1)]  # Default T2I resolution

                    latent_height = target_height // (config.vae_scale_factor * 2)
                    latent_width = target_width // (config.vae_scale_factor * 2)

                    if latent_height <= 0 or latent_width <= 0:
                        raise ValueError(
                            f"Invalid latent grid computed from target resolution {(target_width, target_height)} "
                            f"with vae_scale_factor {config.vae_scale_factor}."
                        )

                    art_model_input = torch.randn(
                        bsz,
                        latent_height * latent_width,
                        int(16*4),
                        device=device,
                        dtype=weight_dtype,
                    )

                    # T2I: Simple single shape (no reference images)
                    base_img_shape = (1, latent_height, latent_width)
                    img_shapes = [[base_img_shape]] * bsz


                    ##########################################################
                    #  DMD: no TDM trajectory — generator uses t=0 → add noise #
                    ##########################################################
                    k_step = int(config.distillation_config["K_step"])
                    fm_timesteps = flow_timesteps_for_k(k_step, device, dtype=weight_dtype)
                    # Single-step (K=1): start from pure noise at t=1 (flow maximum).
                    t_gen = fm_timesteps[0].expand(bsz).to(device=device, dtype=weight_dtype)
                ##########################################################
                #       Part 2      Train Fake Score                     #
                ##########################################################
                # Switch to fake adapter for training
                if micro_sample_idx < num_samples_in_macro_batch: # 先做fake的更新
                    transformer.set_adapter("fake")
                    transformer.train()

                    with torch.no_grad():
                        x0 = torch.randn_like(art_model_input, device=device, dtype=weight_dtype)
                        x1 = torch.randn_like(x0)
                        tau = torch.rand((bsz,), device=device, dtype=weight_dtype)
                        x_tau = predictor.add_noise(
                            samples=x0,
                            noise=x1,
                            t1=torch.zeros_like(tau),
                            t2=tau,
                        )

                    if reg_lambda > 0:
                        with torch.no_grad():
                            transformer.set_adapter("teacher")
                            _, _, v_pred_real = predictor.predict(
                                transformer, x_tau, tau,
                                img_shapes,
                                prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                return_all=True
                            )

                    # Fake adapter prediction (with gradient)
                    transformer.set_adapter("fake")
                    _, _, v_pred_fake = predictor.predict(
                        transformer, x_tau, tau,
                        img_shapes,
                        prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                        return_all=True
                    )

                    target = x1 - x0
                    loss = F.mse_loss(v_pred_fake.float(), target.float())
                    loss_fake = loss
                    acc_loss_fake += loss.detach().item()

                    # 检查loss是否异常，避免数值不稳定
                    if loss_fake > config.max_loss_fake:
                        print(f"Skipping backward pass due to abnormal loss_fake: {loss_fake.item():.2f}")

                    else:
                        if reg_lambda > 0:
                            loss_reg = F.mse_loss(v_pred_fake.float(), v_pred_real.float())
                            accelerator.backward((loss + reg_lambda * loss_reg) / config.gradient_accumulation_steps)
                        else:
                            accelerator.backward(loss / config.gradient_accumulation_steps)
                            fake_acc += 1

                    should_update_step = (fake_acc == config.gradient_accumulation_steps) or (micro_sample_idx == num_samples_in_macro_batch - 1)
                    if should_update_step:
                        torch.nn.utils.clip_grad_norm_(transformer_fake_parameters, config.max_grad_norm)
                        optimizer_fake.step()
                        optimizer_fake.zero_grad()
                        lr_scheduler_fake.step()
                        fake_acc = 0
                else:
                ##########################################################
                #       Part 3     Train K-step generator                #
                ##########################################################
                # Switch to student adapter for training

                    transformer.set_adapter(student_adapter_name)
                    transformer.train()

                    with torch.no_grad():
                        x0_anchor = torch.randn_like(art_model_input, device=device, dtype=weight_dtype)

                    k_step = int(config.distillation_config["K_step"])
                    fm_timesteps = flow_timesteps_for_k(k_step, device, dtype=weight_dtype)
                    t_anchor = fm_timesteps[0].expand(bsz).to(device=device, dtype=weight_dtype)

                    x_t_anchor = predictor.add_noise(
                        samples=x0_anchor,
                        noise=torch.randn_like(x0_anchor, dtype=weight_dtype),
                        t1=torch.zeros_like(t_anchor),
                        t2=t_anchor,
                    )

                    # 通过 "预测 x0/x1 → ODE 插值到 t_ode_next → 再加噪到 t_train" 的两步走
                    x1_pred_student, x0_pred_student = predictor.predict(
                        transformer, x_t_anchor, t_anchor,
                        img_shapes,
                        prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                        return_double=True
                    )

                    with torch.no_grad():
                           # 2. 采样两个考场的时间 tau
                        # tau_DM: [0, 0.98]
                        x_DM = 0
                        # Use multi step for calculating the IKL = \int [0,1] KL(p(x_t|x0_pred_student) || p(x_t|x0_real_pred_cond_CA)) dt + \int [0,1] KL(p(x_t|x0_pred_student) || p(x_t|x0_real_pred_cond_DM)) dt
                        for rcgm_rcgm in range(rcgm_step):
                            tau_DM = torch.rand_like(t_anchor)
                            if stable_fake: # stable_fake, now fuse to more to 1 for better detail
                                # tau_DM = tau_DM
                                tau_DM = tau_DM * timestep_shift / (1 + (timestep_shift - 1) * tau_DM)
                            x_tau_DM = predictor.add_noise(
                                samples=x0_pred_student,
                                noise=torch.randn_like(x1_pred_student, dtype=weight_dtype),
                                t1=torch.zeros_like(tau_DM),
                                t2=tau_DM
                            )

                            # Teacher (real) predictions
                            transformer.set_adapter("teacher")
                            if ucgm_step == 1:
                                x0_real_pred_cond_DM  = predictor.predict(
                                    transformer, x_tau_DM, tau_DM,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                    cfg=4.0,
                                )
                            else:
                                tau_DM_list = [tau_DM * i / ucgm_step for i in range(ucgm_step + 1)][::-1]
                                x0_real_pred_cond_DM  = predictor.predict_multistep(
                                    transformer, x_tau_DM, tau_DM_list,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                    cfg=4.0,
                                )

                            # Fake adapter prediction
                            transformer.enable_adapters()
                            transformer.set_adapter("fake")
                            if ucgm_step == 1:
                                x0_fake_pred_cond_DM  = predictor.predict(
                                    transformer, x_tau_DM, tau_DM,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                )
                            else:
                                x0_fake_pred_cond_DM  = predictor.predict_multistep(
                                    transformer, x_tau_DM, tau_DM_list,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                )

                            x_DM = x_DM + (x0_real_pred_cond_DM - x0_fake_pred_cond_DM)
                        x_DM = x_DM / rcgm_step
                        revised_latents = (x0_pred_student + x_DM).detach().float() # USING FIXED CFG = 4

                        # Store CFG diagnostic values for wandb logging
                        _diag_teacher_norm = x0_real_pred_cond_DM.float().norm().item()
                        _diag_fake_norm = x0_fake_pred_cond_DM.float().norm().item()
                        _diag_xdm_norm = x_DM.float().norm().item()
                        _diag_student_norm = x0_pred_student.float().norm().item()
                        _diag_diff_ratio = _diag_xdm_norm / (_diag_student_norm + 1e-8)

                    transformer.set_adapter(student_adapter_name)  # turn back to student model to get gradient
                    config.huber_c = 1e-3
                    weighting_factor = torch.abs(x0_pred_student - x0_real_pred_cond_DM).mean(dim=[1, 2], keepdim=True).detach()
                    weighting_factor = torch.clamp(weighting_factor, max=5.0, min = 0.01)

                    dmd_loss = torch.mean(
                        (torch.sqrt((x0_pred_student.float() - revised_latents.detach().float()) ** 2 + config.huber_c ** 2) - config.huber_c)
                    ) / weighting_factor
                    loss = dmd_loss
                    if LIPS_lambda > 0: # Loss for constraint G(x_anchor + eps) \approx G(x_anchor) (LIPS loss)
                        # Lipschitz regularization: ||G(x + eps) - G(x)|| should be small
                        # We want gradient to flow through G(x + eps), so we compute it with grad
                        # and use detached G(x) as target
                        if LIPS_sample == "Gaussian":
                            disturbance = LIPS_eps * torch.randn_like(x_t_anchor, dtype=weight_dtype)
                        elif LIPS_sample == "Uniform":
                            disturbance = LIPS_eps * (2 * torch.rand_like(x_t_anchor, dtype=weight_dtype) - 1) # use uniform distribution instead of normal distribution
                        else:
                            raise ValueError(f"Invalid LIPS_sample: {LIPS_sample}")

                        # Compute G(x + eps) with gradient
                        x0_pred_disturbed = predictor.predict(
                            transformer, x_t_anchor.detach() + disturbance, t_anchor.detach(),
                            img_shapes,
                            prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                        )
                        # Target: G(x) detached - we want G(x+eps) to be close to G(x)
                        lips_loss = torch.mean((x0_pred_disturbed.float() - x0_pred_student.float()) ** 2)
                        # if accelerator.is_main_process and global_step % 50 == 0:
                        #     logger.info(f"LIPS loss: {lips_loss.detach().item():.6f}")
                        # print(f"LIPS loss: {lips_loss.detach().item():.6f}")
                        loss = loss + LIPS_lambda * lips_loss
                    if diversity_lambda > 0 and dreamsim_model is not None:
                        # Perceptual diversity loss via antithetic noise + K-step ODE + DreamSim.
                        # Path 1 (reference): entire ODE chain is no_grad → frozen target.
                        # Path 2 (trainable): first K-1 steps no_grad, only last step has gradient →
                        #   gradient flows through last prediction + VAE decode + DreamSim.
                        with torch.no_grad():
                            pure_noise_1 = torch.randn_like(art_model_input, dtype=weight_dtype)
                            pure_noise_2 = torch.randn_like(art_model_input, dtype=weight_dtype)

                        K = diversity_ode_steps
                        # Uniform timesteps, then apply shift: s*t / (1 + (s-1)*t)
                        ode_ts_uniform = torch.linspace(1.0, 1.0 / max(K, 1), max(K, 1), device=device, dtype=weight_dtype)
                        ode_ts = 3 * ode_ts_uniform / (1.0 + (3 - 1.0) * ode_ts_uniform)

                        # --- Path 1: fully detached reference ---
                        with torch.no_grad():
                            x_t_1 = pure_noise_1
                            for step_i in range(K):
                                t_cur = torch.full((bsz,), ode_ts[step_i], device=device, dtype=weight_dtype)
                                x0_1 = predictor.predict(
                                    transformer, x_t_1, t_cur,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                )
                                if step_i < K - 1:
                                    ratio = (ode_ts[step_i + 1] / ode_ts[step_i]).item()
                                    x_t_1 = ratio * x_t_1 + (1.0 - ratio) * x0_1
                            x0_pred_div_1 = x0_1  # no grad

                        # --- Path 2: only last step has gradient ---
                        # Run first K-1 steps under no_grad to build up x_t,
                        # then do the final prediction with gradient.
                        with torch.no_grad():
                            x_t_2 = pure_noise_2
                            for step_i in range(K - 1):
                                t_cur = torch.full((bsz,), ode_ts[step_i], device=device, dtype=weight_dtype)
                                x0_2 = predictor.predict(
                                    transformer, x_t_2, t_cur,
                                    img_shapes,
                                    prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                                )
                                ratio = (ode_ts[step_i + 1] / ode_ts[step_i]).item()
                                x_t_2 = ratio * x_t_2 + (1.0 - ratio) * x0_2
                        # Final step WITH gradient
                        t_cur = torch.full((bsz,), ode_ts[K - 1], device=device, dtype=weight_dtype)
                        x0_pred_div_2 = predictor.predict(
                            transformer, x_t_2, t_cur,
                            img_shapes,
                            prompt_embeds, prompt_embeds_mask, txt_seq_lens,
                        )  # has grad only through last step

                        # --- Decode latents to pixel space for DreamSim ---
                        # Unpack: (B, H*W/4, 64) → (B, 16, 1, H_lat, W_lat)
                        _h_dec = 2 * (int(target_height) // (config.vae_scale_factor * 2))
                        _w_dec = 2 * (int(target_width) // (config.vae_scale_factor * 2))
                        _ch = x0_pred_div_1.shape[-1]  # 64

                        # Path 1: decode under no_grad (reference image)
                        with torch.no_grad():
                            lat_1 = x0_pred_div_1.float().view(bsz, _h_dec // 2, _w_dec // 2, _ch // 4, 2, 2)
                            lat_1 = lat_1.permute(0, 3, 1, 4, 2, 5).reshape(bsz, _ch // 4, 1, _h_dec, _w_dec)
                            lat_1 = lat_1 / latents_std + latents_mean
                            img_1 = vae.decode(lat_1.to(vae.dtype), return_dict=False)[0][:, :, 0]  # (B, 3, H_px, W_px)
                            img_1 = (img_1.float() * 0.5 + 0.5).clamp(0, 1)
                            img_1 = F.interpolate(img_1, size=(224, 224), mode='bilinear', align_corners=False)

                        # Path 2: decode WITH gradient (trainable path)
                        lat_2 = x0_pred_div_2.float().view(bsz, _h_dec // 2, _w_dec // 2, _ch // 4, 2, 2)
                        lat_2 = lat_2.permute(0, 3, 1, 4, 2, 5).reshape(bsz, _ch // 4, 1, _h_dec, _w_dec)
                        lat_2 = lat_2 / latents_std + latents_mean
                        img_2 = vae.decode(lat_2.to(vae.dtype), return_dict=False)[0][:, :, 0]  # (B, 3, H_px, W_px)
                        img_2 = (img_2.float() * 0.5 + 0.5).clamp(0, 1)
                        img_2 = F.interpolate(img_2, size=(224, 224), mode='bilinear', align_corners=False)

                        # DreamSim perceptual distance: cosine distance in [0, 2]
                        # Higher = more diverse. We maximize it by minimizing -dist.
                        dreamsim_dist = dreamsim_model(img_1.float(), img_2.float())  # (B,) or (B,1)
                        output_diff = dreamsim_dist.mean()
                        diversity_loss = -output_diff  # simple negation, bounded & stable
                        loss = loss + diversity_lambda * diversity_loss
                    accelerator.backward(loss / config.gradient_accumulation_steps)
                    acc_loss_gen += loss.detach().item()
                    gen_acc += 1
                    if gen_acc == config.gradient_accumulation_steps:
                        # Clip gradients for student adapter parameters
                        torch.nn.utils.clip_grad_norm_(transformer_student_parameters, config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        gen_acc = 0

                torch.cuda.empty_cache()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if micro_sample_idx == num_samples_in_macro_batch + config.gradient_accumulation_steps - 1: # Now seen one gen step as a real update step
                    global_step += 1

                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        if global_step % config.checkpointing_steps == 0 or global_step == config.max_train_steps:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(config.model_output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= config.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config.model_output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                    if (global_step % config.checkpointing_steps == 0) or (global_step == config.max_train_steps) or (global_step == 10):

                        save_checkpoint(
                            transformer,
                            optimizer,
                            lr_scheduler,
                            global_step + 1,
                            ckpt_root=config.model_output_dir,
                            is_final=False,
                            name=f"checkpoint-step-{global_step+1}",
                            save_model_only=False,
                            save_lora_only=getattr(config, "train_lora_only", False),
                            accelerator=accelerator,
                        )

                    accelerator.wait_for_everyone()

                    # Only main process logs to avoid duplicate logs
                    if accelerator.is_main_process:
                        logs = {
                            "loss_fake": acc_loss_fake / (config.gradient_accumulation_steps * train_k_step_generator_every_n_steps),
                            "loss_du": acc_loss_gen / config.gradient_accumulation_steps,
                            "loss_dmd": dmd_loss.detach().item(),
                            "lr": lr_scheduler.get_last_lr()[0]
                        }
                        if LIPS_lambda > 0:
                            logs["lips_loss"] = lips_loss.detach().item()
                        if diversity_lambda > 0:
                            logs["diversity_loss"] = diversity_loss.detach().item()
                            logs["output_diff"] = output_diff.detach().item()
                        logs["cfg_diag/teacher_norm"] = _diag_teacher_norm
                        logs["cfg_diag/fake_norm"] = _diag_fake_norm
                        logs["cfg_diag/x_DM_norm"] = _diag_xdm_norm
                        logs["cfg_diag/student_norm"] = _diag_student_norm
                        logs["cfg_diag/diff_ratio"] = _diag_diff_ratio
                        # 使用 update + refresh=False 的 set_postfix，避免重复刷新
                        progress_bar.update(1)
                        progress_bar.set_postfix(refresh=False, **logs)
                        accelerator.log(logs, step=global_step)

                    train_loss = 0.0
                    acc_loss_fake = 0.0
                    acc_loss_gen = 0.0

                    # ===== Validation (only after a real training step, only on main process) =====
                    if config.validation_prompts is not None:
                        should_run_validation = (
                            (config.validation_steps and global_step % config.validation_steps == 0)
                            or global_step == 1
                            or global_step == config.max_train_steps - 1
                        )
                    else:
                        should_run_validation = False

                    if should_run_validation:
                        torch.cuda.empty_cache()
                        # Set transformer to student adapter before validation
                        transformer.enable_adapters()
                        transformer.set_adapter(student_adapter_name)
                        run_qwenimage_t2i_validation(
                            vae=vae,
                            transformer=transformer,
                            text_encoder=text_encoding_pipeline.text_encoder,
                            accelerator=accelerator,
                            scheduler=noise_scheduler,
                            tokenizer=text_encoding_pipeline.tokenizer,
                            args=config,
                            valid_steps=config.num_validation_steps,
                        )
                        # Restore transformer to training state after validation
                        transformer.train()
                        transformer.enable_adapters()
                        transformer.set_adapter(student_adapter_name)
                        torch.cuda.empty_cache()

                # Check if reached max training steps
                if global_step >= config.max_train_steps:
                    break

                accelerator.wait_for_everyone()

        # Also break out of epoch loop when max steps reached
        if global_step >= config.max_train_steps:
            break

    # Save the full transformer model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            transformer,
            optimizer,
            lr_scheduler,
            global_step + 1,
            is_final=False,
            name=f"checkpoint-step-{global_step+1}",
            save_model_only=False,
            save_lora_only=getattr(config, "train_lora_only", False),
            accelerator=accelerator,
        )

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPC: Qwen-Image T2I DMD training")
    parser.add_argument("config", type=str, help="Path to mmengine YAML config")
    args = parser.parse_args()
    train_loop_full(args.config)
