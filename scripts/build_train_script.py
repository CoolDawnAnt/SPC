#!/usr/bin/env python3
"""Generate spc/train_dmd_qwen_t2i.py from internal Canva source (one-time / refresh)."""
import re
from pathlib import Path

SRC = Path("/home/coder/work/canva/ai_platform_exploration_server/models/qwenedit_fewstep/src/train_lora_DMD2_fm_qwen_t2i.py")
DST = Path(__file__).resolve().parents[1] / "spc" / "train_dmd_qwen_t2i.py"

HEADER = '''# ruff: noqa
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


'''

DMD_PART_13_REPLACEMENT = '''
                    ##########################################################
                    #  DMD: no TDM trajectory — generator uses t=0 → add noise #
                    ##########################################################
                    k_step = int(config.distillation_config["K_step"])
                    fm_timesteps = flow_timesteps_for_k(k_step, device, dtype=weight_dtype)
                    # Single-step (K=1): start from pure noise at t=1 (flow maximum).
                    t_gen = fm_timesteps[0].expand(bsz).to(device=device, dtype=weight_dtype)
'''

DMD_FAKE_INNER = """                        x0 = torch.randn_like(art_model_input, device=device, dtype=weight_dtype)
                        x1 = torch.randn_like(x0)
                        tau = torch.rand((bsz,), device=device, dtype=weight_dtype)"""

DMD_GEN_BLOCK = '''
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
'''


def main():
    text = SRC.read_text()

    # Drop internal preamble through old imports (keep from calculate_dimensions / Predictor)
    start = text.index("\n\n@torch.no_grad()\ndef sample_trajectory")
    end_predictor = text.index("\n\ndef calculate_dimensions(")
    tail_start = text.index("\n\ndef make_serializable_for_tracking")
    text = HEADER + text[end_predictor:tail_start] + text[tail_start:]

    # Remove sample_trajectory if still present (should be gone)
    text = re.sub(
        r"@torch.no_grad\(\)\ndef sample_trajectory\(.*?\n\n\ndef calculate_dimensions",
        "\n\ndef calculate_dimensions",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Imports already replaced by HEADER - remove duplicate old header if any
    if "from ray import train" in text:
        raise RuntimeError("Ray imports still present after patch")

    # Wandb key injection: use env only
    text = text.replace(
        """    try:
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']['WANDB_API_KEY']
        config = config['config_file']
    except:
        print('wandb_api_key not found')
        pass


""",
        "",
    )

    # dataset_config
    if "config.dataset_config" not in text:
        text = text.replace(
            "    config.local_debug = local_debug_cfg or {}",
            "    config.dataset_config = _get_section(\"dataset_config\") or {}\n    config.local_debug = local_debug_cfg or {}",
        )

    # Dataloader
    text = text.replace(
        """    train_dataset = ray_train.get_dataset_shard("train")
    train_dataloader = train_dataset.iter_torch_batches(
        batch_size=config.train_batch_size * config.gradient_accumulation_steps * train_k_step_generator_every_n_steps,
        collate_fn=collate_batch_fn_with_unified_size,
        drop_last=True,
        prefetch_batches=3,
        local_shuffle_seed=42,
        local_shuffle_buffer_size=config.train_batch_size * config.gradient_accumulation_steps * train_k_step_generator_every_n_steps * 3,
    )""",
        """    macro_batch_size = (
        config.train_batch_size
        * config.gradient_accumulation_steps
        * train_k_step_generator_every_n_steps
    )
    train_dataloader = create_train_dataloader(
        config.dataset_config,
        batch_size=macro_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle_seed=config.local_shuffle_seed,
    )""",
    )

    # Resume: strip Ray S3 block — keep only local accelerator.load_state branch
    s3_resume_start = text.index("        else:\n            # load from remote")
    s3_resume_end = text.index("\n    else:\n        initial_global_step = 0\n        global_step = 0\n\n    progress_bar")
    text = text[:s3_resume_start] + text[s3_resume_end:]

    # Fix resume local_debug check
    text = text.replace(
        "        if config.get('local_debug', False): # local debug",
        "        if True:  # local checkpoint resume",
    )

    # Part 1.3 -> DMD comment block
    part13_start = text.index("                    ##########################################################\n                    #              Part 1.3 Student trajectory sampling")
    part13_end = text.index("                ##########################################################\n                #       Part 2      Train Fake Score")
    text = text[:part13_start] + DMD_PART_13_REPLACEMENT + text[part13_end:]

    # Part 2 fake x0
    text = text.replace(
        """                    # x_anchor: rand bsz x_t (使用之前采样的 rand_step_idx)
                    x0_anchor = torch.randn_like(art_model_input, device=device, dtype=weight_dtype)
                    for i in range(bsz):
                        x0_anchor[i] = student_x0_list[rand_step_idx][i]

                    t_anchor = fm_timesteps[rand_step_idx].to(device=device, dtype=weight_dtype)
                    t_ode_next = fm_timesteps[rand_step_idx - 1].to(device=device, dtype=weight_dtype)
                    tau = torch.rand_like(t_anchor)
                    # if stable_fake: # stable_fake, now fuse to more to 1 for better detail
                    #     tau = tau * timestep_shift / (1 + (timestep_shift - 1) * tau)

                    with torch.no_grad():
                        transformer.set_adapter(student_adapter_name)

                        x0 = x0_anchor
                        x1 = torch.randn_like(x0)
                        x_tau = predictor.add_noise(
                            samples=x0,
                            noise=x1,
                            t1=torch.zeros_like(tau),
                            t2=tau
                        )""",
        """                    with torch.no_grad():
""" + DMD_FAKE_INNER + """
                        x_tau = predictor.add_noise(
                            samples=x0,
                            noise=x1,
                            t1=torch.zeros_like(tau),
                            t2=tau,
                        )""",
    )

    # Part 3 generator
    text = text.replace(
        """                    with torch.no_grad():
                        # x_anchor: rand bsz x_t (使用 Part 1.3 中采样的 rand_step_idx)
                        x0_anchor = torch.randn_like(art_model_input, device=device, dtype=weight_dtype)
                        for i in range(bsz):
                            x0_anchor[i] = student_x0_list[rand_step_idx][i]

                    t_anchor = fm_timesteps[rand_step_idx].to(device=device, dtype=weight_dtype)

                    x_t_anchor = predictor.add_noise(
                        samples=x0_anchor,
                        noise=torch.randn_like(x0_anchor, dtype=weight_dtype),
                        t1=torch.zeros_like(t_anchor),
                        t2=t_anchor
                    )""",
        DMD_GEN_BLOCK.strip(),
    )

    # save_checkpoint calls: pass accelerator, fix ckpt_root default
    text = text.replace(
        "def save_checkpoint(",
        "def _removed_save_checkpoint(",
        1,
    )
    # Remove old save_checkpoint function body - already removed in HEADER approach

    # Actually old save_checkpoint might still be in file if we didn't remove it
    if "def _removed_save_checkpoint" in text or "Ray Train's checkpoint" in text:
        sc_start = text.find("def save_checkpoint(")
        if sc_start == -1:
            sc_start = text.find("def _removed_save_checkpoint(")
        if sc_start != -1:
            sc_end = text.find("\n\n@torch.no_grad()", sc_start)
            if sc_end == -1:
                sc_end = text.find("\n\ndef calculate_dimensions", sc_start)
            if sc_end != -1 and "flow_timesteps_for_k" not in text[sc_start:sc_end]:
                text = text[:sc_start] + text[sc_end:]

    text = text.replace(
        """                        save_checkpoint(
                            transformer,
                            optimizer,
                            lr_scheduler,
                            global_step + 1,
                            ckpt_root=config.model_output_dir,
                            is_final=False,
                            name=f"checkpoint-step-{global_step+1}",
                            local_debug=config.local_debug,
                            save_model_only=False,
                            save_lora_only=getattr(config, "train_lora_only", False),
                        )""",
        """                        save_checkpoint(
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
                        )""",
    )
    text = text.replace(
        """        save_checkpoint(
            transformer,
            optimizer,
            lr_scheduler,
            global_step + 1,
            is_final=False,
            name=f"checkpoint-step-{global_step+1}",
            local_debug=config.local_debug,
            save_model_only=False,
            save_lora_only=getattr(config, "train_lora_only", False),
        )""",
        """        save_checkpoint(
            transformer,
            optimizer,
            lr_scheduler,
            global_step + 1,
            is_final=False,
            name=f"checkpoint-step-{global_step+1}",
            save_model_only=False,
            save_lora_only=getattr(config, "train_lora_only", False),
            accelerator=accelerator,
        )""",
    )

    # __main__
    text = re.sub(
        r'if __name__ == "__main__":.*',
        '''if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPC: Qwen-Image T2I DMD training")
    parser.add_argument("config", type=str, help="Path to mmengine YAML config")
    args = parser.parse_args()
    train_loop_full(args.config)
''',
        text,
        flags=re.DOTALL,
    )

    DST.write_text(text)
    print(f"Wrote {DST} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
