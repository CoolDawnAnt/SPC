import logging
import os
import tempfile

import torch
from accelerate.logging import get_logger
from safetensors.torch import save_file

logger = get_logger(__name__)


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    global_step,
    is_final,
    ckpt_root="./outputs",
    name="checkpoint",
    local_debug=False,
    save_model_only=False,
    save_lora_only=False,
    accelerator=None,
):
    """Save checkpoint to a local directory (Accelerate-friendly, no Ray/S3)."""
    if accelerator is not None and not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    os.makedirs(ckpt_root, exist_ok=True)
    sub_save_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
    os.makedirs(sub_save_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model

    if save_lora_only:
        for adapter_name in ["student", "fake", "teacher"]:
            adapter_dir = os.path.join(sub_save_dir, f"adapter_{adapter_name}")
            os.makedirs(adapter_dir, exist_ok=True)
            unwrapped.save_lora_adapter(
                adapter_dir,
                adapter_name=adapter_name,
                safe_serialization=True,
            )
            logger.info("LoRA adapter '%s' saved to %s", adapter_name, adapter_dir)
    else:
        state = unwrapped.state_dict()
        torch.save(state, os.path.join(sub_save_dir, "transformer_full.pt"))

    if not save_model_only and not save_lora_only and optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "global_step": global_step,
            },
            os.path.join(sub_save_dir, "trainer_states.bin"),
        )

    metadata = {
        "global_step": global_step,
        "checkpoint_name": name,
        "is_final": is_final,
        "save_model_only": save_model_only,
        "save_lora_only": save_lora_only,
    }
    torch.save(metadata, os.path.join(sub_save_dir, "checkpoint_metadata.pt"))

    with open(os.path.join(sub_save_dir, "CHECKPOINT_INFO.txt"), "w") as f:
        f.write(f"Checkpoint Name: {name}\n")
        f.write(f"Global Step: {global_step}\n")
        f.write(f"Is Final: {is_final}\n")

    logger.info("Checkpoint saved to %s", sub_save_dir)
    if accelerator is not None:
        accelerator.wait_for_everyone()
