"""Local WebDataset loader for T2I DMD training (no Ray / S3)."""
from functools import partial
import glob
import json
import random
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from torch.utils.data import DataLoader


def get_nearest_divisor_size(h_original, w_original, h_divisor, w_divisor, max_pixels):
    orig_area = int(h_original) * int(w_original)
    if orig_area > max_pixels:
        scale = np.sqrt(max_pixels / orig_area)
        h_new = max(int(round(h_original * scale)), h_divisor)
        w_new = max(int(round(w_original * scale)), w_divisor)
        resized = True
    else:
        h_new, w_new = h_original, w_original
        resized = False

    h_aligned = (h_new // h_divisor) * h_divisor
    w_aligned = (w_new // w_divisor) * w_divisor
    if h_aligned == 0:
        h_aligned = h_divisor
    if w_aligned == 0:
        w_aligned = w_divisor
    if (h_aligned != h_original) or (w_aligned != w_original):
        resized = True

    while (h_aligned * w_aligned) > max_pixels:
        if h_aligned >= w_aligned and h_aligned > h_divisor:
            h_aligned -= h_divisor
        elif w_aligned > w_divisor:
            w_aligned -= w_divisor
        else:
            break
        resized = True

    return np.array([h_aligned, w_aligned], dtype=np.int64), resized


def _rgba_to_rgb_white(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] == 3:
        return arr
    x = arr.astype(np.float32)
    rgb = x[..., :3]
    a = x[..., 3:4]
    a01 = a if a.max() <= 1.0 else (a / 255.0)
    white = np.array([255.0, 255.0, 255.0], dtype=np.float32).reshape(1, 1, 3)
    out = rgb * a01 + white * (1.0 - a01)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ensure_bw_to_rgb(arr):
    if arr.ndim == 2:
        arr = np.stack((arr,) * 3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def process_single_t2i(
    row,
    max_pixels,
    h_divisor,
    w_divisor,
    default_resolution=1024,
):
    processed = {}
    sample_key = row.get("__key__", "")

    keys_to_pop = []
    for k, v in row.items():
        if k.endswith((".png", ".jpg", ".webp")):
            if isinstance(v, float) and np.isnan(v):
                keys_to_pop.append(k)
    for k in keys_to_pop:
        row.pop(k)

    def _load_image_single(row, key):
        for ext in ["png", "jpg", "webp"]:
            full_key = f"{key}.{ext}"
            if full_key in row:
                return row[full_key]
        return None

    image = _load_image_single(row, "image")
    has_gt_image = image is not None

    if has_gt_image:
        image = _ensure_bw_to_rgb(image)
        image = _rgba_to_rgb_white(image)
        H, W = image.shape[:2]
    else:
        H, W = default_resolution, default_resolution
        image = np.zeros((H, W, 3), dtype=np.uint8)

    processed["ori_size"] = np.asarray([H, W], dtype=np.int64)
    processed["has_gt_image"] = has_gt_image

    target_size, need_resize = get_nearest_divisor_size(H, W, h_divisor, w_divisor, max_pixels)

    if has_gt_image and need_resize:
        from skimage.transform import resize

        image = resize(
            image, (target_size[0], target_size[1]),
            preserve_range=True, anti_aliasing=True,
        ).astype(np.uint8)
    elif has_gt_image:
        image = image.astype(np.uint8)

    processed["input_size"] = target_size
    processed["image"] = image

    json_data = row.get("json") or {}
    if isinstance(json_data, bytes):
        json_data = json.loads(json_data.decode("utf-8"))
    prompt = json_data.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    processed["prompt"] = prompt
    processed["task_id"] = 0
    processed["sample_key"] = sample_key
    return processed


def collate_batch_fn_t2i(batch):
    n = len(next(iter(batch.values())))
    rows = [{k: batch[k][i] for k in batch} for i in range(n)]

    images = []
    ori_sizes = []
    input_sizes = []
    prompts = []
    keys = []
    has_gt_images = []

    for data in rows:
        image_tensor = TF.to_tensor(data["image"]).mul(2.0).sub(1.0)
        images.append(image_tensor)
        ori_sizes.append(torch.tensor(data["ori_size"]))
        input_sizes.append(torch.tensor(data["input_size"]))
        prompts.append(data["prompt"])
        keys.append(str(data.get("sample_key", "")))
        has_gt_images.append(data.get("has_gt_image", True))

    return {
        "images": images,
        "edit_images": [[] for _ in range(len(images))],
        "prompts": prompts,
        "ori_sizes": torch.stack(ori_sizes),
        "input_sizes": torch.stack(input_sizes),
        "num_edits": torch.zeros(len(images), dtype=torch.int64),
        "edit_images_vl": [[] for _ in range(len(images))],
        "task_ids": torch.zeros(len(images), dtype=torch.int64),
        "keys": keys,
        "has_gt_images": has_gt_images,
    }


collate_batch_fn_with_unified_size = collate_batch_fn_t2i


def _resolve_data_urls(data_path: Union[str, List[str]]) -> List[str]:
    if isinstance(data_path, str):
        paths = [data_path]
    else:
        paths = list(data_path)

    urls: List[str] = []
    for p in paths:
        if p.endswith("/"):
            urls.extend(sorted(glob.glob(p + "*.tar")))
        elif "*" in p:
            urls.extend(sorted(glob.glob(p)))
        elif p.endswith(".tar"):
            urls.append(p)
        else:
            urls.extend(sorted(glob.glob(p + "/*.tar")))
    if not urls:
        raise FileNotFoundError(f"No WebDataset tar shards found for data_path={data_path!r}")
    return urls


def create_train_dataloader(
    dataset_config: Dict[str, Any],
    batch_size: int,
    num_workers: int = 0,
    shuffle_seed: int = 42,
):
    urls = _resolve_data_urls(dataset_config["data_path"])
    max_pixels = dataset_config.get("max_pixels", 1024 * 1024)
    h_divisor = dataset_config.get("h_divisor", 32)
    w_divisor = dataset_config.get("w_divisor", 32)
    default_resolution = dataset_config.get("default_resolution", 1024)

    preprocess_fn = partial(
        process_single_t2i,
        max_pixels=max_pixels,
        h_divisor=h_divisor,
        w_divisor=w_divisor,
        default_resolution=default_resolution,
    )

    def _decode_json(sample):
        if "json" in sample and isinstance(sample["json"], bytes):
            sample["json"] = json.loads(sample["json"].decode("utf-8"))
        return sample

    def _map_sample(sample):
        return preprocess_fn(sample)

    rng = random.Random(shuffle_seed)
    dataset = wds.WebDataset(urls, shardshuffle=rng.randint(0, 2**31 - 1), repeat=True)
    dataset = dataset.shuffle(512, rng=rng)
    dataset = dataset.decode("rgb").map(_decode_json).map(_map_sample)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch_fn_t2i,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
