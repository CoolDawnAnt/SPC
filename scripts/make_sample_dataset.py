#!/usr/bin/env python3
"""Create a tiny WebDataset tar for smoke tests."""
import io
import json
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample"
OUT_TAR = OUT_DIR / "sample-000000.tar"

PROMPTS = [
    "A red circle on a white background.",
    "A blue square on a gray background.",
    "A green triangle, simple flat illustration.",
    "A yellow star on black background.",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT_TAR, "w") as tar:
        for i, prompt in enumerate(PROMPTS):
            key = f"{i:06d}"
            meta = {"prompt": prompt, "task": "text_to_image"}
            meta_bytes = json.dumps(meta).encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))

            img = Image.new("RGB", (512, 512), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), prompt[:40], fill=(0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.image.png")
            info.size = len(png)
            tar.addfile(info, io.BytesIO(png))

    print(f"Wrote {OUT_TAR} ({len(PROMPTS)} samples)")


if __name__ == "__main__":
    main()
