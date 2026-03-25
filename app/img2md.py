from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28

DEFAULT_MARKDOWN_PROMPT = """Convert this image to Markdown.
Return Markdown only."""

DEFAULT_QWEN_PRESET = "qwen7b"


@dataclass(frozen=True)
class ModelPreset:
    key: str
    model_id: str
    description: str
    default_prompt: str
    family: str = "generic"
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS


MODEL_PRESETS = {
    "qwen3b": ModelPreset(
        key="qwen3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Smaller general VLM baseline for image-to-markdown tasks.",
        default_prompt=DEFAULT_MARKDOWN_PROMPT,
        family="qwen",
    ),
    "qwen7b": ModelPreset(
        key="qwen7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Stronger default general VLM for image-to-markdown tasks.",
        default_prompt=DEFAULT_MARKDOWN_PROMPT,
        family="qwen",
    ),
    "paddleocr_vl": ModelPreset(
        key="paddleocr_vl",
        model_id="PaddlePaddle/PaddleOCR-VL",
        description="Document parsing model tuned for OCR, tables, formulas, and charts.",
        default_prompt=DEFAULT_MARKDOWN_PROMPT,
    ),
    "mineru2_5": ModelPreset(
        key="mineru2_5",
        model_id="opendatalab/MinerU2.5-2509-1.2B",
        description="High-resolution document parsing model optimized for structured documents.",
        default_prompt=DEFAULT_MARKDOWN_PROMPT,
    ),
}


@dataclass
class ConversionResult:
    input_file: str
    output_file: str | None
    status: str
    elapsed_seconds: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert images in a folder to Markdown with a selectable local VLM."
    )
    parser.add_argument("--input-dir", default="./in", help="Directory containing source images.")
    parser.add_argument("--output-dir", default="./out", help="Directory where markdown files are written.")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default=DEFAULT_QWEN_PRESET,
        help="Named model preset to use. Defaults to qwen7b.",
    )
    parser.add_argument(
        "--model-id",
        help="Optional Hugging Face model id override. If omitted, the selected preset is used.",
    )
    parser.add_argument(
        "--list-model-presets",
        action="store_true",
        help="Print available model presets and exit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate per image.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        help="Optional processor min_pixels override.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Optional processor max_pixels override.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional inline prompt override.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional file containing a custom prompt.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip files when the markdown output already exists.",
    )
    parser.add_argument(
        "--combine-output",
        action="store_true",
        help="Also write a combined markdown file containing every result.",
    )
    parser.add_argument(
        "--combined-name",
        default="combined.md",
        help="Filename for the combined markdown output.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Filename for the conversion manifest written in the output directory.",
    )
    return parser.parse_args()


def list_model_presets() -> str:
    lines = ["Available model presets:"]
    for key in sorted(MODEL_PRESETS):
        preset = MODEL_PRESETS[key]
        lines.append(f"- {preset.key}: {preset.model_id} | {preset.description}")
    return "\n".join(lines)


def iter_images(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_prompt(args: argparse.Namespace, preset: ModelPreset) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return preset.default_prompt


def get_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


def image_has_transparency(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def prepare_image_for_model(image_path: Path) -> tuple[Path, bool]:
    with Image.open(image_path) as image:
        if not image_has_transparency(image):
            return image_path, False

        rgba_image = image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba_image).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        composited.save(temp_path, format="PNG")
        return temp_path, True


class Img2MdConverter:
    def __init__(
        self,
        preset: ModelPreset,
        model_id: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
        prompt: str,
    ) -> None:
        self.preset = preset
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN") or None

        processor_kwargs: dict[str, object] = {"token": self.hf_token}
        if preset.family == "qwen":
            processor_kwargs["min_pixels"] = min_pixels
            processor_kwargs["max_pixels"] = max_pixels
            processor_kwargs["use_fast"] = False

        print("Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        print("Processor ready.", flush=True)

        model_kwargs: dict[str, object] = {
            "torch_dtype": get_torch_dtype(self.device),
            "low_cpu_mem_usage": True,
            "token": self.hf_token,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["attn_implementation"] = "sdpa"

        print("Loading model weights... This can take several minutes on the first run.", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        print("Model ready.", flush=True)

    def convert_image(self, image_path: Path) -> str:
        prepared_path, is_temporary = prepare_image_for_model(image_path)
        if is_temporary:
            print(f"Flattened transparency onto white background: {image_path.name}", flush=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(prepared_path)},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            target_device = next(self.model.parameters()).device
            inputs = inputs.to(target_device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            prompt_length = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, prompt_length:]
            markdown = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return markdown.strip()
        finally:
            if is_temporary and prepared_path.exists():
                prepared_path.unlink()


def write_manifest(
    output_dir: Path,
    manifest_name: str,
    model_preset: str,
    model_id: str,
    prompt: str,
    results: list[ConversionResult],
) -> None:
    manifest = {
        "model_preset": model_preset,
        "model_id": model_id,
        "prompt": prompt,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": [
            {
                "input_file": item.input_file,
                "output_file": item.output_file,
                "status": item.status,
                "elapsed_seconds": round(item.elapsed_seconds, 3),
                "error": item.error,
            }
            for item in results
        ],
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.list_model_presets:
        print(list_model_presets())
        return 0

    preset = MODEL_PRESETS[args.model_preset]
    model_id = args.model_id or preset.model_id
    min_pixels = args.min_pixels or preset.min_pixels
    max_pixels = args.max_pixels or preset.max_pixels
    prompt = resolve_prompt(args, preset)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    images = list(iter_images(input_dir))
    if not images:
        print(f"No supported images found in {input_dir}", file=sys.stderr)
        write_manifest(output_dir, args.manifest_name, preset.key, model_id, prompt, [])
        return 0

    print(f"Model preset: {preset.key}", flush=True)
    print(f"Model id: {model_id}", flush=True)
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}", flush=True)
    if os.getenv("HF_TOKEN"):
        print("HF_TOKEN detected.", flush=True)
    else:
        print("HF_TOKEN not set. Public download limits may be slower.", flush=True)

    converter = Img2MdConverter(
        preset=preset,
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=args.max_new_tokens,
        prompt=prompt,
    )

    results: list[ConversionResult] = []
    combined_parts: list[str] = []
    failures = 0

    for image_path in images:
        output_path = output_dir / f"{image_path.name}.md"
        if output_path.exists() and args.no_overwrite:
            print(f"Skipping existing output: {output_path.name}", flush=True)
            results.append(
                ConversionResult(
                    input_file=image_path.name,
                    output_file=output_path.name,
                    status="skipped",
                    elapsed_seconds=0.0,
                )
            )
            continue

        started_at = time.perf_counter()
        try:
            markdown = converter.convert_image(image_path)
            output_path.write_text(markdown + "\n", encoding="utf-8")
            elapsed = time.perf_counter() - started_at
            results.append(
                ConversionResult(
                    input_file=image_path.name,
                    output_file=output_path.name,
                    status="ok",
                    elapsed_seconds=elapsed,
                )
            )
            print(f"Converted {image_path.name} -> {output_path.name} ({elapsed:.2f}s)", flush=True)
            if args.combine_output:
                combined_parts.append(f"# {image_path.name}\n\n{markdown}\n")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started_at
            failures += 1
            results.append(
                ConversionResult(
                    input_file=image_path.name,
                    output_file=None,
                    status="error",
                    elapsed_seconds=elapsed,
                    error=str(exc),
                )
            )
            print(f"Failed {image_path.name}: {exc}", file=sys.stderr, flush=True)

    if args.combine_output and combined_parts:
        combined_path = output_dir / args.combined_name
        combined_path.write_text("\n".join(combined_parts).strip() + "\n", encoding="utf-8")

    write_manifest(output_dir, args.manifest_name, preset.key, model_id, prompt, results)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
