from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

DEFAULT_PROMPT = """Convert this image into clean Markdown.

Rules:
- Output Markdown only.
- Preserve the original reading order.
- Keep the original language found in the image.
- Reproduce headings, paragraphs, lists, tables, and code blocks when present.
- Use GitHub-flavored Markdown tables when table structure is visible.
- For charts, diagrams, slides, or UI screens, summarize visible text and structure with Markdown headings and bullet points.
- Do not invent text. If part of the text is unreadable, write [unclear].
"""


@dataclass
class ConversionResult:
    input_file: str
    output_file: str | None
    status: str
    elapsed_seconds: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert images in a folder to Markdown with Qwen2.5-VL."
    )
    parser.add_argument("--input-dir", default="./in", help="Directory containing source images.")
    parser.add_argument("--output-dir", default="./out", help="Directory where markdown files are written.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face model id for Qwen2.5-VL.",
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
        default=DEFAULT_MIN_PIXELS,
        help="Minimum image pixels passed to the processor.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum image pixels passed to the processor.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional file containing a custom prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files in the output directory.",
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


def iter_images(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return DEFAULT_PROMPT
    return prompt_file.read_text(encoding="utf-8").strip()


class Img2MdConverter:
    def __init__(
        self,
        model_id: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
        prompt: str,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["attn_implementation"] = "sdpa"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )
        if self.device == "cpu":
            self.model.to("cpu")

    def convert_image(self, image_path: Path) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
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


def write_manifest(output_dir: Path, manifest_name: str, model_id: str, results: list[ConversionResult]) -> None:
    manifest = {
        "model_id": model_id,
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
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    prompt = read_prompt(args.prompt_file)
    images = list(iter_images(input_dir))
    if not images:
        print(f"No supported images found in {input_dir}", file=sys.stderr)
        write_manifest(output_dir, args.manifest_name, args.model_id, [])
        return 0

    print(f"Loading model: {args.model_id}", flush=True)
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}", flush=True)
    converter = Img2MdConverter(
        model_id=args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        prompt=prompt,
    )

    results: list[ConversionResult] = []
    combined_parts: list[str] = []
    failures = 0

    for image_path in images:
        output_path = output_dir / f"{image_path.name}.md"
        if output_path.exists() and not args.overwrite:
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

    write_manifest(output_dir, args.manifest_name, args.model_id, results)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
