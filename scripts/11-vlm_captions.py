"""Generate Source B captions by running Qwen3-VL on map tile patches.

For each patch in the Source A captions JSONL, loads the patch image and runs
Qwen2-VL with the pre-built vlm_prompt (GB1900 context + instruction) to produce
a visually-grounded description of symbols, land use, boundaries, and vegetation.

Two OS characteristic sheets are passed as visual reference images in every
message so the model can match what it sees in the patch against the symbol key:
  - 12807_128076894.png  Plate IV, six-inch, 1923 (primary reference)
  - 12807_128076789.png  Engraved six-inch characteristic sheet, 1897
                         (large illustrated land-cover symbol examples)

Output JSONL format:
  {"patch_id": "...", "parent_id": "...", "caption": "...", "source": "vlm"}

Resumable: already-written patch_ids in the output file are skipped.

Usage (on Isambard):
  uv run python scripts/11-vlm_captions.py \\
      --captions data/patches_6inch_2nd_ed/captions.jsonl \\
      --patches-dir data/patches_6inch_2nd_ed \\
      --output data/patches_6inch_2nd_ed/vlm_captions.jsonl \\
      --model Qwen/Qwen3-VL-7B-Instruct \\
      --batch-size 8
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as err:
    raise ImportError("Install qwen-vl-utils: uv add qwen-vl-utils") from err


# ---------------------------------------------------------------------------
# Reference materials loaded once at startup
# ---------------------------------------------------------------------------

_SHEETS_DIR = Path(__file__).parent.parent / "data/characteristic_sheets"

_SHEET_PRIMARY = _SHEETS_DIR / "12807_128076894.png"  # Plate IV, six-inch 1923
_SHEET_SECONDARY = _SHEETS_DIR / "12807_128076789.png"  # Engraved six-inch 1897
_ABBREV_JSON = _SHEETS_DIR / "abbreviations.json"
_WRITING_1914_JSON = _SHEETS_DIR / "character_of_writing_1914.json"

SYSTEM_PROMPT = (
    "You are an expert in historical Ordnance Survey maps. "
    "You are analysing a 512x512 pixel patch from an Ordnance Survey six-inch to the mile map "
    "(approximately 1:10,560 scale, surveyed c.1888-1914). "
    "Two OS characteristic sheets and two reference JSON files (abbreviations and character of writing) are provided before the map patch. "
    "Use them to identify symbols, land cover types, boundaries, linear features, "
    "and the meaning of any text or abbreviations visible in the patch. "
    "Provide detailed, accurate descriptions of the map patch, including locations of features and spatial relationships. "
    "Do not hallucinate features that are not present. "
)


def load_reference_materials() -> tuple[list[Image.Image], str, str]:
    sheets = []
    for path in (_SHEET_PRIMARY, _SHEET_SECONDARY):
        if not path.exists():
            raise FileNotFoundError(f"Characteristic sheet not found: {path}")
        sheets.append(Image.open(path).convert("RGB"))
    abbrev_text = _ABBREV_JSON.read_text()
    writing_1914_text = _WRITING_1914_JSON.read_text()
    return sheets, abbrev_text, writing_1914_text


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_batch(
    model,
    processor,
    ref_sheets: list[Image.Image],
    abbrev_text: str,
    writing_1914_text: str,
    images: list[Image.Image],
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    ref_content = [
        {"type": "image", "image": ref_sheets[0]},
        {
            "type": "text",
            "text": "Conventional signs for the six-inch series (Plate IV, 1923).",
        },
        {"type": "image", "image": ref_sheets[1]},
        {
            "type": "text",
            "text": "Characteristic sheet for the engraved six-inch maps, showing land-cover symbols (1897).",
        },
        {
            "type": "text",
            "text": f"Abbreviations used on this map series (from OS 1914 characteristic sheet):\n{abbrev_text}",
        },
        {
            "type": "text",
            "text": f"Character of writing conventions, font style per feature type (OS 1914):\n{writing_1914_text}",
        },
    ]
    messages_batch = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *ref_content,
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        for img, prompt in zip(images, prompts, strict=True)
    ]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Strip the input tokens to get only generated text
    trimmed = [
        out[len(inp) :] for out, inp in zip(output_ids, inputs.input_ids, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Source B VLM captions for map patches"
    )
    parser.add_argument(
        "--captions",
        required=True,
        help="Source A captions JSONL (from 10-generate_captions.py)",
    )
    parser.add_argument(
        "--patches-dir",
        required=True,
        help="Directory containing patch PNG files",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: <captions_dir>/vlm_captions.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        help="HuggingFace model ID (default: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per caption (default: 512)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of patches to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to from_pretrained (default: cuda)",
    )
    args = parser.parse_args()

    captions_path = Path(args.captions)
    patches_dir = Path(args.patches_dir)
    output_path = (
        Path(args.output)
        if args.output
        else captions_path.parent / "vlm_captions.jsonl"
    )

    # Load Source A captions
    source_a = {}
    with open(captions_path) as f:
        for line in f:
            rec = json.loads(line)
            source_a[rec["patch_id"]] = rec

    # Resume: skip already-written patch_ids
    done: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["patch_id"])
        print(f"Resuming — {len(done):,} already written, skipping.")

    # Build work list
    work = [
        rec
        for pid, rec in source_a.items()
        if pid not in done and (patches_dir / pid).exists()
    ]
    if args.max_samples:
        work = work[: args.max_samples]

    missing = sum(
        1
        for rec in source_a.values()
        if rec["patch_id"] not in done and not (patches_dir / rec["patch_id"]).exists()
    )
    print(f"Patches to process: {len(work):,}  |  missing image files: {missing:,}")

    if not work:
        print("Nothing to do.")
        return

    # Load reference materials once
    print("Loading characteristic sheets and reference JSON …")
    ref_sheets, abbrev_text, writing_1914_text = load_reference_materials()

    # Load model
    print(f"Loading {args.model} …")
    model, processor = load_model(args.model, args.device)
    model.eval()

    n_written = 0
    with open(output_path, "a") as fout:
        for i in tqdm(range(0, len(work), args.batch_size), desc="Batches"):
            batch_recs = work[i : i + args.batch_size]
            images = [
                Image.open(patches_dir / rec["patch_id"]).convert("RGB")
                for rec in batch_recs
            ]
            prompts = [rec["vlm_prompt"] for rec in batch_recs]

            captions = run_batch(
                model,
                processor,
                ref_sheets,
                abbrev_text,
                writing_1914_text,
                images,
                prompts,
                args.max_new_tokens,
                args.device,
            )

            for rec, caption in zip(batch_recs, captions, strict=True):
                fout.write(
                    json.dumps(
                        {
                            "patch_id": rec["patch_id"],
                            "parent_id": rec["parent_id"],
                            "caption": caption,
                            "source": "vlm",
                        }
                    )
                    + "\n"
                )
                n_written += 1
            fout.flush()

    print(f"Done — wrote {n_written:,} captions → {output_path}")


if __name__ == "__main__":
    main()
