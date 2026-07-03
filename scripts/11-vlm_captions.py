"""Generate Source B captions using a two-stage quadrant → synthesis approach.

Stage 1 — Quadrant descriptions: the 512x512 patch is split into four 256x256
quadrant crops (NW, NE, SW, SE). Each crop is described individually with the
reference characteristic sheets as context.

Stage 2 — Synthesis: the full 512x512 patch is shown together with all four
quadrant descriptions as grounding context. The model writes a single unified
caption integrating spatial information from all quadrants.

Intermediate quadrant descriptions are saved to a sidecar JSONL so the synthesis
stage can be re-run without repeating the quadrant passes (resumable at both stages).

Multi-node usage on Isambard-AI:
  Submit via batch/11-vlm_captions.sh (2 nodes x 4 GPUs = 8 GPUs total).
  Only SLURM_PROCID=0 runs inference; worker nodes join the Ray cluster and wait.

Output JSONL:
  {"patch_id": "...", "parent_id": "...", "caption": "...", "source": "vlm"}

Usage (on Isambard):
  See batch/11-vlm_captions.sh
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Reference materials
# ---------------------------------------------------------------------------

_SHEETS_DIR = Path(__file__).parent.parent / "data/characteristic_sheets"

_SHEET_PRIMARY = _SHEETS_DIR / "12807_128076894.png"  # Plate IV, six-inch 1923
_SHEET_SECONDARY = _SHEETS_DIR / "12807_128076789.png"  # Engraved six-inch 1897
_ABBREV_JSON = _SHEETS_DIR / "abbreviations.json"
_WRITING_1914_JSON = _SHEETS_DIR / "character_of_writing_1914.json"

QUAD_NAMES = ("NW", "NE", "SW", "SE")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

QUADRANT_SYSTEM_PROMPT = (
    "You are an expert in historical Ordnance Survey maps. "
    "You are analysing a 256x256 pixel quadrant crop from an Ordnance Survey six-inch to the mile map (approximately 1:10,560 scale, surveyed c.1888-1914). "
    "Two OS characteristic sheets are provided before the map crop. "
    "Use them to identify symbols, land cover types, boundaries and linear features visible in the crop. "
    "The final image in this message is the map crop to describe. All preceding images are reference materials only — do not describe the reference sheets. "
    "Name all visible features that are present on the map using the reference materials to guide you. "
    "If it is a symbol/linear feature - name it using the reference characteristic sheets. "
    "Write in plain prose; do not use markdown headers, bullet points, or bold text. "
    "Name features using the reference characteristic sheets; do not describe or explain the conventional sign symbols used to represent them. "
    "Provide detailed, accurate descriptions of the map crop, including locations of features and spatial relationships. "
    "Do not mention any reference materials, documents, or characteristic sheets in your description of the crop. "
    "Do not include any introductory sentence or preamble. "
    "Do not hallucinate features that are not present. "
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an expert in historical Ordnance Survey maps. "
    "You are analysing a 512x512 pixel patch from an Ordnance Survey six-inch to the mile map (approximately 1:10,560 scale, surveyed c.1888-1914). "
    "Two OS characteristic sheets and two reference JSON files (abbreviations and character of writing) are provided before the map patch. "
    "Use them to identify symbols, land cover types, boundaries, linear features, and the meaning of any text or abbreviations visible in the patch. "
    "The final image in this message is the map patch to describe. All preceding images are reference materials only — do not describe the reference sheets. "
    "Name all visible features that are present on the map using the reference materials to guide you. "
    "If it is a symbol/linear feature - name it using the reference characteristic sheets. "
    "If it is text - provide the original text and expand any abbreviations using the reference JSON and identify the character of writing (font style) using the reference JSON. "
    "Write in plain prose; do not use markdown headers, bullet points, or bold text. "
    "Name features using the reference characteristic sheets; do not describe or explain the conventional sign symbols used to represent them. "
    "Provide detailed, accurate descriptions of the map patch, including locations of features and spatial relationships. "
    "Do not mention any reference materials, documents, or characteristic sheets in your description of the patch. "
    "Do not include any introductory sentence or preamble. "
    "Do not hallucinate features that are not present. "
)

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def crop_to_quadrants(image: Image.Image) -> dict[str, Image.Image]:
    """Split a 512x512 image into four named 256x256 quadrant crops."""
    w, h = image.size
    hw, hh = w // 2, h // 2
    return {
        "NW": image.crop((0, 0, hw, hh)),
        "NE": image.crop((hw, 0, w, hh)),
        "SW": image.crop((0, hh, hw, h)),
        "SE": image.crop((hw, hh, w, h)),
    }


def _pil_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(image: Image.Image) -> dict:
    return {"type": "image_url", "image_url": {"url": _pil_to_data_uri(image)}}


def _txt(text: str) -> dict:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_reference_materials() -> tuple[list[Image.Image], str, str]:
    sheets = []
    for path in (_SHEET_PRIMARY, _SHEET_SECONDARY):
        if not path.exists():
            raise FileNotFoundError(f"Characteristic sheet not found: {path}")
        sheets.append(Image.open(path).convert("RGB"))
    return sheets, _ABBREV_JSON.read_text(), _WRITING_1914_JSON.read_text()


def load_llm(
    model_name: str, tensor_parallel_size: int, pipeline_parallel_size: int
) -> LLM:
    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="ray",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 4},
        trust_remote_code=True,
        dtype="bfloat16",
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _ref_content_quad(ref_sheets: list[Image.Image]) -> list[dict]:
    return [
        _img(ref_sheets[0]),
        _txt("Conventional signs for the six-inch series (Plate IV, 1923)."),
        _img(ref_sheets[1]),
        _txt("Characteristic sheet for the engraved six-inch maps (1897)."),
    ]


def _ref_content_synthesis(
    ref_sheets: list[Image.Image], abbrev_text: str, writing_text: str
) -> list[dict]:
    return [
        *_ref_content_quad(ref_sheets),
        _txt(f"Abbreviations used on this map series:\n{abbrev_text}"),
        _txt(f"Character of writing conventions:\n{writing_text}"),
    ]


def run_quadrant_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    ref_sheets: list[Image.Image],
    quad_images: list[Image.Image],
) -> list[str]:
    ref = _ref_content_quad(ref_sheets)
    messages_batch = [
        [
            {"role": "system", "content": QUADRANT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    *ref,
                    _img(img),
                    _txt("Describe every visible feature in this map crop."),
                ],
            },
        ]
        for img in quad_images
    ]
    outputs = llm.chat(messages_batch, sampling_params=sampling_params)
    return [o.outputs[0].text for o in outputs]


def build_synthesis_prompt(source_a_caption: str, quad_descs: dict[str, str]) -> str:
    grounding = source_a_caption[0].lower() + source_a_caption[1:]
    quad_lines = "\n".join(f"{q} quadrant: {quad_descs[q]}" for q in QUAD_NAMES)
    return (
        f"From the map text and known symbol detections: {grounding}\n\n"
        f"Detailed quadrant observations:\n{quad_lines}\n\n"
        "Using the quadrant observations and the full patch image above, write a single "
        "integrated description of all visible features with their locations in the patch. "
        "Begin immediately with the first feature."
    )


def run_synthesis_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    ref_sheets: list[Image.Image],
    abbrev_text: str,
    writing_text: str,
    images: list[Image.Image],
    prompts: list[str],
) -> list[str]:
    ref = _ref_content_synthesis(ref_sheets, abbrev_text, writing_text)
    messages_batch = [
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": [*ref, _img(img), _txt(prompt)]},
        ]
        for img, prompt in zip(images, prompts, strict=True)
    ]
    outputs = llm.chat(messages_batch, sampling_params=sampling_params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Inference loop (rank 0 only)
# ---------------------------------------------------------------------------


def _run_inference(args: argparse.Namespace) -> None:
    captions_path = Path(args.captions)
    patches_dir = Path(args.patches_dir)
    output_path = (
        Path(args.output)
        if args.output
        else captions_path.parent / "vlm_captions.jsonl"
    )
    intermediate_path = (
        Path(args.intermediate)
        if args.intermediate
        else output_path.with_name(output_path.stem + "_quadrants.jsonl")
    )

    source_a: dict[str, dict] = {}
    with open(captions_path) as f:
        for line in f:
            rec = json.loads(line)
            source_a[rec["patch_id"]] = rec

    done: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                done.add(json.loads(line)["patch_id"])
        print(f"Resuming — {len(done):,} final captions already written.")

    quad_done: dict[str, dict[str, str]] = {}
    if intermediate_path.exists():
        with open(intermediate_path) as f:
            for line in f:
                rec = json.loads(line)
                quad_done[rec["patch_id"]] = rec["quadrant_captions"]
        print(f"  {len(quad_done):,} quadrant description sets already cached.")

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

    print("Loading characteristic sheets and reference JSON ...")
    ref_sheets, abbrev_text, writing_text = load_reference_materials()

    pp_size = args.pipeline_parallel_size or int(os.environ.get("SLURM_NNODES", "1"))
    print(f"Loading {args.model} (tp={args.tensor_parallel_size}, pp={pp_size}) ...")
    llm = load_llm(args.model, args.tensor_parallel_size, pp_size)

    quad_params = SamplingParams(
        max_tokens=args.max_new_tokens, repetition_penalty=1.1, temperature=0.0
    )
    synth_params = SamplingParams(
        max_tokens=args.max_new_tokens * 2, repetition_penalty=1.1, temperature=0.0
    )

    n_written = 0
    with open(output_path, "a") as fout, open(intermediate_path, "a") as f_int:
        for i in tqdm(range(0, len(work), args.batch_size), desc="Batches"):
            batch_recs = work[i : i + args.batch_size]
            images = {
                rec["patch_id"]: Image.open(patches_dir / rec["patch_id"]).convert(
                    "RGB"
                )
                for rec in batch_recs
            }

            # --- Stage 1: quadrant descriptions ---
            needs_quads = [
                rec for rec in batch_recs if rec["patch_id"] not in quad_done
            ]
            if needs_quads:
                nq_images = [images[rec["patch_id"]] for rec in needs_quads]
                nq_crops = [crop_to_quadrants(img) for img in nq_images]
                for q in QUAD_NAMES:
                    q_imgs = [crops[q] for crops in nq_crops]
                    descs = run_quadrant_batch(llm, quad_params, ref_sheets, q_imgs)
                    for rec, desc in zip(needs_quads, descs, strict=True):
                        quad_done.setdefault(rec["patch_id"], {})[q] = desc

                for rec in needs_quads:
                    f_int.write(
                        json.dumps(
                            {
                                "patch_id": rec["patch_id"],
                                "parent_id": rec["parent_id"],
                                "quadrant_captions": quad_done[rec["patch_id"]],
                            }
                        )
                        + "\n"
                    )
                f_int.flush()

            # --- Stage 2: synthesis ---
            batch_images = [images[rec["patch_id"]] for rec in batch_recs]
            batch_prompts = [
                build_synthesis_prompt(rec["caption"], quad_done[rec["patch_id"]])
                for rec in batch_recs
            ]
            captions = run_synthesis_batch(
                llm,
                synth_params,
                ref_sheets,
                abbrev_text,
                writing_text,
                batch_images,
                batch_prompts,
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
    print(f"Quadrant descriptions saved → {intermediate_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Source B VLM captions (quadrant → synthesis, multi-node vLLM)"
    )
    parser.add_argument("--captions", required=True, help="Source A captions JSONL")
    parser.add_argument(
        "--patches-dir", required=True, help="Directory containing patch PNG files"
    )
    parser.add_argument(
        "--output", help="Output JSONL (default: <captions_dir>/vlm_captions.jsonl)"
    )
    parser.add_argument("--intermediate", help="Quadrant captions sidecar JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=4, help="GPUs per node (default: 4)"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Number of nodes (default: SLURM_NNODES)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Checkpoint batch size (default: 16)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per quadrant; synthesis uses 2x (default: 512)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    _run_inference(args)


if __name__ == "__main__":
    main()
