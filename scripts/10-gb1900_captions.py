"""Generate spatially-detailed captions from GB1900-aligned patch annotations.

For each patch in the JSONL produced by 8-align_gb1900.py, derives a natural-language
caption describing named features and their positions within the tile.

Spatial positions are computed geometrically from tile_x/tile_y coordinates.
Abbreviations are expanded using the OS map convention lookup table.

Output JSONL format:
  {"patch_id": "...", "parent_id": "...", "caption": "...", "n_annotations": N}

Usage:
  uv run python scripts/10-gb1900_captions.py \\
      --input data/patches_6inch_2nd_ed/gb1900_annotations.jsonl \\
      --output data/patches_6inch_2nd_ed/captions.jsonl
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# OS map abbreviation table — loaded from OS 1914 characteristic sheet (O.S. 404)
# ---------------------------------------------------------------------------

_ABBREV_JSON = (
    Path(__file__).parent.parent / "data/characteristic_sheets/abbreviations.json"
)
ABBREV: dict[str, str] = json.loads(_ABBREV_JSON.read_text())

# Compile longest-match regex for abbreviation expansion
_ABBREV_SORTED = sorted(ABBREV.keys(), key=len, reverse=True)
_ABBREV_PAT = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ABBREV_SORTED) + r")\b"
)


def expand_abbreviations(text: str) -> str:
    def _replace(m: re.Match) -> str:
        expansion = ABBREV[m.group(0)]
        return m.group(0) if "," in expansion else expansion

    return _ABBREV_PAT.sub(_replace, text)


# ---------------------------------------------------------------------------
# Spatial position helpers
# ---------------------------------------------------------------------------

TILE_SIZE = 512
CENTRE_LO = TILE_SIZE // 3  # 170
CENTRE_HI = TILE_SIZE * 2 // 3  # 341
EDGE_MARGIN = 64


def quadrant(x: int, y: int) -> str:
    """Return the named region of a 512x512 tile for a given pixel position."""
    near_edge = []
    if y < EDGE_MARGIN:
        near_edge.append("northern")
    elif y > TILE_SIZE - EDGE_MARGIN:
        near_edge.append("southern")
    if x < EDGE_MARGIN:
        near_edge.append("western")
    elif x > TILE_SIZE - EDGE_MARGIN:
        near_edge.append("eastern")
    if len(near_edge) == 2:
        ns, ew = near_edge[0][0].upper(), near_edge[1][0].upper()
        return f"near the {ns}{ew} corner"
    if near_edge:
        return f"near the {near_edge[0]} edge"

    in_centre_x = CENTRE_LO <= x <= CENTRE_HI
    in_centre_y = CENTRE_LO <= y <= CENTRE_HI
    if in_centre_x and in_centre_y:
        return "near the centre"

    ns = "northern" if y < TILE_SIZE // 2 else "southern"
    ew = "western" if x < TILE_SIZE // 2 else "eastern"

    if in_centre_x:
        return f"in the {ns} half"
    if in_centre_y:
        return f"in the {ew} half"
    return f"in the {ns[:1].upper()}{ew[:1].upper()} quadrant"  # NW, NE, SW, SE


def compass_direction(x1: int, y1: int, x2: int, y2: int) -> str:
    """Cardinal/intercardinal direction from point 1 to point 2."""
    dx = x2 - x1
    dy = y2 - y1  # positive dy = south (y increases downward)
    angle = (90 - math.degrees(math.atan2(-dy, dx))) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(angle / 45) % 8]


def proximity_label(x1: int, y1: int, x2: int, y2: int) -> str:
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if dist < 40:
        return "immediately adjacent to"
    if dist < 100:
        return "close to"
    if dist < 200:
        return "near"
    return None


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

ORDINALS = ["", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]


def _count_prefix(n: int) -> str:
    return ORDINALS[n] if n < len(ORDINALS) else str(n)


def generate_caption(annotations: list[dict]) -> str:
    """Build a spatially detailed caption from a list of GB1900 annotations."""
    if not annotations:
        return ""

    # Expand abbreviations and assign spatial positions
    entries = []
    for ann in annotations:
        raw = ann["text"].strip()
        # Skip noise: single non-alpha characters and pure digit strings
        if not raw or (len(raw) == 1 and not raw.isalpha()) or raw.isdigit():
            continue
        expanded = expand_abbreviations(raw)
        # Sentence-case labels that GB1900 transcribed in lowercase
        expanded = expanded[0].upper() + expanded[1:] if expanded else expanded
        pos = quadrant(ann["tile_x"], ann["tile_y"])
        entries.append(
            {
                "text": expanded,
                "raw": raw,
                "pos": pos,
                "x": ann["tile_x"],
                "y": ann["tile_y"],
            }
        )

    # Build sentences, sorted by position (NW → N → NE → W → centre → E → SW → S → SE → edges)
    position_order = [
        "in the NW quadrant",
        "in the northern half",
        "in the NE quadrant",
        "in the western half",
        "near the centre",
        "in the eastern half",
        "in the SW quadrant",
        "in the southern half",
        "in the SE quadrant",
        "near the northern edge",
        "near the southern edge",
        "near the western edge",
        "near the eastern edge",
        "near the NW corner",
        "near the NE corner",
        "near the SW corner",
        "near the SE corner",
    ]

    def pos_sort_key(pos: str) -> int:
        try:
            return position_order.index(pos)
        except ValueError:
            return len(position_order)

    # Group all entries by text to detect labels appearing across multiple positions
    by_text: dict[str, list] = defaultdict(list)
    for e in entries:
        by_text[e["text"]].append(e)

    # Build clauses: "<label> <position>" or "<count> instances of <label>"
    clauses = []
    emitted_texts: set[str] = set()
    for pos in sorted({e["pos"] for e in entries}, key=pos_sort_key):
        pos_entries = [e for e in entries if e["pos"] == pos]
        text_groups: dict[str, list] = defaultdict(list)
        for e in pos_entries:
            text_groups[e["text"]].append(e)
        for text, _group in text_groups.items():
            if text in emitted_texts:
                continue
            all_for_text = by_text[text]
            all_positions = sorted({e["pos"] for e in all_for_text}, key=pos_sort_key)
            total = len(all_for_text)
            count = _count_prefix(total).lower()
            if len(all_positions) == 1:
                if total == 1:
                    clauses.append(f"{text} {all_positions[0]}")
                else:
                    clauses.append(f"{count} instances of {text} {all_positions[0]}")
            else:
                clauses.append(f"{count} instances of {text}")
            emitted_texts.add(text)

    sentences = []
    if clauses:
        if len(clauses) == 1:
            sentences.append(f"The map shows {clauses[0]}.")
        elif len(clauses) == 2:
            sentences.append(f"The map shows {clauses[0]} and {clauses[1]}.")
        else:
            sentences.append(
                f"The map shows {', '.join(clauses[:-1])}, and {clauses[-1]}."
            )

    # Add relative-proximity sentences for spatially close distinct features
    if len(entries) >= 2:
        proximity_added = set()
        for i, e1 in enumerate(entries):
            for e2 in entries[i + 1 :]:
                if e1["text"] == e2["text"]:
                    continue
                label = proximity_label(e1["x"], e1["y"], e2["x"], e2["y"])
                if label:
                    pair = tuple(sorted([e1["text"], e2["text"]]))
                    if pair not in proximity_added:
                        sentences.append(f"{e1['text']} is {label} {e2['text']}.")
                        proximity_added.add(pair)
                        if len(proximity_added) >= 3:
                            break
            if len(proximity_added) >= 3:
                break

    return " ".join(sentences)


_VLM_PROMPT_TEMPLATE = (
    "This is a patch from an Ordnance Survey 6-inch to the mile map. "
    "From the map text, we see {caption} "
    "Your task is to describe all visible cartographic features in this tile, including symbols, "
    "land use, boundaries, vegetation, and any other details you can identify. "
    "Be as specific and detailed as possible."
)


def build_vlm_prompt(caption: str) -> str:
    caption = caption[0].lower() + caption[1:]
    return _VLM_PROMPT_TEMPLATE.format(caption=caption)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions from GB1900-aligned patch annotations"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL from 8-align_gb1900.py",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: <input_dir>/captions.jsonl)",
    )
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=1,
        help="Skip patches with fewer than this many annotations (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else input_path.parent / "captions.jsonl"
    )

    n_written = 0
    n_skipped = 0
    total_annotations = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            annotations = record["annotations"]
            if len(annotations) < args.min_annotations:
                n_skipped += 1
                continue

            caption = generate_caption(annotations)
            if not caption:
                n_skipped += 1
                continue

            fout.write(
                json.dumps(
                    {
                        "patch_id": record["patch_id"],
                        "parent_id": record["parent_id"],
                        "caption": caption,
                        "vlm_prompt": build_vlm_prompt(caption),
                        "n_annotations": len(annotations),
                    }
                )
                + "\n"
            )
            n_written += 1
            total_annotations += len(annotations)

    print(f"Written: {n_written:,} captions → {output_path}")
    print(f"Skipped: {n_skipped:,} patches (below min-annotations threshold)")
    if n_written:
        print(f"Mean annotations per caption: {total_annotations / n_written:.1f}")


if __name__ == "__main__":
    main()
