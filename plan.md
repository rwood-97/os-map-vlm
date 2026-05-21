# OS Map VLM — Project Plan

## Context and Motivation

This project builds a Vision-Language Model (VLM) for historical Ordnance Survey maps, covering multiple series (town plans ~1:500–1:1,056; 25-inch 1:2,500; 6-inch 1:10,560) and multiple editions. Development starts with the OS 6-inch 2nd edition (c.1888–1914) as the best-annotated baseline, then extends to other series and editions. The primary goals are:

1. **Skill development** — learn vision encoder pretraining (MAE) and multimodal alignment from hands-on experimentation, building on existing LLM pretraining experience (OLMo) and familiarity with MapReader and T0
2. **Community contribution** — release a useful open model and encoder that DH researchers without ML capacity can use to query historical OS maps in natural language

This is not a research paper. Success means: a working model on HuggingFace that a historian can use from a Jupyter notebook.

---

## Constraints

- **Time**: 10% FTE + 4–5 evenings, 2 months
- **Compute**: 10,000 GPUh remaining on Isambard-AI (GH200s), separate from ongoing OLMo allocation
- **Annotation**: No capacity for human annotation — all supervision must come from existing datasets or automatic alignment
- **Novelty bar**: Not required — applied and useful is the goal

---

## What Makes This Different from Repeating Molmo

- **Domain-specific MAE encoder**: train a ViT on historical map tiles rather than using CLIP (which has never seen OS cartography)
- **Zero-annotation pipeline**: demonstrate what is achievable using only existing crowdsourced and georeferenced data (GB1900 + NLS + MapReader + OSM)

---

## Data Assets

### Primary

Three OS series, all available via NLS. Multiple editions are in scope — they multiply training data and enable change-detection tasks.

| Series | Scale | Editions / period | Sheets (approx) | Licence | Notes |
|---|---|---|---|---|---|
| OS Town Plans | ~1:500–1:1,056 | c.1840s–1880s; mostly one-off surveys | ~2,000 | NLS terms | Major towns/cities; very dense features; fewer sheets but highest detail |
| OS 25-inch (County Series) | 1:2,500 | Multiple revisions c.1870s–1940s | ~100K total; NLS subset TBC | NLS terms | Covers cultivated areas; field parcels with acreages, individual buildings |
| OS 6-inch | 1:10,560 | 1st ed. c.1843–1882; 2nd ed. c.1888–1914; later revisions | ~16K per edition | NLS terms | 2nd edition best covered by MapReader and GB1900; 1st edition has different cartographic conventions |

**Edition as metadata**: cartographic conventions differ between editions (symbology, hachuring vs contours, etc.), so edition and approximate survey date should be passed as text tokens to the LLM alongside scale (e.g. `[1:10560] [6-inch 2nd ed. c.1900]`). If the edition is unknown, use `[edition unknown]`. All inputs are single-tile; change detection is out of scope for v0.1.

**Start with 2nd edition 6-inch**: this is the best-covered series (MapReader, GB1900) and the right baseline. Add other editions and series once the pipeline is working.

| Dataset | Content | Size | Licence | Notes |
|---|---|---|---|---|
| GB1900 | 2.55M georeferenced text strings (lat/lon + transcription) from OS 6-inch c.1900 | 2.55M entries | CC0 | Primary annotation source; 6-inch series |
| London 1890s OS Text Layer (Zou et al., EPFL) | 285,846 georeferenced text sequences from 729 OS town plan sheets (1:1,056) covering Greater London, 1891–1896 | 285,846 entries | CC BY 4.0 | On Zenodo (record 14982947); GeoJSON format; London only — supplements OSM alignment for town plan tiles |
| MapReader SIGSPATIAL 2022 | ~62K human-annotated patches (railspace, building); inferred labels for 30.5M patches | 30.5M patches | Open | On Zenodo + HuggingFace; based on 6-inch |
| MapReader extended datasets | Railways, buildings, trees, heather, dew ponds, other features | TBC | TBC | 6-inch and 25-inch series, only South Downs and Peak District National parks |

### Secondary (automatic alignment)

| Dataset | Use | Notes |
|---|---|---|
| OS OpenData / OSM | Modern vector layers for georeferenced feature masks (roads, water, buildings) | Alignment noisy for changed features; reliable for stable ones (rivers, coastlines, woodland, marshes) |
| OS Characteristic Sheets | Conventional signs reference sheets listing all symbols and their meanings per series/edition | NLS likely holds these; used to build symbol vocabulary for targeted GPT-4V prompting (Source C) — covers unlabelled symbols that GB1900 and MapReader cannot |

---

## Minimum Data Requirements

Each OS sheet is ~9,600×7,200 px at 400 DPI. At 512×512 tiles (no overlap): ~252 tiles/sheet.

### Per-series breakdown

| Series | Scale | Ground area per 512px tile | Sheets available | Tiles @ min viable | GB1900 coverage |
|---|---|---|---|---|---|
| Town plans | ~1:500–1:1,056 | ~16m–34m × 16m–34m | ~14,600 (12,724 E&W + 1,872 Scotland) | ~2,000 | Partial — London only (Zou et al.); OSM/MapReader alignment elsewhere |
| 25-inch | 1:2,500 | ~81m × 81m | ~122K main editions (E&W + Scotland) | ~2,000 | None — use OSM/MapReader alignment |
| 6-inch | 1:10,560 | ~344m × 344m | ~20K 2nd edition (Scotland: 7,486; E&W: TBC by edition) | ~2,000 | Strong — GB1900 covers this series (1888–1913) |

### 6-inch starting-point table

| Sheets | Tiles (no overlap) | Download (~20 MB/sheet) | Use |
|--------|-------------------|------------------------|-----|
| 500 | ~126K | ~10 GB | Smoke test only |
| 2,000 | ~500K | ~40 GB | **Minimum viable** — expect to beat MapReader ResNet |
| 5,000 | ~1.25M | ~100 GB | Comfortable baseline, sufficient for ablations |
| 16,000 | ~4.0M | ~320 GB | Full scale |

**Recommended approach**: start with 2,000 sheets per series. The MAE encoder sees 512×512 raster tiles regardless of source scale; training across all three series teaches it representations of OS cartography at different levels of detail, which increases its utility for DH researchers working with different collections.

Domain-specific MAE (satellite, medical, document imagery) consistently reaches competitive encoder quality at 200K–500K images. The MapReader ResNet baseline was trained on 62K *supervised* patches; a ViT-B MAE-pretrained on ~1.5M tiles across all three series should comfortably match or exceed it.

**Scale metadata in instruction tuning**: because a tile's geographic footprint varies by series, include the scale as a text token in all VLM prompts (e.g. `[1:2500]`). This lets the model reason correctly about distances and feature sizes without any architectural changes.

**GB1900 gap for 25-inch and town plans**: GB1900 covers 6-inch. For the other two series, MapReader data and OSM alignment (roads, water, buildings from OS OpenData) provides the primary automatic annotation signal for instruction tuning.

---

## Architecture

A standard three-component VLM:

```
[Map ViT Encoder]  +  [Connector MLP]  +  [OLMo 3 7B]
      |                      |                   |
 Pretrained on          Trained from         Existing OLMo
 OS map tiles           scratch               checkpoint
 via MAE                                      (frozen or
                                              lightly tuned)
```

### Vision Encoder
- Architecture: ViT-B (86M params) — small enough for fast ablations, large enough to be useful
- Training: Masked Autoencoders (MAE) on OS map tiles across all three series (town plans, 25-inch, 6-inch)
- Input: 512×512 pixel tiles, 16×16 pixel patches → 1024 patches per tile
- Masking ratio: 75% (MAE default; ablate this)
- Scale is not encoded in the image — the encoder sees raster tiles regardless of source series. Scale metadata is passed as a text token to the LLM at inference time (`[1:500]`, `[1:2500]`, `[1:10560]`).
- **Feature-aware masking** (the novel contribution): instead of random patch masking, mask patches selected by feature type using MapReader labels and GB1900 text coordinates

### Connector
- Architecture: 2-layer MLP
- Input: ViT-B patch embeddings (dim 768)
- Output: projected to OLMo token dimension (dim 4096)
- Training: frozen encoder + frozen LLM, only connector trained

### Language Model
- OLMo 3 7B (existing checkpoint from current allocation)
- Frozen during connector training
- Lightly fine-tuned (top layers only) during instruction tuning

---

## MAE Pretraining

Standard random masking at 75%. For each tile, patches are selected uniformly at random to mask — the encoder sees the remaining 25% and must reconstruct the masked pixels.

Training a single encoder to convergence (~20 epochs on ~500K–5M tiles, depending on how many sheets are downloaded): **400–500 GPUh** at full scale; proportionally less for the 2,000-sheet starting point (~50–100 GPUh).

### Evaluation (free — uses existing MapReader infrastructure)
Freeze encoder → attach linear probe → train on 62K MapReader annotated patches → evaluate on held-out set. Compare against existing MapReader ResNet/EfficientNet baselines. Decision gate: match or exceed MapReader ResNet before proceeding to Stage 2.

---

## Training Stages

### Stage 0: Data pipeline (Month 1, engineering)
No GPU required. Can be done at 10% time alongside other work.

- [ ] Download NLS georeferenced OS 6-inch tiles (or confirm MapReader pipeline access)
- [ ] Download GB1900 dataset from NLS Data Foundry
- [ ] Write georeferencing alignment: GB1900 lat/lon → tile → pixel → patch index
- [ ] Build tile index with GB1900 annotations per tile
- [ ] Confirm access to MapReader extended feature datasets (national parks, trees, etc.)
- [ ] Download MapReader SIGSPATIAL 2022 inferred predictions (30.5M patches) from Zenodo
- [ ] Write webdataset-format dataloader for MAE training (must stream; tiles won't fit in RAM)
- [ ] Write feature-aware masking logic (patch selector by label type)
- [ ] Test: submit a small 1-node training job, confirm it runs

Deliverable: can submit MAE training jobs and have them run unattended.

### Stage 1: MAE encoder pretraining (Month 1–2)
Single run, trains to convergence (~20 epochs on ~500K tiles from 2,000 6-inch sheets). Unattended — submit and come back.

- [ ] Run MAE with random 75% masking
- [ ] Evaluate against MapReader ResNet/EfficientNet baselines
- [ ] Confirm encoder quality before proceeding to Stage 2

Deliverable: trained ViT-B encoder. **Release on HuggingFace before the VLM is built.** It has standalone value for MapReader users.

### Stage 2: Caption pretraining — connector training (Month 2)

Generate caption dataset from existing sources (no human annotation):

**Source A — GB1900 tile descriptions (automatic)**
For each tile, aggregate all GB1900 entries and generate a structured description:
```
"This map tile contains: Swinton Farm, St Mary's Church (Ch),
 2 Public Houses (PH), a Footbridge (FB), River Wharfe"
```
Use OLMo/any LLM to expand abbreviations once, save the mapping, apply at scale. This is using an LLM to expand ground-truthed labels — not to hallucinate visual content.

Note: GB1900 records text position, not symbol position — labels and symbols are typically within a few pixels of each other on OS maps, so boundary cases are acceptable noise for caption generation. For pixel-precision instruction tuning tasks (Stage 3), exclude GB1900 entries within 32px of a tile edge to avoid ambiguous boundary cases.

**Source B — OSM alignment descriptions (automatic)**
For georeferenced tiles, query OSM for stable features in the bounding box and generate natural descriptions ("this tile covers agricultural land with a river running NE-SW and a settlement of approximately 40 buildings").

**Source C — GPT-4V synthetic captions (targeted symbol coverage)**
For a sample of tiles, use GPT-4V to generate descriptions targeted at unlabelled visual symbols (marshes, orchards, rough pasture, cliff hachures, footpaths, parish boundaries, etc.) — features that GB1900 and MapReader cannot cover. Use OS characteristic sheets to build a per-series symbol vocabulary and include it in the GPT-4V prompt, so captions specifically describe visible symbols rather than giving generic descriptions. Be explicit in model card that this is used only for caption pretraining, not instruction tuning. Document the fraction of training data this represents.

Training:
- Freeze encoder (best from Stage 1)
- Freeze OLMo 3 7B
- Train connector MLP only
- Loss: next-token prediction on caption text given image tokens prepended
- Duration: ~200–400 GPUh

Deliverable: connector checkpoint; model can generate basic tile descriptions.

### Stage 3: Instruction tuning (Month 2–3)

Generate instruction dataset from GB1900 + MapReader labels (no human annotation):

**Text grounding tasks (GB1900)**
```
Q: "Point to where it says 'Windmill' on this map"
A: [pixel coordinates of all Windmill entries in tile]

Q: "What is written in this region?" [bounding box]
A: [GB1900 transcription for that location]
```

**Feature counting tasks (GB1900 aggregated)**
```
Q: "How many Public Houses are marked on this tile?"
A: [count of PH entries in tile bounds]

Q: "How many churches are in this area?"
A: [count of Ch/Church entries]
```

**Named entity location (GB1900)**
```
Q: "Where is [Farm Name] on this map?"
A: [pixel coordinates of that GB1900 entry]
```

**Spatial relationship tasks (GB1900 derived)**
```
Q: "What is the nearest Public House to St Mary's Church?"
A: [derived from relative pixel coordinates — no additional annotation needed]
```

**Feature presence tasks (MapReader labels)**
```
Q: "Does this map tile contain any railway infrastructure?"
A: "Yes" / "No" [from MapReader railspace label]

Q: "Describe the built environment in this tile"
A: [derived from building density and railspace labels]
```

Training:
- Unfreeze top layers of OLMo
- Unfreeze encoder (lower learning rate)
- Train end-to-end
- Duration: ~300–500 GPUh

Deliverable: instruction-tuned model capable of text spotting, feature counting, named entity location, spatial queries.

---

## Compute Budget Summary

| Stage | Activity | GPUh estimate |
|---|---|---|
| Stage 1 | MAE encoder pretraining | 400–500 |
| Stage 2 | Connector training | 200–400 |
| Stage 3 | Instruction fine-tuning | 300–500 |
| Buffer | Failed runs, debugging, reruns | 500–1,000 |
| **Total** | | **~1,400–2,400** |

Remaining headroom: ~7,600–8,600 GPUh. Consider scaling to ViT-L (307M params, ~4× compute) only if ViT-B evaluation shows it clearly plateaus below MapReader baselines — not as a default next step. Additional ablations (feature-aware masking, more sheets, more series/editions) are likely better uses of headroom first.

---

## Timeline

### Weeks 1–3: Data pipeline (Stage 0)
Concentrated push — aim to have the pipeline working and first job submitted by end of week 3. Use evenings here if needed; this is the highest-risk phase.

- Data pipeline engineering
- GB1900 georeferencing alignment
- Submit MAE run (unattended from here)

### Weeks 3–5: MAE training + caption dataset generation
MAE trains unattended. Use this window to generate caption and instruction datasets in parallel — both are scripted, low-effort work that fits well at 10% FTE.

- Caption dataset generation (GB1900 aggregation + OSM queries)
- Instruction dataset generation (GB1900 QA pairs + MapReader labels)

### Week 5–6: Evaluate encoder + release
- Evaluate encoder against MapReader baselines
- Release encoder on HuggingFace (v0)
- Submit connector training (unattended)

### Weeks 6–7: Connector training + instruction fine-tuning
- Connector training completes → submit instruction fine-tuning (unattended)

### Week 8: Release
- Instruction fine-tuning completes
- Release full VLM on HuggingFace (v0.1)
- Write model card aimed at DH researchers (not ML engineers)

---

## Honest Capability Expectations

### The model will be good at:
- Reading and locating text labels (GB1900 training signal is strong)
- Counting named feature types (PH, Ch, FB etc.)
- Answering "where is X" for named places and features
- Basic tile description (settlement, agricultural land, coastal etc.)
- Railspace and building presence/absence (MapReader training signal is strong)

### The model will be weaker at:
- Visual symbol interpretation without adjacent text labels — partially mitigated by OSM alignment (stable features: woodland, marsh, water) and targeted GPT-4V captions using OS characteristic sheets as a symbol vocabulary reference; unlabelled symbols remain weaker than labelled ones
- Features not covered by GB1900 or MapReader datasets
- Series or editions not well-represented in training data
- Heavily degraded or unusual map scans

### What this pipeline cannot claim:
- Full provenance (GPT-4V used for some caption data — document this clearly)
- State-of-the-art performance vs general VLMs on standard benchmarks
- Equal quality across all series and editions — 6-inch 2nd edition will be strongest

---

## Open Questions to Resolve Early

**NLS tile access** — confirm the most efficient way to download tiles at scale (IIIF API, bulk download, or reuse existing MapReader pipeline). Avoid re-downloading what already exists from MapReader work.

---

## Key References and Resources

- MapReader paper + code: https://github.com/maps-as-data/MapReader
- MapReader SIGSPATIAL 2022 dataset: https://zenodo.org/record/7147906
- GB1900 gazetteer: https://data.nls.uk/data/map-spatial-data/gb1900/
- NLS georeferenced maps viewer: https://maps.nls.uk
- MAE paper (He et al. 2021): https://arxiv.org/abs/2111.06377
- Molmo paper (for VLM architecture reference): https://arxiv.org/abs/2409.17146
- OLMo 3: https://allenai.org/papers/olmo3
