# OS Map VLM — Project Plan

## Context and Motivation

This project builds a Vision-Language Model (VLM) for historical Ordnance Survey maps, covering multiple series (town plans ~1:500–1:1,056; 25-inch 1:2,500; 6-inch 1:10,560) and multiple editions.

The primary goals are:

1. **Skill development** — learn vision encoder pretraining (MAE) and multimodal alignment from hands-on experimentation
2. **Community contribution** — release a useful open model and encoder that DH researchers without ML capacity can use to query historical OS maps in natural language
3. **Font detection** — a distinctive capability that general VLMs lack: recognising OS typographic conventions (italic = water features, gothic = antiquities, spaced caps = settlements) to identify feature types from text style alone. DH scholars have identified this as a high-value target — it enables automated feature extraction from map tiles without relying on adjacent symbols or modern reference data.

This is not a research paper. Success means: a working model on HuggingFace.

---

## Constraints

- **Time**: ~10% FTE over 10 weeks (2026-05-27 → 2026-08-03)
- **Compute**: 10,000 GPUh remaining on Isambard-AI (GH200s)
- **Annotation**: No capacity for human annotation — all supervision must come from existing datasets or automatic alignment
- **Novelty bar**: Not required — applied and useful is the goal

---

## Data Assets

### Primary

Three OS series, all downloaded via NLS or MapTiler Cloud using MapReader. MapReader downloads from XYZ tiles as a full-sheet PNG with georeferenced metadata, then `patchify` splits them into 512×512 pixel patches. The can be PNG or GeoTIFF. Patches include per-patch lat/lon bounding boxes, which enables GB1900 alignment directly.

| Series | Scale | Editions / period | Sheets (approx) | Source | Notes |
|---|---|---|---|---|---|
| OS Town Plans | ~1:500–1:1,056 | c.1840s–1880s; mostly one-off surveys | ~2,800 (~992 at 1:1,056 + ~1,820 at 1:500, 4 sheets/town across 455 towns); actual sheet size ~11,000×7,500 px (~294 tiles/sheet) | NLS | Major towns/cities; very dense features; highest detail |
| OS 25-inch (County Series) | 1:2,500 | Multiple revisions c.1870s–1940s | ~4,000 (400/band across 10 latitude bands); actual sheet size ~6,700×4,600 px (~117 tiles/sheet) | NLS | Covers cultivated areas; field parcels with acreages, individual buildings |
| OS 6-inch 2nd ed. | 1:10,560 | c.1888–1914 | ~4,000 (from internal filestore); actual sheet size ~7,168×4,864 px (~126 tiles/sheet) | MapTiler (`uk-osgb10k1888`) | Best covered by MapReader and GB1900 |
| OS 6-inch 1st ed. | 1:10,560 | c.1843–1882 | ~4,000 (from internal filestore); actual sheet size ~6,656×4,608 px (~117 tiles/sheet) | NLS | Different cartographic conventions from 2nd ed.; no GB1900 coverage |

**Download approach**: write one MapReader download script per series/edition. MapReader handles georeferencing and outputs `parent_df` (sheet-level metadata including `published_date`, `coordinates`, `crs`) and `patch_df` (per-patch lat/lon bounds). Patchify at 512×512 pixels — consistent input size for MAE regardless of series. MapReader can also export patches as GeoJSON if needed for alignment tasks.

**Edition and date metadata**: cartographic conventions differ between editions (symbology, hachuring vs contours, etc.), so edition, survey date, and publication date should all be passed as text tokens to the LLM alongside scale. Survey date and publication date can differ significantly (a sheet surveyed c.1875 may be published c.1890), and both carry meaning — survey date reflects what was on the ground; publication date reflects when conventions were standardised. MapReader's `parent_df` provides `published_date`; survey date is often recorded in sheet metadata or can be approximated from edition ranges. Format: `[1:10560] [6-inch 2nd ed.] [surveyed: c.1895] [published: c.1900]`. If survey date is unknown, use `[surveyed: unknown]`; if publication date is unknown, use `[published: unknown]`. All inputs are single-tile; change detection is out of scope for v0.1.

| Dataset | Content | Size | Licence | Notes |
|---|---|---|---|---|
| GB1900 | 2.55M georeferenced text strings (lat/lon + transcription) from OS 6-inch c.1900 | 2.55M entries | CC0 | Primary annotation source; 6-inch series |
| London 1890s OS Text Layer (Zou et al., EPFL) | 285,846 georeferenced text sequences from 729 OS town plan sheets (1:1,056) covering Greater London, 1891–1896 | 285,846 entries | CC BY 4.0 | On Zenodo (record 14982947); GeoJSON format; London only — primary text annotation source for town plan tiles |
| MapReader SIGSPATIAL 2022 | ~62K human-annotated patches (railspace, building); inferred labels for 30.5M patches | 30.5M patches | Open | On Zenodo + HuggingFace; based on 6-inch |
| MapReader extended datasets | Railways, buildings, trees, heather, dew ponds, other features | TBC | TBC | 6-inch and 25-inch series, only South Downs and Peak District National parks |

### Secondary

| Dataset | Use | Notes |
|---|---|---|
| OS Characteristic Sheets | Conventional signs reference sheets listing all symbols and their meanings per series/edition | NLS likely holds these; used to build symbol vocabulary for targeted VLM prompting (Source C) and as the authoritative source for typographic conventions — which text styles map to which feature categories per series/edition |
| NLS Urban Trees | Georeferenced tree locations extracted from historical OS maps | https://data.nls.uk/data/map-spatial-data/urban-trees/ — annotation source for tree/woodland feature tasks; OS town plans 1:500 series, Edinburgh and Leeds only. |

### OS Typographic Conventions

OS maps use font style as a *semantic signal* — text appearance encodes feature category, not just emphasis. Key conventions for the 6-inch series:

| Style | Feature category | Examples |
|---|---|---|
| Italic | Water features | rivers, streams, pools, canals, bogs |
| Gothic / Old English | Antiquities and ancient earthworks | Roman roads, earthworks, tumuli |
| Spaced Roman caps | Towns and larger settlements | `OXFORD`, `LEEDS` |
| Small Roman caps | Villages, hamlets | `Thornton`, `Little Houghton` |
| Roman | Administrative labels, farm names, most other features | `Manor Farm`, `St Mary's Ch` |
| Condensed / small Roman | Field names, minor features | field acreages, minor paths |

Conventions differ by edition and series — characteristic sheets document the full mapping. These conventions are a **free annotation source**: any GB1900 entry can have its visual text style classified, then cross-referenced with the characteristic sheet to infer feature category. This gives a font-to-feature-type training signal with no human annotation beyond the initial characteristic sheet lookup.

Note: conventions are consistent within a series/edition but differ across editions (e.g. 1st vs 2nd 6-inch use different gothic conventions for antiquities). Edition and date metadata passed as text tokens at inference time (e.g. `[6-inch 2nd ed.] [surveyed: c.1895] [published: c.1900]`) allows the model to apply the correct convention mapping.

---

## Minimum Data Requirements

Each OS sheet is ~9,600×7,200 px at 400 DPI. At 512×512 tiles (no overlap): ~252 tiles/sheet.

### Per-series breakdown

| Series | Scale | Ground area per 512px tile | Sheets | Tiles (est.) | GB1900 coverage |
|---|---|---|---|---|---|
| Town plans | ~1:500–1:1,056 | ~16m–34m × 16m–34m | ~2,800 | ~823K (~294 tiles/sheet) | Partial — London only (Zou et al.); MapReader + local VLM captions elsewhere |
| 25-inch | 1:2,500 | ~81m × 81m | ~4,000 | ~468K (~117 tiles/sheet) | None — MapReader labels + local VLM captions |
| 6-inch 1st ed. | 1:10,560 | ~344m × 344m | ~4,000 | ~468K (~117 tiles/sheet) | None — GB1900 covers 2nd ed. only; local VLM captions |
| 6-inch 2nd ed. | 1:10,560 | ~344m × 344m | ~4,000 | ~504K (~126 tiles/sheet) | Strong — GB1900 covers this series (1888–1913) |
| **Total** | | | **~14,800** | **~2.26M** | |

**Recommended approach**: 4,000 sheets per series/edition, treating each edition as a distinct visual domain for the MAE encoder. This keeps representation balanced across series (~43% 6-inch combined, ~36% town plans, ~21% 25-inch by tile count) and ensures sufficient 6-inch 2nd ed. coverage for GB1900 alignment in instruction tuning.

**Metadata in instruction tuning**: because a tile's geographic footprint varies by series, include scale, edition, survey date, and publication date as text tokens in all VLM prompts (e.g. `[1:2500] [25-inch County Series] [surveyed: c.1893] [published: c.1896]`). This lets the model reason correctly about distances, feature sizes, and cartographic conventions without any architectural changes.

**GB1900 gap for 25-inch and town plans**: GB1900 covers 6-inch. For the other two series, MapReader labels, the Zou et al. London text layer (town plans), Urban trees, and local VLM captions using characteristic sheet symbol vocabularies provide the primary annotation signal for instruction tuning.

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
- Scale is not encoded in the image — the encoder sees raster tiles regardless of source series. Scale, edition, survey date, and publication date are passed as text tokens to the LLM at inference time (e.g. `[1:500] [Town Plans] [surveyed: c.1865] [published: c.1867]`).

### Connector
- Architecture: 2-layer MLP
- Input: ViT-B patch embeddings (dim 768)
- Output: projected to OLMo token dimension (dim 4096)
- Training: frozen encoder + frozen LLM, only connector trained

### Language Model
- OLMo 3 7B
- Frozen during connector training
- Lightly fine-tuned (top layers only) during instruction tuning

---

## MAE Pretraining

Standard random masking at 75%. For each tile, patches are selected uniformly at random to mask — the encoder sees the remaining 25% and must reconstruct the masked pixels.

Training a single encoder to convergence (~20 epochs on ~2.26M tiles): **~225–285 GPUh** estimated.

### Evaluation (free — uses existing MapReader infrastructure)
Freeze encoder → attach linear probe → train on 62K MapReader annotated patches → evaluate on held-out set. Compare against existing MapReader ResNet/EfficientNet baselines. Decision gate: match or exceed MapReader ResNet before proceeding to Stage 2.

---

## Training Stages

### Stage 0: Data pipeline (Month 1, engineering)
No GPU required. Can be done at 10% time alongside other work.

- [x] Write MapReader download scripts for each series (6-inch 1st ed., 6-inch 2nd ed., 25-inch, town plans) — one script per source
- [ ] Run downloads and patchify at 512×512 pixels; confirm `patch_df` lat/lon bounds are correct
- [ ] Download GB1900 dataset from NLS Data Foundry
- [ ] Write GB1900 alignment: point-in-patch lookup using `patch_df` coordinates (lat/lon bounding box per patch)
- [ ] Build tile index with GB1900 annotations per patch
- [ ] Confirm access to MapReader extended feature datasets (national parks, trees, etc.)
- [ ] Download MapReader SIGSPATIAL 2022 inferred predictions (30.5M patches) from Zenodo
- [ ] Write webdataset-format dataloader for MAE training (must stream; tiles won't fit in RAM)
- [ ] Test: submit a small 1-node training job, confirm it runs
- [ ] Download OS characteristic sheets from NLS (one per series/edition where available)
- [ ] Extract typographic convention table from characteristic sheets: font style → feature category, per series/edition
- [ ] Train or adapt a lightweight font/style classifier to label GB1900 entries with their visual text style (italic, gothic, roman caps, etc.) — options: (a) fine-tune a small vision model on synthetic OS-style text rendered in the correct fonts, or (b) manually annotate ~500 examples from characteristic sheets as a bootstrap set. Output: each GB1900 entry tagged with its font class.

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
For each tile, aggregate all GB1900 entries and generate a spatially detailed description using each entry's pixel coordinates to derive position within the tile:
```
"Swinton Farm sits in the NW quadrant. St Mary's Church (Ch) is near the centre.
 Two Public Houses (PH) are in the SE, close together. A Footbridge (FB) crosses
 near the eastern edge. River Wharfe runs SW–NE through the lower half."
```
Use OLMo/any LLM to expand abbreviations once, save the mapping, apply at scale. Derive spatial language (quadrant, direction, relative position) from GB1900 lat/lon coordinates mapped to tile pixel positions — this is geometric, not inferred. Need to define "close" and "next to" etc. This is using an LLM to expand ground-truthed labels — not to hallucinate visual content.

Note: GB1900 records text position, not symbol position — labels and symbols are typically within a few pixels of each other on OS maps, so boundary cases are acceptable noise for caption generation. For pixel-precision instruction tuning tasks (Stage 3), exclude GB1900 entries within 32px of a tile edge to avoid ambiguous boundary cases.

**Source B — Local VLM synthetic captions (targeted symbol coverage)**
Run a locally-hosted open VLM on Isambard (e.g. Qwen2-VL-7B or InternVL2-8B) to generate descriptions targeted at unlabelled visual symbols (marshes, orchards, rough pasture, cliff hachures, footpaths, parish boundaries, etc.) — features that GB1900 and MapReader cannot cover. Use OS characteristic sheets to build a per-series symbol vocabulary and include it in the prompt, so captions specifically describe visible symbols rather than giving generic descriptions. Inference on tens of thousands of tiles costs only a few GPUh on GH200s — scale the sample as needed. No external API cost. Be explicit in model card that synthetic captions are used only for caption pretraining, not instruction tuning. Document the fraction of training data this represents.

**Caption detail requirement (applies to both sources)**: captions must be as spatially and semantically detailed as possible. The connector's learned representations are bounded by what the captions required it to encode — a thin caption produces a coarsely-grounded connector that instruction tuning cannot fully recover. Every visible named feature, every major symbol type (from characteristic sheet vocabulary), and every significant spatial relationship should be included, with tile quadrant positions and directional qualifiers throughout. For Source B (local VLM), structure the prompt explicitly to elicit this — generic summaries are not acceptable. This is especially important for 25-inch and town plan tiles outside London, where captions are the primary connector training signal for those visual domains.

Training:
- Freeze encoder (best from Stage 1)
- Freeze OLMo 3 7B
- Train connector MLP only
- Loss: next-token prediction on caption text given image tokens prepended
- Duration: ~200–400 GPUh

Deliverable: connector checkpoint; model can generate basic tile descriptions.

### Stage 3: Instruction tuning (Weeks 8–10)

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

**Font detection and typographic classification tasks (characteristic sheets + GB1900 font labels)**
```
Q: "What type of feature is the italic text in this region labelling?"
A: "A water feature" [from characteristic sheet: italic = water]

Q: "What does the text style of '[Farm Name]' indicate about this feature?"
A: "The roman text indicates this is a named place or agricultural feature, not a water feature or antiquity"

Q: "What does the gothic script in this tile refer to?"
A: "Gothic script on OS maps indicates antiquities or ancient earthworks" [+ location if GB1900 entry present]

Q: "Are there any water features labelled in this tile?"
A: [derived from italic-classified GB1900 entries + characteristic sheet mapping]

Q: "List the typographic styles used in this tile and what they indicate"
A: [structured list: italic entries = water features, roman caps = settlement name, etc.]
```

Training signal: font-labelled GB1900 entries (from font classifier output) cross-referenced with characteristic sheet convention table. No human annotation required beyond the initial characteristic sheet lookup. Note this task is only as good as the font classifier — document its error rate in the model card.

Training:
- Unfreeze top layers of OLMo
- Unfreeze encoder (lower learning rate)
- Train end-to-end
- Duration: ~300–500 GPUh

Deliverable: instruction-tuned model capable of text spotting, feature counting, named entity location, spatial queries, and typographic feature classification.

---

## Compute Budget Summary

| Stage | Activity | GPUh estimate |
|---|---|---|
| Stage 1 | MAE encoder pretraining (~2.26M tiles, 20 epochs) | 225–285 |
| Stage 2 | Connector training | 200–400 |
| Stage 3 | Instruction fine-tuning | 300–500 |
| Buffer | Failed runs, debugging, reruns | 500–1,000 |
| **Total** | | **~1,225–2,185** |

Remaining headroom: ~7,800–8,800 GPUh. Consider scaling to ViT-L (307M params, ~4× compute) only if ViT-B evaluation shows it clearly plateaus below MapReader baselines — not as a default next step. Additional ablations (feature-aware masking, more sheets, more series/editions) are likely better uses of headroom first.

---

## Timeline

The VLM (Stage 3) is the primary goal. Stages 0–2 exist to make it possible. The two training stages that matter most (connector + instruction tuning) are largely unattended — the constraint is engineering time in Stage 0 and dataset generation time in weeks 4–6.

### Weeks 1–2: Downloads (2026-05-27 → 2026-06-09)
- Write and run MapReader download scripts for each series (start with 6-inch 2nd ed.)
- Patchify at 512×512 pixels
- Download GB1900 from NLS Data Foundry
- Download MapReader SIGSPATIAL 2022 from Zenodo

### Weeks 2–4: Data pipeline engineering (2026-06-03 → 2026-06-23, Stage 0)

- Find GB1900 text entries within each patch and create dataset of patch text
- Write webdataset-format streaming dataloader — WebDataset stores patches as sharded tar archives (e.g. `shard-000000.tar`, each containing image files + metadata JSON). The dataloader streams shards sequentially from disk rather than loading all tiles into RAM, which is necessary at this scale (~500K–4M tiles). Shards are shuffled at the shard level; within each shard, samples are shuffled in a small buffer. During MAE training, the dataloader yields `(image_tensor, patch_coords)` pairs; masking is applied on-the-fly on the GPU. Writing shards is a one-time preprocessing step (can be done in parallel with MapReader patchification).
- Train MAE — first submit a smoke-test run (1 node, ~500 tiles, confirm loss decreases and job completes cleanly), then submit the full training run (~20 epochs, unattended from here).

### Weeks 4–6: Dataset generation (2026-06-24 → 2026-07-07, while MAE trains unattended)

- GB1900 tile descriptions — structured, spatially detailed captions listing all named features with tile-quadrant positions and spatial relationships (e.g. "Swinton Farm sits in the NW quadrant, River Wharfe flows SW–NE through the centre, ford crossing at centre-right, St Mary's Church in the SE")
- Further tile text descriptions — from the Zou et al. London text layer (town plans, London only); use MapReader labels to supplement where text coverage is sparse
- Use local VLM to caption patches — use characteristic sheets to elicit spatially detailed descriptions per series (e.g. Qwen2-VL on Isambard-AI): every visible feature type, every named element from GB1900, spatial qualifiers for all major features (quadrant, direction, relative position). Generic summaries are not sufficient — prompt must explicitly request feature-by-feature spatial description.
- Instruction dataset: GB1900 QA pairs + MapReader label tasks
- **Font detection pipeline**: download characteristic sheets → extract typographic convention tables → train/adapt font classifier on GB1900 entry crops → annotate GB1900 entries with font class → generate font detection QA pairs for instruction tuning

### Weeks 6–7: Evaluate encoder + connector training (2026-07-08 → 2026-07-21)

- Linear probe evaluation against MapReader ResNet baselines
- Train connector

### Weeks 7–8: Eval connector + instruction finetuning (2026-07-15 → 2026-07-28)

- Instruction fine-tuning

### Weeks 8–10: Eval + release (2026-07-22 → 2026-08-03)

- Write model card
- Release on HF

---

## Honest Capability Expectations

### The model will be good at:
- Reading and locating text labels (GB1900 training signal is strong)
- Counting named feature types (PH, Ch, FB etc.)
- Answering "where is X" for named places and features
- Basic tile description (settlement, agricultural land, coastal etc.)
- Railspace and building presence/absence (MapReader training signal is strong)
- Identifying feature types from typographic style (italic = water, gothic = antiquities, spaced caps = settlement) — a capability specific to OS map conventions that general VLMs lack

### The model will be weaker at:
- Visual symbol interpretation without adjacent text labels — partially mitigated by local VLM captions using OS characteristic sheets as a symbol vocabulary reference; unlabelled symbols remain weaker than labelled ones
- Features not covered by GB1900 or MapReader datasets
- Series or editions not well-represented in training data
- Heavily degraded or unusual map scans

### What this pipeline cannot claim:
- Full provenance (synthetic captions from a local VLM used for some caption data — document this clearly)
- State-of-the-art performance vs general VLMs on standard benchmarks
- Equal quality across all series and editions — 6-inch 2nd edition will be strongest


---

## Key References and Resources

- MapReader paper + code: https://github.com/maps-as-data/MapReader
- MapReader SIGSPATIAL 2022 dataset: https://zenodo.org/record/7147906
- GB1900 gazetteer: https://data.nls.uk/data/map-spatial-data/gb1900/
- NLS georeferenced maps viewer: https://maps.nls.uk
- MAE paper (He et al. 2021): https://arxiv.org/abs/2111.06377
- Molmo paper (for VLM architecture reference): https://arxiv.org/abs/2409.17146
- OLMo 3: https://allenai.org/papers/olmo3
- NLS Urban Trees dataset: https://data.nls.uk/data/map-spatial-data/urban-trees/
