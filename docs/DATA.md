# Training data

This document lists the datasets used for training and explains how to prepare them to reproduce the DA-OCT challenge results.

## Datasets and size

Counts below are B-scans prepared for the historical recipes, rather than the full size of each original database.

| Dataset | Labelled B-scans | Unlabelled B-scans | Source |
|---|---:|---:|---|
| DA-OCT synthetic release | 230 | — | [Official baseline and data](https://github.com/wusmai/miccai-challenge-daoct-baseline) |
| JHU HC-MS | 1,715 | — | [Download](https://iacl.ece.jhu.edu/~aaron/data/OCT_Manual_Delineations-2018_June_29_b.zip) |
| Duke DME 2015 | 110 | — | [Download](https://people.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip) |
| OCT5k | 1,672 | — | [Dataset and image preparation](https://doi.org/10.5522/04/22128671) |
| AI-READI v3 | 287,036 | 377,646 | [Dataset](https://doi.org/10.60775/fairhub.3) |
| **Total prepared** | **290,763** | **377,646** | **668,409 B-scans across the two pools** |

These are available pool sizes, not the number of new images per epoch or a sum across training stages. The recipes reuse the same pools. For SAM training and co-teaching, the Cirrus optic-disc contribution is capped at four B-scans per volume: 13,120 instead of 52,433. The CNN stages use the uncapped pool.

| Historical local training view | Labelled training B-scans | Synthetic images held out |
|---|---:|---:|
| SAM phases and co-teaching | 251,406 | 44 |
| CNN stages | 290,717 | 46 |

Both lineages use these views. Co-teaching also uses the 377,646 unlabelled B-scans. Source counts and run-specific splits are recorded in [expected.json](../configs/expected.json). Training settings are in [the recipes](../configs/reproduction/).

## Download and prepare

Set `data` in `configs/local.yaml` to your data root. The download helpers place and extract the supported archives there:

```bash
python scripts/repro.py download official
python scripts/repro.py download public
# SAM initialization, needed only for SAM-based training:
python scripts/repro.py download sam
```

The public command downloads JHU HC-MS, Duke DME and OCT5k annotations. Obtain the OCT5k source images using the image-preparation instructions in its dataset download, and obtain AI-READI through its dataset page.

After downloading, the relevant folders under your configured data root are:

```text
synthetic_v1.0/release_dataset/        # official synthetic images and masks
public/extracted/jhu_hcms/            # extracted JHU archive
public/extracted/duke_dme_2015/        # extracted Duke archive
public/extracted/oct5k_annotations/   # extracted OCT5k annotations
public/extracted/isfahan/             # OCT5k source images
public/ai_readi/                      # AI-READI v3 release
```

Keep the AI-READI release structure, including `participants.tsv`, the structural OCT and OCTA manifests, and their DICOM files. The builder reads these files directly.

```bash
python scripts/repro.py check data
python scripts/repro.py build --dry
python scripts/repro.py build --local
python scripts/repro.py check pools
```

`build` produces the labelled and unlabelled training pools under `derived/`. Checks compare them with the historical counts and report differences as advice. To build only the public labelled sources, use `build --public-only --local`.

For a first standalone CNN run, only the official synthetic images and masks are needed; use `configs/framework/cnn.yaml`. See [REPRODUCE](REPRODUCE.md) for training and inference commands.

## Label mapping

All sources are mapped to the nine interfaces shown in [METHOD](METHOD.md#task-and-labels). The ten pixel classes are the regions separated by those interfaces. Missing interfaces remain partially supervised rather than being filled with invented labels.

| Source | Interfaces used | Preparation for training |
|---|---|---|
| DA-OCT synthetic release | b1, b5, b7, b8, b9 | Map the released b4 line to b5 (INL/OPL); discard the four interpolated lines. |
| JHU HC-MS | b1, b2, b4, b5, b6, b7, b8, b9 | Drop OPL/ONL; leave b3 unannotated. Convert MATLAB coordinates and interpolate sparse points along each annotated surface with PCHIP. |
| Duke DME 2015 | b1, b2, b4, b5, b7, b8, b9 | Drop OPL/ONL; leave b3 and b6 unannotated. Convert MATLAB row coordinates. |
| OCT5k | b1, b5, b7, b8, b9 | Map the dataset's OPL line to INL/OPL (b5), IBRPE to b8, and OBRPE to b9. Use grader 1 in the historical recipes. |
| AI-READI Spectralis | b1–b9 | Select nine of the eleven surfaces; drop OPL/ONL and the RPE-centre surface. |
| AI-READI Maestro2 | b1–b9 | Map the nine surfaces in depth order to the nine interfaces. |
| AI-READI Triton | b1–b9 | Map the first nine depth-ordered surfaces; drop the choroid/sclera interface. |
| AI-READI Cirrus | b1, b9 | Use ILM for b1; shift the RPE-centre surface 9 pixels deeper as the BM target (b9). |
| AI-READI unlabelled pool | None | Images only; co-teaching generates targets during training. |

`octtta/data/partial_labels.py` defines these mappings and offsets. The builders apply them before rasterization; the recipes select sources, graders and sampling settings.
