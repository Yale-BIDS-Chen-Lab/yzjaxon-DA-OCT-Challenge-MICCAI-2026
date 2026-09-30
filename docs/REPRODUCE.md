# From an empty checkout to your first result

The framework needs a Python environment, a private path configuration, and the inputs for the operation you choose. Existing weights can predict immediately. A custom model can train in one run. Rebuilding the historical published pair is a separate, optional workflow.

## 1. Install and fill in paths

Use Python 3.11 and the recorded dependencies as a starting point:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
cp configs/local.example.yaml configs/local.yaml
```

For example, edit the local file to contain:

```yaml
scratch: /work/oct
# These can point to existing folders; null uses a subfolder of scratch.
data: /work/oct/data
runs: /work/oct/runs
ckpt: /work/oct/checkpoints
published_weights: /work/oct/weights
```

`data` is the root containing the source directories, not the directory for one image cohort. `runs` stores resolved configs, training checkpoints and output reports. `ckpt` stores initialization weights; final published weights go in `published_weights`. Environment variables such as `OCTTTA_DATA` override the local file. The local file is ignored by Git.

[DATA](DATA.md) lists the datasets, download commands and directory layout. For a first single-model run using the official synthetic set, place it at `/work/oct/data/synthetic_v1.0/release_dataset/` for the example above. You do not need AI-READI, SAM initialization, or a second model for that example. Full historical reproduction requires the additional licensed sources described there.

## 2. Ask what matches the recorded reproduction

```bash
python scripts/repro.py check all
```

The check reports installed versions and available inputs against the recorded reproduction. A version, count or hash difference produces advice; it does not prevent you from running. Missing historical inputs also appear as advice when you only want to use a different recipe. An operation still needs its actual inputs: a missing training directory or invalid YAML cannot produce a valid run.

For a particular custom recipe, check its actual configuration:

```bash
python scripts/repro.py check framework --config configs/framework/cnn.yaml
```

The framework check examines the selected recipe, rather than requiring all historical inputs. Optional reproduction comparisons are separate from runtime validity.

## 3. Choose one of three workflows

### Predict using weights

Place `model.pt` and `model_b.pt` in the configured published-weight directory once they are available. No public weights URL has been approved yet.

```bash
python scripts/repro.py infer --images /path/to/images --out /path/to/masks \
  --weights published --local
```

The input is a flat directory of image files. Each output is a one-channel PNG containing labels 0–9 at the original image size. For CPU inference or a single checkpoint:

```bash
python -m octtta.infer --input /path/to/images --output /path/to/masks \
  --checkpoint /path/to/checkpoint.pt --device cpu --no-degrade
```

Add `--checkpoint-b /path/to/model_b.pt` for a dual-model pair. Use a new output directory for each run.

### Train your own model in one run

Start with `configs/framework/cnn.yaml`. Edit the experiment name, data root, model and training settings there, or copy the file beside it to keep the example. Relative `_base_` references resolve from the configuration file's directory.

```bash
python scripts/repro.py run --config configs/framework/cnn.yaml --dry
python scripts/repro.py run --config configs/framework/cnn.yaml --local
```

This trains a CNN from scratch on the configured labelled data. It does not require pretrained SAM weights or historical co-teaching. Each run writes `config.resolved.yaml` and checkpoints under `runs/<experiment.id>/`. A plain trainer invocation also accepts dotted overrides:

```bash
python -m octtta.train configs/framework/cnn.yaml experiment.id=my_cnn train.epochs=2
```

Export the trained checkpoint for inference, retaining its own model and inference settings:

```bash
python scripts/export/export_pair.py \
  --a /work/oct/runs/my_cnn/checkpoints/last.pt --out /work/oct/my_export
python -m octtta.infer --input /path/to/images --output /work/oct/my_masks \
  --checkpoint /work/oct/my_export/model.pt --no-degrade
```

The exporter uses EMA weights when present and otherwise the trained weights. `--b` optionally adds a second model. It accepts changed architectures and does not require a competition deployment contract. It refuses to overwrite an existing export.

### Rebuild the historical pair

Obtain all required data and SAM initialization through their authorized channels. Download helpers are available:

```bash
python scripts/repro.py download public
python scripts/repro.py download official
python scripts/repro.py download sam
```

The historical workflow builds the local pools, then follows the recipe dependencies:

```bash
python scripts/repro.py reproduce --dry
python scripts/repro.py reproduce --local
```

It starts with reproduction advice and then executes the data and training workflow. For a Slurm cluster, omit `--local` and fill in the `slurm` section of the local file. The dry preview does not submit jobs. Configuration changes are allowed; resolved settings are saved for each run.

If the GPU partition or QoS rejects CPU-only jobs such as export, set `slurm.cpu.partition: day` and `slurm.cpu.qos: normal` in `configs/local.yaml` (see `configs/local.example.yaml`). CPU jobs inherit `slurm.account` unless overridden; `null` explicitly clears an inherited CPU partition or QoS. GPU job flags stay on the top-level Slurm settings.

Recipes are grouped under `configs/reproduction/model_a/` and `configs/reproduction/model_b/`. Each lineage trains SAM phase 1, SAM phase 2, CNN and co-teaching; the workflow supplies the warm starts automatically. Export takes A from the first lineage and B from the second. [METHOD](METHOD.md) explains why that particular published pair used eight runs. This cost is specific to historical reproduction. The direct framework and inference workflows above do not require it.

## 4. Measure a pipeline result

Obtain the official synthetic data and official scorer locally, then run:

```bash
python scripts/repro.py score --weights published --local
# Or, after historical reproduction:
python scripts/repro.py score --weights mine --local
```

The report includes the measured score, weight origin, environment comparison and difference from the recorded synthetic reference. A reproduction difference does not fail the command. Inference failures, incomplete masks or a failed scorer do. Most of these synthetic images were included in historical training: the result is a pipeline check, not a held-out performance estimate or a remeasurement of the competition Final score. New training is not promised to produce identical weights.

Run the repository tests with `python -m pytest -q tests`. Tests requiring unavailable licensed data are marked through the declared data requirements.

## Official execution and model publication

For competition-specific Docker, kit and submission instructions, follow the [official resources in the README](../README.md#official-resources). Competition-time fine-tuning and ZIP packaging are outside this framework. The original submission ZIP is preserved separately; its public hosting is awaiting approval.
