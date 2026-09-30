# Choose a configuration

Copy `local.example.yaml` to `local.yaml` and fill in your paths. This file applies to all workflows.

| Directory or file | Purpose |
|---|---|
| `framework/cnn.yaml` | Train one model on the official synthetic data; a starting point for your own configuration. |
| `reproduction/model_a/` | Historical model A lineage: SAM phase 1, SAM phase 2, CNN and co-teaching. |
| `reproduction/model_b/` | Historical model B lineage with its own seeds and overrides. |
| `reproduction/fusion.json` | Historical pair inference settings. |
| `expected.json` | Recorded environment, data counts and reproduction references used by checks. |
| `published_pair.json` | Published pair weight identities and inference configuration. |

Start with `python scripts/repro.py check all` to see reproduction advice, or `python scripts/repro.py run --config configs/framework/cnn.yaml --local` to train one model. Run `python scripts/repro.py reproduce --local` to build the historical pair. Add `--dry` to preview a training workflow.

The historical experiment IDs still contain `s33` and `s34` so checkpoints and recorded evidence retain their original identities. You select the readable model A/B directories; the workflow resolves stage dependencies.
