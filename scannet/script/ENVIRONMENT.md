## Python environment

This repo includes a single environment spec that is intended to run:
- Grounded-SAM openset masking (`scannet/script/grounded_sam/...`)
- RAM tagging (`scannet/script/ram/...`)
- OpenAI batch tooling (`scannet/script/openai_tools/...`)
- Visualization / utilities under `script/` and `scannet/script/`

Create and activate:

```bash
conda env create -f environment.yml
conda activate scannet-sg
```

Update an existing env:

```bash
conda env update -f environment.yml --prune
```

Notes:
- The optional third-party repos are cloned on demand into `scannet/script/thirdparty/` when you run the scripts.
- Model checkpoints (e.g. SAM, GroundingDINO, RAM++ weights) are **not** included in the environment and must be downloaded separately.

