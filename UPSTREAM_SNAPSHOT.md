# Upstream snapshot record

Fill this in before publishing a frozen GitHub release / Zenodo release.

- Upstream repository: `https://github.com/TadaoYamaoka/python-dlshogi2`
- Upstream commit hash: `<FILL ME>`
- Snapshot date: `<FILL ME>`
- Vendored / adapted files:
  - `pydlshogi2/features.py`
  - `pydlshogi2/network/policy_value_resnet.py`
- Checkpoint source path: `<FILL ME>`
- Checkpoint SHA256: `<FILL ME>`
- Notes on any local modifications: `<FILL ME>`

## Suggested pinning workflow

1. Clone upstream and checkout the exact commit.
2. Copy the target files into this repo and review diffs.
3. Download or vendor the checkpoint selected for the study.
4. Compute SHA256 for the checkpoint and record it here.
5. Run `dlshogi2-export-reference` and record the exported program hash.
6. Tag the repo and create a GitHub release.
7. Archive the release with Zenodo and note the DOI here.
