# Upstream snapshot record

- Upstream repository: `https://github.com/TadaoYamaoka/python-dlshogi2`
- Upstream commit hash: `358a704eb3ebc87871fff36a436eaad233d85a44`
- Snapshot date: `2026-03-21`
- Vendored / adapted files:
  - upstream `pydlshogi2/features.py` -> local `src/dlshogi2_eval/features.py`
  - upstream `pydlshogi2/network/policy_value_resnet.py` -> local `src/dlshogi2_eval/model.py`
- Checkpoint source path: `checkpoints/checkpoint.pth`
- Checkpoint SHA256: `90eb745be1079a76371d0dd96009d712b9383631f85bcffb050169d28bd5afd5`
- Notes on any local modifications:
  - extracted eval-only functionality
  - removed training / MCTS / engine-loop responsibilities
  - added standalone CLI / export helpers / tests
