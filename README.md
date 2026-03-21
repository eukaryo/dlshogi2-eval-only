# dlshogi2-eval-only

`python-dlshogi2` から **評価関数だけ**を切り出して、

- 将棋盤面 `->` feature tensor
- feature tensor `->` dense policy logits / raw value logit
- `torch.export` / `torch.fx` に流せる最小 PyTorch core

を提供するための独立 repo 骨格です。

この repo は将棋エンジンではありません。MCTS、USI ループ、学習コードは含めず、
研究で必要な **eval-only reference** に責務を絞っています。

## 何を upstream から切り出すか

最小限として、次の upstream 実装をベースにします。

- `pydlshogi2/features.py`
- `pydlshogi2/network/policy_value_resnet.py`
- `checkpoints/checkpoint.pth` または別 checkpoint

この repo の `src/dlshogi2_eval/features.py` と `src/dlshogi2_eval/model.py` は、
上記 2 ファイルを eval-only 用に整理し直したものです。

## 設計方針

正準出力は **dense policy logits** と **raw value logit** です。

- dense policy logits: shape `[MOVE_LABELS_NUM]` (= 2187)
- raw value logit: shape `[]`
- legal move policy: dense logits から合法手ラベルだけ gather して softmax
- value probability: `sigmoid(value_logit)` を後段で計算

この分離により、数値誤差評価の対象を NN 本体に限定できます。

## インストール

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 一括セットアップ

研究用に依存導入・upstream snapshot 取得・checkpoint 取得・pytest・smoke inference / export / golden 生成までをまとめて回すには、repo 直下の `prepare_all.sh` を使えます。

```bash
./prepare_all.sh
```

主な出力は `artifacts/prepare_all/` にまとまります。CPU 用 PyTorch wheel を明示したい場合は、たとえば次のように実行します。

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./prepare_all.sh
```

## 使い方

評価:

```bash
dlshogi2-eval-position   --checkpoint /path/to/checkpoint.pth   --position "position startpos moves 7g7f 3c3d"   --topk 10
```

`torch.export` で参照グラフを保存:

```bash
dlshogi2-export-reference   --checkpoint /path/to/checkpoint.pth   --position "position startpos"   --out out/reference.pt2
```

golden 生成:

```bash
dlshogi2-gen-goldens   --checkpoint /path/to/checkpoint.pth   --positions-file positions.txt   --outdir goldens/
```

## release / Zenodo pinning のすすめ方

1. upstream の commit hash を `UPSTREAM_SNAPSHOT.md` に記録
2. その commit から vendoring したファイル名を記録
3. 採用した checkpoint の SHA256 を manifest に記録
4. この repo を GitHub release
5. Zenodo で DOI を付与

## ライセンス

この repo 骨格は GPL-3.0 を前提にしています。
`python-dlshogi2` 由来のコードを整理・改変して含めるため、
公開時も GPL 整合で出す前提にしています。

詳細は `NOTICE` と `THIRD_PARTY_LICENSES/` を見てください。
