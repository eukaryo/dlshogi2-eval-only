# dlshogi2-eval-only

[`python-dlshogi2`](https://github.com/eukaryo/python-dlshogi2) （fork元: https://github.com/TadaoYamaoka/python-dlshogi2 ）から **評価関数だけ**を切り出して、

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

`pip install -e .` のあと、または `prepare_all.sh` 完了後に `.venv` を activate すると、
次の console script が使えるようになります。

- `dlshogi2-eval-position`
- `dlshogi2-export-reference`
- `dlshogi2-gen-goldens`

## 一括セットアップ

研究用に依存導入・upstream snapshot 取得・checkpoint 取得・pytest・smoke inference / export / golden 生成までをまとめて回すには、repo 直下の `prepare_all.sh` を使えます。

```bash
./prepare_all.sh
```

既定では、**bootstrap 用の取得先は自分で管理する mirror fork**
`https://github.com/eukaryo/python-dlshogi2` を使います。
一方で provenance は `TadaoYamaoka/python-dlshogi2` に対して記録します。

主な出力は `artifacts/prepare_all/` にまとまります。CPU 用 PyTorch wheel を明示したい場合は、たとえば次のように実行します。

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./prepare_all.sh
```

## 使い方

準備:

```bash
./prepare_all.sh
source .venv/bin/activate
CHECKPOINT=third_party/upstream/eukaryo-python-dlshogi2-358a704eb3ebc87871fff36a436eaad233d85a44/checkpoints/checkpoint.pth
UPSTREAM_COMMIT=358a704eb3ebc87871fff36a436eaad233d85a44
```

盤面評価:

```bash
dlshogi2-eval-position   --checkpoint "$CHECKPOINT"   --position "position startpos moves 7g7f 3c3d"   --topk 10   --pretty
```

`torch.export` で参照グラフを保存し、`.pt2` と manifest と可読テキストをまとめて出す:

```bash
dlshogi2-export-reference   --checkpoint "$CHECKPOINT"   --position "position startpos"   --out out/reference.pt2   --manifest out/reference.manifest.json   --text-dump-dir out   --text-dump-stem reference_startpos   --upstream-commit "$UPSTREAM_COMMIT"
```

これで少なくとも次が出ます。

- `out/reference.pt2`
- `out/reference.manifest.json`
- `out/reference_startpos.exported_program.txt`
- `out/reference_startpos.graph_ir.txt`
- `out/reference_startpos.graph_module_code.py`

golden 生成（ここでいう golden とは、PyTorch/CPU の参照実装で出した **正解側の出力ファイル群** のことです）:

```bash
dlshogi2-gen-goldens   --checkpoint "$CHECKPOINT"   --positions-file positions.txt   --outdir goldens/
```

## release / Zenodo pinning のすすめ方

1. `UPSTREAM_SNAPSHOT.md` に **元の upstream repository** と **bootstrap mirror repository** の両方を記録
2. 採用した commit hash を 40 桁で記録
3. vendored / adapted files の対応表を記録
4. 採用した checkpoint の SHA256 を `UPSTREAM_SNAPSHOT.md` と `<out>.manifest.json` に記録
5. この repo を GitHub release
6. Zenodo で DOI を付与

## ライセンス

この repo 骨格は GPL-3.0 を前提にしています。
`python-dlshogi2` 由来のコードを整理・改変して含めるため、
公開時も GPL 整合で出す前提にしています。

詳細は `NOTICE` と `THIRD_PARTY_LICENSES/` を見てください。
