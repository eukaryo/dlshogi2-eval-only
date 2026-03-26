# dlshogi2-eval-only

[`python-dlshogi2`](https://github.com/eukaryo/python-dlshogi2) （fork 元: `https://github.com/TadaoYamaoka/python-dlshogi2`）から **評価関数だけ**を切り出して、次を独立に扱えるようにした repo です。

- 将棋盤面 `->` feature tensor
- feature tensor `->` dense policy logits / raw value logit
- `torch.export` / `torch.fx` に流せる最小 PyTorch core
- 参照用 package の出力
- 他のソフトが出した raw output を、将棋の静的評価として解釈する機能

この repo は **将棋エンジンではありません**。MCTS、USI ループ、学習コードは含めず、研究で必要な **eval-only reference** に責務を絞っています。

---

## 何を upstream から切り出すか

最小限として、次の upstream 実装をベースにします。

- `pydlshogi2/features.py`
- `pydlshogi2/network/policy_value_resnet.py`
- `checkpoints/checkpoint.pth` または別 checkpoint

この repo の `src/dlshogi2_eval/features.py` と `src/dlshogi2_eval/model.py` は、上記 2 ファイルを eval-only 用に整理し直したものです。

---

## 設計方針

### 正準の生出力

この repo における **NN 本体の正準出力** は次の 2 つです。

- `policy_logits`: dense policy logits
- `value_logit`: raw value logit

将棋としての意味付けはその後段で行います。

- legal move policy: dense logits から合法手ラベルだけ gather して softmax
- value probability: `sigmoid(value_logit)`

この分離により、数値誤差評価の対象を NN 本体に限定しやすくなります。

### shape / key の約束

外部ソフトとの受け渡しに使う **canonical raw tensor format** は次です。

- input feature tensor: `features`, shape `[1, 104, 9, 9]`
- output policy tensor: `policy_logits`, shape `[1, 2187]`
- output value tensor: `value_logit`, shape `[1, 1]`

この repo 内の convenience API では、扱いやすさのために次のように見えることがあります。

- dense policy logits: shape `[2187]`
- raw value logit: scalar 相当

ただし `.npz` に保存して他のソフトと比較するときは、**必ず canonical raw tensor format**（`[1, 2187]` と `[1, 1]`）で出します。

### JSON 出力の統一 schema

PyTorch 側で局面評価した結果も、他のソフトが出した raw output を将棋として解釈した結果も、次のトップレベル schema に揃います。

```json
{
  "backend": {...},
  "dense": {...},
  "legal": {...}
}
```

- `backend`: どの backend で得た結果か
- `dense`: dense policy / raw value の情報
- `legal`: 合法手に制限した policy と value の情報

研究用途では、

- raw tensor 同士の比較
- legal move policy の比較
- value probability の比較

を同じ土俵で扱えます。

---

## インストール

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

この版では `safetensors` も依存に含まれます。package export を使う場合に必要です。

`pip install -e .` のあと、または `prepare_all.sh` 完了後に `.venv` を activate すると、次の console script が使えるようになります。

- `dlshogi2-eval-position`
- `dlshogi2-export-reference`
- `dlshogi2-gen-goldens`
- `dlshogi2-export-model-package`
- `dlshogi2-interpret-outputs`

後ろ 2 つは次の用途に使います。

- `dlshogi2-export-model-package`: **外部ソフト向け package exporter**
- `dlshogi2-interpret-outputs`: **他のソフトが出した raw output の将棋解釈器**

---

## 一括セットアップ

研究用に依存導入・upstream snapshot 取得・checkpoint 取得・pytest・smoke inference / export / golden 生成までをまとめて回すには、repo 直下の `prepare_all.sh` を使えます。

```bash
./prepare_all.sh
```

既定では、**bootstrap 用の取得先は自分で管理する mirror fork** `https://github.com/eukaryo/python-dlshogi2` を使います。一方で provenance は `TadaoYamaoka/python-dlshogi2` に対して記録します。

主な出力は `artifacts/prepare_all/` にまとまります。

CPU 用 PyTorch wheel を明示したい場合は、たとえば次のように実行します。

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./prepare_all.sh
```

---

## 準備: checkpoint と upstream commit

以降の例では次を前提にします。

```bash
./prepare_all.sh
source .venv/bin/activate

CHECKPOINT=third_party/upstream/eukaryo-python-dlshogi2-358a704eb3ebc87871fff36a436eaad233d85a44/checkpoints/checkpoint.pth
UPSTREAM_COMMIT=358a704eb3ebc87871fff36a436eaad233d85a44
```

---

## 入力形式

多くの CLI では、局面入力として次のどちらかを受け付けます。

- `--position`: USI の `position ...` 文字列
- `--sfen`: SFEN 文字列

例:

```bash
--position "position startpos moves 7g7f 3c3d"
```

```bash
--sfen "lnsgkgsnl/1r5b1/p1ppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 2"
```

---

## 1. 盤面を PyTorch で静的評価する

### 最小例: CPU / fp32

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --topk 10 \
  --pretty
```

返る JSON は概ね次の形です。

```json
{
  "backend": {
    "kind": "pytorch",
    "device": "cpu",
    "precision": "fp32",
    "use_autocast": false
  },
  "dense": {
    "policy_size": 2187,
    "value_logit": 0.123,
    "value_prob": 0.5307
  },
  "legal": {
    "num_legal_moves": 30,
    "topk": {
      "moves": ["2g2f", "8h2b+"],
      "probs": [0.21, 0.13],
      "logits": [1.25, 0.77],
      "indices": [0, 1]
    },
    "value_logit": 0.123,
    "value_prob": 0.5307
  }
}
```

### GPU で評価する

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cuda:0 \
  --precision fp32 \
  --pretty
```

### GPU の低精度で評価する

RTX 4000 系などで低精度の数値誤差を見たいときは、`--precision` と `--autocast` を使います。

#### bf16

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cuda:0 \
  --precision bf16 \
  --autocast \
  --topk 10 \
  --pretty
```

#### fp16

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cuda:0 \
  --precision fp16 \
  --autocast \
  --topk 10 \
  --pretty
```

### SFEN で評価する

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --sfen "lnsgkgsnl/1r5b1/p1ppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 2" \
  --pretty
```

### 合法手一覧や dense policy 全体も出す

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos" \
  --include-all-legal \
  --include-dense-policy \
  --pretty
```

- `--include-all-legal`: 全合法手の `usi / label / logit / prob` を出す
- `--include-dense-policy`: `[2187]` 全体を JSON に出す

後者は非常に長くなるので、通常は raw tensor を `.npz` に保存する方が扱いやすいです。

### canonical raw outputs を `.npz` で保存する

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cuda:0 \
  --precision bf16 \
  --autocast \
  --raw-outputs-npz out/pytorch_bf16_outputs.npz \
  --pretty
```

この `.npz` には次の 2 キーが入ります。

- `policy_logits` : shape `[1, 2187]`
- `value_logit` : shape `[1, 1]`

他のソフトの出力と比較したいときは、この形式を共通の比較フォーマットとして使ってください。

---

## 2. `torch.export` の参照グラフを保存する

`.pt2` と manifest と可読テキストをまとめて出します。

```bash
dlshogi2-export-reference \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos" \
  --out out/reference.pt2 \
  --manifest out/reference.manifest.json \
  --text-dump-dir out \
  --text-dump-stem reference_startpos \
  --upstream-commit "$UPSTREAM_COMMIT"
```

少なくとも次が出ます。

- `out/reference.pt2`
- `out/reference.manifest.json`
- `out/reference_startpos.exported_program.txt`
- `out/reference_startpos.graph_ir.txt`
- `out/reference_startpos.graph_module_code.py`

用途:

- `torch.export` の graph を調べる
- importer / compiler 側の受け口確認
- graph の provenance 記録

---

## 3. golden を生成する

ここでいう golden とは、**PyTorch の参照実装で出した正解側の出力ファイル群**のことです。

```bash
dlshogi2-gen-goldens \
  --checkpoint "$CHECKPOINT" \
  --positions-file positions.txt \
  --outdir goldens/
```

`positions.txt` はたとえば次のように書けます。

```text
position startpos
position startpos moves 7g7f 3c3d
sfen lnsgkgsnl/1r5b1/p1ppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 2
```

用途:

- PyTorch/CPU の参照結果を固定する
- 比較実験で再利用する
- スモークテストや回帰テストに使う

---

## 4. 外部ソフト向け package を出力する

```bash
dlshogi2-export-model-package \
  --checkpoint "$CHECKPOINT" \
  --positions-file positions.txt \
  --outdir out/package \
  --upstream-commit "$UPSTREAM_COMMIT"
```

### 何が出るか

典型的には次のような tree になります。

```text
out/package/
  manifest.json
  graph/
    model.pt2
  weights/
    manifest.json
    weights_000.safetensors
  cases/
    case_000001/
      inputs.npz
      meta.json
      reference_outputs.npz
      readable.txt
    case_000002/
      ...
```

### package 内の主なファイル

- `graph/model.pt2`
  - 参照用の `torch.export` graph
- `weights/weights_000.safetensors`
  - state_dict を safetensors で保存したもの
- `weights/manifest.json`
  - 重みの名前・dtype・shape・対応ファイル
- `cases/<case_id>/inputs.npz`
  - `features` を含む入力 tensor 群
- `cases/<case_id>/reference_outputs.npz`
  - `policy_logits` / `value_logit` を含む参照出力
- `cases/<case_id>/meta.json`
  - 局面文字列、shape、dtype、hash などのメタデータ
- `cases/<case_id>/readable.txt`
  - 人間がざっと確認するための要約

### 主なオプション

```bash
dlshogi2-export-model-package --help
```

よく使うもの:

- `--overwrite`
  - 出力先を上書き
- `--no-reference-outputs`
  - `reference_outputs.npz` を省略
- `--no-readable`
  - `readable.txt` を省略
- `--device cpu|cuda:0`
  - export 時の実行 device
- `--producer-name`
- `--producer-version`
- `--producer-git-commit`
- `--notes`
- `--package-id-prefix`

### package の I/O 名

package では、少なくとも次の名前を使います。

- input: `features`
- outputs: `policy_logits`, `value_logit`

この名前は、後段のソフトが `inputs.npz` / `reference_outputs.npz` を機械的に扱えるようにするための socket です。

---

## 5. 他のソフトが出した raw output を将棋として解釈する

この repo には、**外部ソフトがすでに計算した raw output** を受け取って、それを将棋の静的評価として解釈する CLI があります。

- 外部ソフトが出した `policy_logits` / `value_logit` を受け取る
- それを将棋の合法手 policy / value probability に変換する
- PyTorch 参照実装と同じ JSON schema に揃える

### 受け取れる入力形式

次のいずれか 1 つを使います。

#### 1. `.npz` 1 ファイル

```bash
dlshogi2-interpret-outputs \
  --outputs-npz out/outputs.npz \
  --position "position startpos moves 7g7f 3c3d" \
  --pretty
```

この `.npz` には次が必要です。

- `policy_logits`
- `value_logit`

#### 2. run directory

```bash
dlshogi2-interpret-outputs \
  --run-dir out/run_0001 \
  --position "position startpos moves 7g7f 3c3d" \
  --pretty
```

`--run-dir` では次のどれかを探します。

- `outputs.npz`
- `reference_outputs.npz`
- `policy_logits.npy` と `value_logit.npy` のペア

#### 3. `.npy` 2 本

```bash
dlshogi2-interpret-outputs \
  --policy-logits-npy out/policy_logits.npy \
  --value-logit-npy out/value_logit.npy \
  --position "position startpos moves 7g7f 3c3d" \
  --pretty
```

### 局面の与え方

局面は次のいずれかで指定します。

- `--position`
- `--sfen`
- `--case-meta`

#### package の `meta.json` をそのまま使う

```bash
dlshogi2-interpret-outputs \
  --outputs-npz out/run_0001/outputs.npz \
  --case-meta out/package/cases/case_000001/meta.json \
  --pretty
```

これにより、局面文字列をもう一度手で打たずに、export 時のケース情報から自動で盤面を復元できます。

### 出力される JSON

返る JSON の形は `dlshogi2-eval-position` と同じです。

- `backend.kind` は `external` を返します
- `backend.device` / `backend.precision` は外部ソフト由来なので `external`

つまり、同じ局面に対して

- PyTorch/CPU
- PyTorch/GPU fp32
- PyTorch/GPU bf16
- 他のソフトの output

を **同じ JSON schema** で比較できます。

### legal move 一覧や dense policy も出せる

```bash
dlshogi2-interpret-outputs \
  --outputs-npz out/outputs.npz \
  --position "position startpos" \
  --include-all-legal \
  --include-dense-policy \
  --pretty
```

---

## 6. 典型的な比較ワークフロー

### A. PyTorch 参照実装だけをまず確認する

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cpu \
  --precision fp32 \
  --raw-outputs-npz out/ref_cpu.npz \
  --pretty > out/ref_cpu.json
```

### B. GPU 低精度の参照結果を取る

```bash
dlshogi2-eval-position \
  --checkpoint "$CHECKPOINT" \
  --position "position startpos moves 7g7f 3c3d" \
  --device cuda:0 \
  --precision bf16 \
  --autocast \
  --raw-outputs-npz out/ref_bf16.npz \
  --pretty > out/ref_bf16.json
```

### C. package を作る

```bash
dlshogi2-export-model-package \
  --checkpoint "$CHECKPOINT" \
  --positions-file positions.txt \
  --outdir out/package \
  --upstream-commit "$UPSTREAM_COMMIT"
```

### D. 外部ソフトが出した結果を将棋として解釈する

```bash
dlshogi2-interpret-outputs \
  --outputs-npz out/external_outputs.npz \
  --case-meta out/package/cases/case_000001/meta.json \
  --pretty > out/external.json
```

この時点で

- `out/ref_cpu.npz`
- `out/ref_bf16.npz`
- `out/external_outputs.npz`

はいずれも `policy_logits` / `value_logit` を持つので、raw tensor 比較ができます。

また

- `out/ref_cpu.json`
- `out/ref_bf16.json`
- `out/external.json`

はいずれも `backend/dense/legal` schema に揃っているので、合法手 policy や value probability の比較もできます。

### E. `.npz` 同士をざっくり比較する例

```bash
python - <<'PY'
import numpy as np

a = np.load('out/ref_cpu.npz')
b = np.load('out/external_outputs.npz')

for key in ['policy_logits', 'value_logit']:
    da = a[key].astype(np.float64)
    db = b[key].astype(np.float64)
    diff = da - db
    print(key)
    print('  shape      =', da.shape)
    print('  max_abs    =', np.max(np.abs(diff)))
    print('  mean_abs   =', np.mean(np.abs(diff)))
    print('  rms        =', np.sqrt(np.mean(diff * diff)))
PY
```

---

## 7. 補助的な約束ごと

### `policy_logits` / `value_logit` の意味

- `policy_logits`
  - softmax 前の dense policy logits
  - 2187 ラベル空間全体
- `value_logit`
  - sigmoid 前の raw value logit

### `legal` セクションの意味

`legal` は、その局面の合法手だけを `policy_logits` から gather して作ったものです。

- `legal.topk.moves`
  - USI 文字列
- `legal.topk.probs`
  - 合法手上の softmax 確率
- `legal.topk.logits`
  - 対応する logits
- `legal.value_prob`
  - `sigmoid(value_logit)`

### 温度 `--temperature`

`--temperature` は合法手 policy の softmax に使います。

- `1.0`: 通常の softmax
- `<= 0`: argmax を 1.0 にした one-hot 的扱い

---

## 8. release / Zenodo pinning のすすめ方

1. `UPSTREAM_SNAPSHOT.md` に **元の upstream repository** と **bootstrap mirror repository** の両方を記録
2. 採用した commit hash を 40 桁で記録
3. vendored / adapted files の対応表を記録
4. 採用した checkpoint の SHA256 を `UPSTREAM_SNAPSHOT.md` と manifest に記録
5. この repo を GitHub release
6. Zenodo で DOI を付与

package export を使う場合は、次も残しておくと便利です。

- producer version
- producer git commit
- package id
- case ごとの input / output hash

---

## 9. ライセンス

この repo 骨格は GPL-3.0 を前提にしています。
`python-dlshogi2` 由来のコードを整理・改変して含めるため、公開時も GPL 整合で出す前提にしています。

詳細は `NOTICE` と `THIRD_PARTY_LICENSES/` を見てください。
