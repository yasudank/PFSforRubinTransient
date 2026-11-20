# RubinTransient (`PFSforRubinTransient`)

[English](README_EN.md)

Subaru/PFS の `pfsDesign` FITS ファイルの視野内にある Rubin/LSST transient を [Fink/LSST object API](https://api.lsst.fink-portal.org/) から検索し、`SKY` ファイバーのコブラ到達判定・[RubinTargetSelection](https://github.com/yasudank/RubinTargetSelection) (`stamp_classifier`) による超新星候補選定・衝突回避チェック付きのファイバー再割り当てを行うツールです。

## セットアップ (`uv`)

`UV_PROJECT_ENVIRONMENT=.venv-$(uname -m)` の設定に従って仮想環境が作成されます。
`pfsDesign` 処理・コブラ到達判定には Subaru PFS Project のモジュール（`pfs-datamodel`, `pfs-utils`, `pfs-instdata`, `ics-cobraCharmer`, `ics-cobraOps`）、超新星分類には GitHub の `yasudank/RubinTargetSelection` を使用しています。

```bash
uv sync
```

### PFS ソフトウェアバージョンの更新 (`update-pfs-env`)

[config.yaml](config.yaml)（`NetflowByUser` の `config_*.yaml` と互換形式）の `pfs:` セクションから Subaru PFS モジュールのバージョンを解決し、`pyproject.toml` と `uv` 仮想環境を一括更新できます。

```bash
# 解決されるタグと変更有無の事前確認 (--dry-run)
uv run update-pfs-env config.yaml --dry-run

# pyproject.toml 更新 + uv sync 実行
uv run update-pfs-env config.yaml

# >= 指定の最新タグではなく下限タグそのもの (例: w.2026.39) に固定する場合
uv run update-pfs-env config.yaml --exact-floor
```

## 第1段階: `pfsDesign` 視野内の transient 取得 (`find-transients`)

```bash
# 単一ファイルまたはディレクトリを検索（--sky-reachable で SKY コブラ到達可能天体のみに絞り込み）
uv run find-transients \
  /path/to/pfs_designs/pfsDesign-0x....fits \
  --start 2026-11-01 --stop 2026-11-19 \
  --exclude-sso --sky-reachable \
  -o transients.csv
```

## 第2段階: SKY ファイバー到達可能天体の超新星候補選定 (`select-sn-targets`)

以下の順で PostgreSQL DB を使わずにオンデマンド処理を行います：

1. `pfsDesign` の六角形視野内を Fink `/api/v1/conesearch` で検索（非 SSO）
2. `pfs_utils` / `ics_cobraOps` / `pfs_instdata` で **GOOD な SKY ファイバーのコブラ可動範囲内かつブラックドット外** の天体に絞り込み
3. Fink `/api/v1/sources` で全検出履歴を取得し、**2 晩以上観測されている天体** (`--min-nights 2` が既定) に限定
4. Fink `/api/v1/cutouts` から 30×30 のマルチ HDU FITS（Science / Template / Difference、`UNCERT`/`PSFIMAGE` 付き）を 2 段階並列取得してローカルキャッシュ（`--local-cutouts` で既存カットアウトがあれば再利用）
5. `RubinTargetSelection` (`stamp_classifier`) の固定モデルバンドル (`snp_v1`) で Step 1 / moving 判定 / Step 2 分類を実行し、`design_sn_candidates.csv`, `candidates.csv`, `candidates.pdf` 等を出力

```bash
uv run select-sn-targets \
  /path/to/pfs_designs \
  --start 2026-11-01 --stop 2026-11-19 \
  --bundle-dir /path/to/models/sn_pipeline/snp_v1 \
  -o sn_output/run1
```

### 主なオプション (`select-sn-targets`)
- `--start` / `--stop`: 検索期間 (UTC)
- `--min-nights`: 最小観測夜数（既定 `2`）
- `--bundle-dir`: 固定モデルバンドル (`snp_v1`) のパス
- `--cache-dir`: カットアウトのローカルキャッシュ先（既定 `cache/cutouts`）
- `--local-cutouts`: 既存ローカルカットアウトディレクトリ（存在すればシンボリックリンクで再利用、空文字で無効化）
- `--obstime`: sky→PFI 座標変換に使う観測時刻 (UTC ISO)
- `--strict`: Step 2 が `uncertain` の天体を除外
- `--include-negative-galaxy`: Step 1 が `negative_real` かつ Step 2 が `galaxy` の天体も含める（既定は除外）
- `--no-pdf`: `candidates.pdf` の生成をスキップ

## 第3段階: `SKY` ファイバーの再割り当てと新 `pfsDesign` 書き出し (`reassign-pfs-design`)

`select-sn-targets` が出力した `design_sn_candidates.csv` を読み込み、各 `pfsDesign` について以下の安全チェックを通過した SN 候補だけを `SKY` ファイバーへ再割り当てし、新しい `pfsDesign-0x....fits`（`variant` インクリメント、`designId0` に元 ID 記録、`calculate_pfsDesignId` で ID 再計算、`catId`/`proposalId` は元ファイルの `SCIENCE` ファイバーから継承）を出力します。

- 明るさ制限（バンドにかかわらず最新観測の `scienceFlux` AB 等級が `--min-mag` 以下は除外、既定 `18.0` 等）
- コブラ可動範囲（`rMin <= r <= rMax`）およびブラックドット回避（`margin=1.65`）
- Fiducial 回避（`>= 2.9 mm`）
- 隣接割当コブラとの端点距離（`>= 2.1 mm`）
- `ics.cobraOps.CollisionSimulator` による **新規端点衝突 0 件 & 新規軌道衝突 0 件** の検証

```bash
uv run reassign-pfs-design \
  --candidates sn_output/run1/design_sn_candidates.csv \
  --design-dir /path/to/pfs_designs \
  --min-mag 18.0 \
  -o modified_designs/run1
```
