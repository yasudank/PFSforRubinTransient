# RubinTransient (`PFSforRubinTransient`)

Subaru/PFS の `pfsDesign` FITS ファイルの視野内にある Rubin/LSST transient を [Fink/LSST object API](https://api.lsst.fink-portal.org/) から検索し、`SKY` ファイバーのコブラ到達判定および [RubinTargetSelection](https://github.com/yasudank/RubinTargetSelection) (`stamp_classifier`) による超新星候補選定を行うツールです。

## セットアップ (`uv`)

`UV_PROJECT_ENVIRONMENT=.venv-$(uname -m)` の設定に従って仮想環境が作成されます。
`pfsDesign` 処理・コブラ到達判定には Subaru PFS Project のモジュール（`pfs-datamodel`, `pfs-utils`, `pfs-instdata`, `ics-cobraCharmer`, `ics-cobraOps`）、超新星分類には GitHub の `yasudank/RubinTargetSelection` を使用しています。

```bash
uv sync
```

### PFS ソフトウェアバージョンの更新 (`update-pfs-env`)

`NetflowByUser` の `config_2026Nov.yaml` のような設定ファイル（`pfs:` セクション）から Subaru PFS モジュールのバージョンを解決し、`pyproject.toml` と `uv` 仮想環境を一括更新できます。

```bash
# 変更内容の事前確認 (--dry-run)
uv run update-pfs-env /mnt/ugnas/work/PFS/galuda/NetflowByUser/config_2026Nov.yaml --dry-run

# pyproject.toml 更新 + uv sync 実行
uv run update-pfs-env /mnt/ugnas/work/PFS/galuda/NetflowByUser/config_2026Nov.yaml

# >= 指定の最新タグではなく下限タグそのもの (例: w.2026.39) に固定する場合
uv run update-pfs-env /mnt/ugnas/work/PFS/galuda/NetflowByUser/config_2026Nov.yaml --exact-floor
```

## 第1段階: `pfsDesign` 視野内の transient 取得 (`find-transients`)

```bash
# 単一ファイルを検索（--sky-reachable を付けると SKY コブラ到達可能天体のみに絞り込み）
uv run find-transients \
  /mnt/ugnas/work/PFS/spt_ssp_observation/runs/2026-11/pfs_designs/CO/pfsDesign-0x29ef57d63c7e0d25.fits \
  --start 2026-11-01 --stop 2026-11-19 \
  --exclude-sso --sky-reachable \
  -o transients.csv
```

## 第2段階: SKY ファイバー到達可能天体の超新星候補選定 (`select-sn-targets`)

以下の順で PostgreSQL DB を使わずにオンデマンド処理を行います：

1. `pfsDesign` の六角形視野内を Fink `/api/v1/conesearch` で検索（非 SSO）
2. `pfs_utils` / `ics_cobraOps` / `pfs_instdata` で **GOOD な SKY ファイバーのコブラ可動範囲内かつブラックドット外** の天体に絞り込み
3. Fink `/api/v1/sources` で全検出履歴を取得し、**2 晩以上観測されている天体** (`--min-nights 2` が既定) に限定
4. Fink `/api/v1/cutouts` から 30×30 の Science / Template / Difference カットアウトを並列取得してローカルキャッシュ（既存の `/mnt/ugnas/work/LSST/fink2/cutouts` があればシンボリックリンクで再利用）
5. `RubinTargetSelection` (`stamp_classifier`) の固定モデルバンドル (`snp_v1`) で Step 1 / moving 判定 / Step 2 分類を実行し、`design_sn_candidates.csv`, `candidates.csv`, `candidates.pdf` 等を出力

```bash
uv run select-sn-targets \
  /mnt/ugnas/work/PFS/spt_ssp_observation/runs/2026-11/pfs_designs/CO/pfsDesign-0x29ef57d63c7e0d25.fits \
  --start 2026-11-01 --stop 2026-11-19 \
  -o sn_output/run1
```

### 主なオプション (`select-sn-targets`)
- `--start` / `--stop`: 検索期間 (UTC)
- `--min-nights`: 最小観測夜数（既定 `2`）
- `--bundle-dir`: 固定モデルバンドルのパス（既定 `/mnt/ugnas/work/LSST/fink2/models/sn_pipeline/snp_v1`）
- `--cache-dir`: カットアウトのローカルキャッシュ先（既定 `cache/cutouts`）
- `--local-cutouts`: 既存ローカルカットアウト（既定 `/mnt/ugnas/work/LSST/fink2/cutouts`、空文字で無効化）
- `--obstime`: sky→PFI 座標変換に使う観測時刻 (UTC ISO)
- `--strict`: Step 2 が `uncertain` の天体を除外
- `--include-negative-galaxy`: Step 1 が `negative_real` かつ Step 2 が `galaxy` の天体も含める（既定は除外）
- `--no-pdf`: `candidates.pdf` の生成をスキップ

## 第3段階: `SKY` ファイバーの再割り当てと新 `pfsDesign` 書き出し (`reassign-pfs-design`)

`select-sn-targets` が出力した `design_sn_candidates.csv` を読み込み、各 `pfsDesign` について以下の安全チェックを通過した SN 候補だけを `SKY` ファイバーへ再割り当てし、新しい `pfsDesign-0x....fits`（`variant` インクリメント、`designId0` に元 ID 記録、`calculate_pfsDesignId` で ID 再計算）を出力します。

- コブラ可動範囲（`rMin <= r <= rMax`）およびブラックドット回避（`margin=1.65`）
- Fiducial 回避（`>= 2.9 mm`）
- 隣接割当コブラとの端点距離（`>= 2.1 mm`）
- `ics.cobraOps.CollisionSimulator` による **新規端点衝突 0 件 & 新規軌道衝突 0 件** の検証

```bash
uv run reassign-pfs-design \
  --candidates sn_output/co_2026-11/design_sn_candidates.csv \
  --design-dir /mnt/ugnas/work/PFS/spt_ssp_observation/runs/2026-11/pfs_designs/CO \
  -o modified_designs/co_2026-11
```
