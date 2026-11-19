# RubinTransient

Subaru/PFS の `pfsDesign` FITS ファイルの視野内にある Rubin/LSST transient を [Fink/LSST object API](https://api.lsst.fink-portal.org/) から検索するツールです。

## セットアップ (`uv`)

`UV_PROJECT_ENVIRONMENT=.venv-$(uname -m)` の設定に従って仮想環境が作成されます。
`pfsDesign` ファイルの読み書きには Subaru PFS Project の `pfs-datamodel` (`Subaru-PFS/datamodel@w.2026.39`) を使用しています。

```bash
uv sync
```

## 第1段階: `pfsDesign` 視野内の transient 取得

単一の `pfsDesign` ファイル、複数ファイル、またはディレクトリを指定して実行できます。

```bash
# 単一ファイルを検索して CSV に保存
uv run find-transients \
  /mnt/ugnas/work/PFS/spt_ssp_observation/runs/2026-11/pfs_designs/CO/pfsDesign-0x0ba54120d2be15a4.fits \
  --start 2026-11-01 --stop 2026-11-19 \
  -o transients.csv

# ディレクトリ内の全 pfsDesign をスキャン（SSO 除外、サマリーも出力）
uv run find-transients \
  /mnt/ugnas/work/PFS/spt_ssp_observation/runs/2026-11/pfs_designs/CO \
  --start 2026-11-01 --stop 2026-11-19 \
  --exclude-sso \
  -o transients_co.csv \
  --summary-output summary_co.csv
```

`uv run python find_transients.py ...` でも同様に実行できます。

### 主なオプション

- `--start` / `--startdate`: 検索開始日時 (UTC: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, MJD, JD)
- `--stop` / `--stopdate`: 検索終了日時 (UTC。日付のみの場合は当日 `23:59:59` まで含む。省略時は現在時刻)
- `--window`: `--start` からの日数 (`--stop` の代わりに使用)
- `--kind`: `across` (既定: 期間中に観測された全天体) / `within` (初回・最終検出とも期間内の天体のみ)
- `--fov-margin`: ファイバー凸包からの外向きマージン（秒角、既定 `100.0` ≈ コブラ可動範囲）
- `--radius`: コーンサーチ半径（秒角）を手動指定（既定: 最外周ファイバー + `--fov-margin`）
- `--cone-only`: 六角形視野判定を行わず、コーン内すべてを出力
- `--exclude-sso`: 既知太陽系天体 (`f:is_sso == True`) を除外
- `--columns` / `--all-columns`: 取得する Fink カラムを指定（既定は厳選28カラム + AB 等級 `psf_mag`, `science_mag`）

