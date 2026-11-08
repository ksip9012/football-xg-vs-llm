# football-xg-vs-llm

サッカーのシュートのゴール期待値（xG）を統計モデルで作り、LLM の予測と比べる（個人プロジェクト Step3）。

## セットアップ

[mise](https://mise.jdx.dev/) で Python と uv の版をそろえ、uv で依存を入れる（版は `mise.toml` と `uv.lock` で固定）。

```bash
mise trust
mise install
uv sync
uv run pre-commit install   # コミットの前に mise run check を動かす
mise run fetch              # 元データを落とし、sha256 を確かめる（86MB ほど）
mise run shots              # シュートの表 data/shots.csv を作り直す（リポジトリにも入っている）
```

## シュートの表

`data/shots.csv` は、元データからシュートを 1行 1本に抜き出した表（45,945本。普通のシュート・直接フリーキック・PK）。打った位置・体の部位・カウンターか・同じチームの直前のイベントと、ゴールになったかを持つ。シュートのあとに決まる情報（枠内か・ブロックか・ゴールのどこか）は入れていない。列の説明と、作るときに数えたことは [notes/shots-table.md](./notes/shots-table.md)。

## テストの実行方法

```bash
mise run check   # ruff check・ruff format --check・pyright・pytest（CI と同じ）
mise run fix     # ruff の自動修正と format
```

## 元データと出典

- 出典: Pappalardo, L., Cintia, P., Rossi, A. et al. "[A public data set of spatio-temporal match events in soccer competitions](https://doi.org/10.1038/s41597-019-0247-7)." *Scientific Data* 6, 236 (2019)
- データ: figshare の「[Soccer match event dataset](https://doi.org/10.6084/m9.figshare.c.4415000.v5)」（v5）。Wyscout の記録した、2017/18 シーズンの欧州 5大リーグ、2018 FIFA ワールドカップ、UEFA ユーロ 2016 の試合のイベント
- ライセンス: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 置き場所（[decisions/0001](./decisions/0001-source-data-placement.md)）
  - 元データはすべて `mise run fetch` で `data/raw/` に落とす（リポジトリには入れない）。zip は展開せずに読む
  - 一覧（URL と sha256）: `data/sources.toml`
- 項目の一覧と読むときの注意: [notes/wyscout-data-columns.md](./notes/wyscout-data-columns.md)
- このリポジトリで作る表は、上の元データを加工して作成したもの

## 技術選定理由

- [0001](./decisions/0001-source-data-placement.md): 元データはすべて落として使い、zip は展開せずに読む
- [0002](./decisions/0002-shots-table.md): シュートの表は 3種類を 1つの CSV にし、シュートのあとに決まる情報とタグ 201 は入れない
