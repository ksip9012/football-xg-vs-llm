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

## 最初の xG

ゴールまでの距離と、両ポストを見込む角度（と、頭か体か）のロジスティック回帰。numpy で自分で書いたもの・scikit-learn・statsmodels の 3通りで作り、同じ答えになることを確かめた。平均の率だけを出すモデルと比べて、テストの AUC は 0.50 → 0.78（頭か体かを足すと 0.79）。

```bash
mise run xg   # 3通りの係数・物差し・時間を Markdown の表で出す
```

結果と読み方は [notes/xg-baseline.md](./notes/xg-baseline.md)。

## 選手・チーム・大会ごとの集計

試合単位の交差検証で、どのシュートにも「その試合を学習に使っていないモデル」の xG を付け（`data/xg_stat.csv`）、選手・チーム・大会ごとに集計した（`data/aggregates/`。PK・直接フリーキックを除く）。試合を 2つに分けて比べると、打つ場所のよさ（1本あたりの xG）は相関 0.70 で続くが、決定力（1本あたりのゴール − xG）は 0.14 しか続かず、選手ごとの差の多くは偶然と読める。

```bash
mise run xg-cv       # シュートごとの xG（5分割の交差検証）
mise run aggregate   # 集計の CSV と Markdown の表、決定力の確かめ
mise run figures     # モデルを確かめる図と集計の図（notes/figures/）
```

結果・図・読み方は [notes/xg-aggregates.md](./notes/xg-aggregates.md)。

## LLM との比べ（進行中）

同じシュートを Gemini 3.8 Flash（Vertex AI）に見せ、統計モデルと同じ情報（距離・ゴールの見える角度・頭か足か）だけで、ゴールになる確率を答えさせる。選手の名前・チーム・試合の日付は渡さない。

テストの 8,626本（388試合）では、**当たり方の差は見分けられなかった**（AUC は統計モデル 0.7945・Gemini 0.7938、差の 95% の幅 −0.0043〜+0.0026）。答え方もよく似ている（確率の相関 0.979）。ただし Gemini は確率を低めに出す（平均 0.092、実際の率 0.104）。考える量（thinking）を増やしても良くならなかった（500本の試し）。

両方に**直前のプレー（9組にまとめたもの）とカウンター**を足すと、統計モデルは少しよくなり（ログ損失 0.2765 → 0.2747、差の 95% の幅 −0.0037〜−0.0000）、Gemini はよくならなかった（0.2773 → 0.2776）。広い情報どうしでは、統計モデルのほうがよい（ログ損失の差の幅 +0.0005〜+0.0054。AUC の差は見分けられない）。Gemini は、スルーパスやカウンターで確率を上げ、コーナーで下げるなど向きは合っているが、大きさが合わず、低めに出すくせも強くなった（平均 0.087）。

Gemini の確率の**全体の高さだけを直す**（テストの中で試合単位の 5分割で学び、別の試合に当てる）と、統計モデルとの差は見分けられなくなった（広い情報どうしで、ログ損失の差 +0.0009、幅 −0.0009〜+0.0027）。当てはめた傾きはほぼ 1 で、ずれは高さだけだった。ただし直した Gemini は、ほかの試合の結果から 1〜2つの数を学んでいる。

**Claude Opus 5.5** も、テストから試合単位で選んだ 2,004本で、同じ説明・同じ情報で答えさせた（Anthropic の API、考える量 low、費用 $2.21）。3つの情報では Gemini と見分けられず（確率の相関 0.979）、広い情報では Claude のほうがよかった（ログ損失の差の幅 −0.0096〜−0.0004）。広い情報の Claude と統計モデルは見分けられない。Claude も確率を低めに出す。

```bash
mise run llm-trial     # 500本の試し（data/llm_runs/trial/ に答えがあれば API は呼ばない）
mise run llm-test      # テストの 8,626本（data/llm_runs/test/。同上）
mise run compare-llm   # 物差し・差の 95% の幅・図
mise run llm-test-context  # 直前のプレーとカウンターも渡して、テストの 8,626本
mise run compare-context   # 3つの情報と広い情報の 4つを比べる
mise run recalibrate       # Gemini の低めに出すくせを直して比べる（API は呼ばない）
mise run llm-claude        # 2,004本を Claude Opus 5.5 に（data/llm_runs/claude/。料金がかかる）
mise run llm-claude-context
mise run compare-claude    # 同じ 2,004本で Claude・Gemini・統計モデルを比べる
```

Vertex AI を使うので、手元の `mise.local.toml`（git の対象外）に `GOOGLE_CLOUD_PROJECT` と `GOOGLE_CLOUD_QUOTA_PROJECT` を書き、`gcloud auth application-default login` でログインしておく。結果は [notes/llm-trial.md](./notes/llm-trial.md)（試し）と [notes/llm-test.md](./notes/llm-test.md)（本番）、[notes/llm-context.md](./notes/llm-context.md)（広い情報）、[notes/llm-recalibrate.md](./notes/llm-recalibrate.md)（くせを直す）、[notes/llm-claude.md](./notes/llm-claude.md)（Claude）。Claude は、手元の `mise.local.toml` に `ANTHROPIC_API_KEY` を書いておく。

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
- [0003](./decisions/0003-pydantic-models.md): 元データのイベントとシュートの表の 1行を pydantic のモデルにして確かめる
- [0004](./decisions/0004-first-xg-three-ways.md): 最初の xG は距離と角度のロジスティック回帰にし、numpy・scikit-learn・statsmodels の 3通りで作って比べる
- [0005](./decisions/0005-cv-aggregates-figures.md): xG は試合単位の交差検証で出して `shot_id, xg` の CSV に置き、選手・チーム・大会ごとに集計して、図は英語のラベルの PNG にする
- [0006](./decisions/0006-compare-with-llm.md): LLM には統計モデルと同じ情報だけを渡して確率を答えさせ、段階を踏んで本数を増やし、同じ物差しと差の幅で比べる
- [0007](./decisions/0007-context-features.md): 直前のプレーは性質で 9組にまとめ、カウンターと一緒に、統計モデルと Gemini の両方に同じ形で渡す
- [0008](./decisions/0008-recalibrate-gemini.md): Gemini の確率は、テストの中で試合単位の 5分割で高さと傾きを直し、高さだけの直しと並べて、統計モデルと比べる
- [0009](./decisions/0009-add-claude.md): Claude は Opus 5.5 を Anthropic の API で呼び、テストから試合単位で選んだ 2,000本ほどで、Gemini と同じ説明・同じ情報で答えさせる
