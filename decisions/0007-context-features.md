# 0007. 直前のプレーは性質で 9組にまとめ、カウンターと一緒に、統計モデルと Gemini の両方に同じ形で渡す

- Status: Accepted
- Date: 2026-11-15

## Context
decisions/0006 の 3つ目の後半。距離・角度・頭か足かだけで比べたら、統計モデルと Gemini の当たり方の差は見分けられなかった。両方に、もっと広い情報（直前のプレーとカウンター）を足したときに、それぞれがどれだけよくなるか、差が出るかを見る。

普通のシュート（42,978本）の直前のプレー（同じチームの直前のイベントの `subEventName`）は 30種類あり、少ないもの（PK の 24本・ゴール率 0.667、オフサイドの 3本など）が多い。そのままでは統計モデルの係数が安定しないので、まとめ方を決める必要があった。

## Decision
- **まとめ方**: プレーの性質で 9組にする（2026-11-15。本数とゴール率は普通のシュート全部）

  | 組 | 元の種類 | 本数 | ゴール率 |
  | --- | --- | --- | --- |
  | `pass`（基準） | Simple pass・Hand pass・Throw in・Goal kick・Free Kick | 10,835 | 0.073 |
  | `through_ball` | Smart pass | 2,017 | 0.187 |
  | `cross` | Cross・Free kick cross | 6,448 | 0.152 |
  | `corner` | Corner | 1,517 | 0.060 |
  | `rebound` | Shot・Free kick shot・Penalty・Reflexes・Save attempt | 1,602 | 0.248 |
  | `lofted_pass` | High pass・Head pass・Launch | 1,863 | 0.127 |
  | `aerial_duel` | Air duel | 5,498 | 0.090 |
  | `dribble_duel` | Ground attacking duel・Ground loose ball duel・Ground defending duel・Acceleration・Touch | 13,146 | 0.088 |
  | `other` | 上にない種類（Ball out of the field・Clearance・Foul・Offside など）と、直前がないもの | 52 | — |

- **統計モデル**: decisions/0004・0005 と同じロジスティック回帰（numpy のニュートン法）・同じ試合単位の 5分割の交差検証に、組（基準の `pass` 以外の 8組を 1/0 の列）とカウンター（1/0）を足す。xG は `data/xg_stat_context.csv`
- **Gemini**: 同じ組の名前と、カウンターかどうかだけを足す。元の 30種類の名前は渡さない。渡す情報が違うと、「データから学んだか、常識で答えたか」ではなく、情報の違いが混ざるため。組の意味はプロンプトで 1行ずつ説明する。考える量・1回の本数・送り方は前の本番と同じ（LOW・50本）。答えは `data/llm_runs/test/low_context.jsonl`、確率は `data/xg_gemini_context.csv`（前の回のファイルは触らない）
- **比べ方**: 4つ（2つのモデル × 2つの情報）を、前と同じテストの 8,626本で並べる。差の 95% の幅は、試合単位のブートストラップ（decisions/0006 と同じ）で 3つ出す。統計モデルの広い − 3つ、Gemini の広い − 3つ、Gemini の広い − 統計モデルの広い

## Consequences
- 結果と読み方は `notes/llm-context.md`
- 前の回のプロンプトと送る JSON は変えていない（広い情報の項目は、ないときは JSON に入れない）。前の回の答えのファイルも変わっていない
- 選ばなかった案
  - 大きく 5組（係数は安定するが、コーナー 0.060 とクロス 0.152 のように率の違うものが混ざる）
  - 1,000本以上の 10種類をそのままにし、残りを「その他」（決め方は機械的だが、PK のあとのこぼれ球のように意味の近いものが「その他」に落ちる）
  - 直前のプレーから何秒たったか（`prev_seconds_before`）も足す（今回は組とカウンターだけにして、情報を足す効き目を先に見る）
