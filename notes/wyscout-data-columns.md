# Wyscout の公開データの項目（2026-11-04）

このリポジトリで使う Wyscout の公開データについて、全ファイルの項目を書き出したもの。

- 出典: Pappalardo, L., Cintia, P., Rossi, A. et al. "A public data set of spatio-temporal match events in soccer competitions." Scientific Data 6, 236 (2019)。データは figshare のコレクション（doi:10.6084/m9.figshare.c.4415000.v5、2020-01-28）
- ライセンス: 各ファイルとも CC BY 4.0（figshare の API で確かめた）
- 対象: 2017/18 シーズンの欧州 5大リーグ（イングランド・フランス・ドイツ・イタリア・スペイン）、2018 FIFA ワールドカップ、UEFA ユーロ 2016
- 作り方: 各ファイルの全件（events だけは 200件に 1件を抜き出して）から、項目の名前・型・例を自動で数えた。例は値の最初の 2つまで（40文字で切る）。選手・監督・審判の名前は例に出していない

## ファイル

| ファイル | 大きさ | 件数 | 中身 |
| --- | --- | --- | --- |
| `events.zip` | 77MB（展開すると 7ファイルで 912MB） | 3,251,294 | ボールに関わったプレーの記録。大会ごとに `events_<大会>.json` |
| `matches.zip` | 0.6MB | 1,941 | 試合（日付・会場・スコア・先発・交代・審判）。大会ごとに `matches_<大会>.json` |
| `competitions.json` | 1KB | 7 | 大会 |
| `teams.json` | 27KB | 142 | チーム |
| `players.json` | 1.7MB | 3,603 | 選手（身長・体重・利き足・ポジション） |
| `coaches.json` | 71KB | 208 | 監督 |
| `referees.json` | 197KB | 626（読めた分） | 審判。**配布元のファイルが途中で切れている**（figshare に載っている大きさと同じで、落とし方の問題ではない）。読めるところまでで 626件 |
| `playerank.json` | 6.2MB | 46,897 | 論文の著者が作った、試合ごとの選手の評価（PlayeRank） |
| `eventid2name.csv` | 1KB | — | イベントの種類の番号と名前の対応 |
| `tags2name.csv` | 2KB | — | タグの番号と名前の対応 |

## 読むときの注意

- 位置（`positions[].x`・`positions[].y`）: ピッチの縦と横を 0〜100 にした割合。`x` はボールを持つチームが攻める向きにそろっている（相手のゴール側が 100。シュートの `x` の中央値は 87）。メートルにするにはピッチの大きさを仮に置く（105m×68m ほど。推定）
- `positions` は 2つ（始まりと終わり）。1つしかないイベントが 741件ある。シュートの終わりの位置は意味のない値で、イングランドの 8,451本では `(0, 0)` が 4,706本、`(100, 100)` が 3,745本だった（すべてこのどちらか）。シュートは始まりの位置だけを使う
- `eventSec`: 前半・後半それぞれの始まりからの秒数（`matchPeriod` と組で使う）
- 直前・直後のイベントは、ファイルの並び（同じ `matchId` の中の順）でたどる。デュエルは両チームのイベントが交互に入るので、直前のイベントが相手チームのことが多い
- **xG の予測に使えない（シュートのあとに決まる）もの**: ゴールのどこに飛んだか（タグ 1201〜1223）、枠に入ったか（1801 Accurate・1802 Not accurate）、ブロックされたか（2101 Blocked）。答えを見ることになるので特徴にしない
- タグ 201 Opportunity（シュートの 30,458件に付く）は、記録する人が「決定機」とみなしたものと推定。打つ前の状況の判断か、結果を見た判断かは確かめていないので、使うなら先に確かめる
- 直接フリーキック（`Free kick shot`、2,209件）と PK（`Penalty`、658件）は、`Shot`（43,078件）とは別のイベント
- ID の型がファイルによって揺れる（`area.id` や `currentTeamId` が文字列のことと数値のことがある、`subEventId` が空文字のことがある）

## 項目

### competitions.json（7件）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `area.alpha2code` | str | 'IT' / 'ES' |
| `area.alpha3code` | str | 'ITA' / 'XEN' |
| `area.id` | str・int | '380' / '0' |
| `area.name` | str | 'Italy' / 'England' |
| `format` | str | 'Domestic league' / 'International cup' |
| `name` | str | 'Italian first division' / 'English first division' |
| `type` | str | 'club' / 'international' |
| `wyId` | int | 524 / 364 |

### teams.json（142件）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `area.alpha2code` | str | 'ES' / 'FR' |
| `area.alpha3code` | str | 'XEN' / 'ESP' |
| `area.id` | str・int | '0' / '724' |
| `area.name` | str | 'England' / 'Spain' |
| `city` | str | 'Newcastle upon Tyne' / 'Vigo' |
| `name` | str | 'Newcastle United' / 'Celta de Vigo' |
| `officialName` | str | 'Newcastle United FC' / 'Real Club Celta de Vigo' |
| `type` | str | 'club' / 'national' |
| `wyId` | int | 1613 / 692 |

### players.json（3,603件）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `birthArea.alpha2code` | str | 'TR' / 'FR' |
| `birthArea.alpha3code` | str | 'TUR' / 'FRA' |
| `birthArea.id` | str・int | '792' / '250' |
| `birthArea.name` | str | 'Turkey' / 'France' |
| `birthDate` | str | '1989-06-17' / '1999-01-23' |
| `currentNationalTeamId` | str・int | 4687 / 4423 |
| `currentTeamId` | int・None・str | 4502 / 3775 |
| `firstName` | str | （名） |
| `foot` | str | 'right' / 'left' |
| `height` | int | 187 / 182 |
| `lastName` | str | （姓） |
| `middleName` | str | （空のことが多い） |
| `passportArea.alpha2code` | str | 'TR' / 'SN' |
| `passportArea.alpha3code` | str | 'TUR' / 'SEN' |
| `passportArea.id` | str・int | '792' / '686' |
| `passportArea.name` | str | 'Turkey' / 'Senegal' |
| `role.code2` | str | 'GK' / 'DF' |
| `role.code3` | str | 'GKP' / 'DEF' |
| `role.name` | str | 'Goalkeeper' / 'Defender' |
| `shortName` | str | （「名の頭文字. 姓」の形） |
| `weight` | int | 78 / 73 |
| `wyId` | int | 32777 / 393228 |

### coaches.json（208件）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `birthArea.alpha2code` | str | 'BR' / 'DE' |
| `birthArea.alpha3code` | str | 'BRA' / 'DEU' |
| `birthArea.id` | int | 76 / 276 |
| `birthArea.name` | str | 'Brazil' / 'Germany' |
| `birthDate` | str・None | '1945-05-09' / '1956-06-16' |
| `currentTeamId` | int | 0 / 674 |
| `firstName` | str | （名） |
| `lastName` | str | （姓） |
| `middleName` | str | （空のことが多い） |
| `passportArea.alpha2code` | str | 'BR' / 'DE' |
| `passportArea.alpha3code` | str | 'BRA' / 'DEU' |
| `passportArea.id` | int | 76 / 276 |
| `passportArea.name` | str | 'Brazil' / 'Germany' |
| `shortName` | str | （短い名前） |
| `wyId` | int | 275580 / 14710 |

### referees.json（読めた 626件。配布元のファイルが途中で切れている）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `birthArea.alpha2code` | str | 'CO' / 'EN' |
| `birthArea.alpha3code` | str | 'COL' / 'XEN' |
| `birthArea.id` | int | 170 / 826 |
| `birthArea.name` | str | 'Colombia' / 'England' |
| `birthDate` | str・None | '1985-01-28' / '1977-12-20' |
| `firstName` | str | （名） |
| `lastName` | str | （姓） |
| `middleName` | str | （空のことが多い） |
| `passportArea.alpha2code` | str・None | 'CO' / 'EN' |
| `passportArea.alpha3code` | str・None | 'COL' / 'XEN' |
| `passportArea.id` | int | 170 / 826 |
| `passportArea.name` | str・None | 'Colombia' / 'England' |
| `shortName` | str | （短い名前） |
| `wyId` | int | 378217 / 385011 |

### playerank.json（46,897件）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `goalScored` | int | 0 / 1 |
| `matchId` | int | 2057991 / 2057992 |
| `minutesPlayed` | int | 90 / 41 |
| `playerId` | int | 10014 / 100140 |
| `playerankScore` | float | 0.0053 / 0.0009 |
| `roleCluster` | str | 'right CB' / 'central MF' |

### matches/matches_<大会>.json（1,941件）

`teamsData` のキーはチームの ID（下では `{id}` とまとめた）。

| 項目 | 型 | 例 |
| --- | --- | --- |
| `competitionId` | int | 364 / 102 |
| `date` | str | 'May 13, 2018 at 4:00:00 PM GMT+2' |
| `dateutc` | str | '2018-05-13 14:00:00' |
| `duration` | str | 'Regular' / 'ExtraTime' |
| `gameweek` | int | 38 / 31 |
| `groupName` | str | 'Group F'（国際大会だけ） |
| `label` | str | 'Burnley - AFC Bournemouth, 1 - 2' |
| `referees` | list（要素数 0〜6） | |
| `referees[].refereeId` | int | 385705 / 385733 |
| `referees[].role` | str | 'referee' / 'firstAssistant' |
| `roundId` | int | 4405654 / 4165474 |
| `seasonId` | int | 181150 / 9291 |
| `status` | str | 'Played' |
| `teamsData.{id}.coachId` | int | 8880 / 8934 |
| `teamsData.{id}.formation.bench` | list | 控えの選手（`playerId`・`ownGoals` など） |
| `teamsData.{id}.formation.lineup` | list | 先発の選手（`playerId`・`ownGoals` など） |
| `teamsData.{id}.formation.substitutions` | list・str | 交代（`playerIn`・`playerOut` など） |
| `teamsData.{id}.hasFormation` | int | 1 |
| `teamsData.{id}.score` | int | 1 / 2 |
| `teamsData.{id}.scoreET` | int | 0 / 1 |
| `teamsData.{id}.scoreHT` | int | 1 / 0 |
| `teamsData.{id}.scoreP` | int | 0 / 6 |
| `teamsData.{id}.side` | str | 'home' / 'away' |
| `teamsData.{id}.teamId` | int | 1646 / 1659 |
| `venue` | str | 'Turf Moor' / 'Selhurst Park' |
| `winner` | int | 1659 / 1628 |
| `wyId` | int | 2500089 / 2500090 |

### events/events_<大会>.json（3,251,294件。型は 200件に 1件を抜き出して見た）

| 項目 | 型 | 例 |
| --- | --- | --- |
| `eventId` | int | 8 / 1 |
| `eventName` | str | 'Pass' / 'Duel' |
| `eventSec` | float | 2.7586489999999912 / 702.436987 |
| `id` | int | 177959171 / 177959396 |
| `matchId` | int | 2499719 / 2499720 |
| `matchPeriod` | str | '1H' / '2H' |
| `playerId` | int | 25413 / 14853 |
| `positions` | list（要素数 1〜2） | |
| `positions[].x` | int | 49 / 31 |
| `positions[].y` | int | 49 / 78 |
| `subEventId` | int・str | 85 / 12 |
| `subEventName` | str | 'Simple pass' / 'Ground defending duel' |
| `tags` | list（要素数 0〜6） | |
| `tags[].id` | int | 1801 / 702 |
| `teamId` | int | 1609 / 1631 |

### イベントの種類ごとの件数（7大会の合計）

| eventName | subEventName | 件数 |
| --- | --- | --- |
| Pass | Simple pass | 1,286,004 |
| Pass | High pass | 130,215 |
| Pass | Head pass | 96,911 |
| Pass | Cross | 62,333 |
| Pass | Launch | 45,738 |
| Pass | Smart pass | 30,303 |
| Pass | Hand pass | 14,004 |
| Duel | Ground attacking duel | 279,991 |
| Duel | Ground defending duel | 278,703 |
| Duel | Air duel | 168,407 |
| Duel | Ground loose ball duel | 151,982 |
| Others on the ball | Touch | 174,535 |
| Others on the ball | Clearance | 56,819 |
| Others on the ball | Acceleration | 25,886 |
| Free Kick | Throw in | 85,080 |
| Free Kick | Free Kick | 45,566 |
| Free Kick | Goal kick | 31,797 |
| Free Kick | Corner | 19,316 |
| Free Kick | Free kick cross | 8,647 |
| Free Kick | Free kick shot | 2,209 |
| Free Kick | Penalty | 658 |
| Interruption | Ball out of the field | 129,180 |
| Interruption | Whistle | 917 |
| Foul | Foul | 47,143 |
| Foul | Hand foul | 2,078 |
| Foul | Protest | 598 |
| Foul | Out of game foul | 499 |
| Foul | Late card foul | 317 |
| Foul | Time lost foul | 221 |
| Foul | Simulation | 110 |
| Foul | Violent Foul | 83 |
| Shot | Shot | 43,078 |
| Save attempt | Reflexes | 10,805 |
| Save attempt | Save attempt | 6,814 |
| Offside | （なし） | 8,182 |
| Goalkeeper leaving line | Goalkeeper leaving line | 6,165 |

### イベントの種類ごとによく付くタグ（上位 8）

タグの番号と名前の対応の全部は `tags2name.csv`（59個）。

| eventName | タグ（件数） |
| --- | --- |
| Pass | 1801 Accurate（1,381,487）, 1802 Not accurate（284,019）, 1401 Interception（67,970）, 801 High（35,299）, 402 Right foot（34,944）, 901 Through（30,258）, 401 Left foot（27,300）, 1901 Counter attack（26,130） |
| Duel | 1801 Accurate（538,211）, 1802 Not accurate（340,872）, 701 Lost（339,783）, 703 Won（338,879）, 702 Neutral（199,401）, 502 Free space left（63,173）, 501 Free space right（63,104）, 503 Take on left（52,599） |
| Others on the ball | 1401 Interception（98,580）, 1802 Not accurate（45,858）, 1801 Accurate（35,916）, 1302 Missed ball（12,695）, 1901 Counter attack（8,352）, 1301 Feint（1,986）, 201 Opportunity（1,097）, 2001 Dangerous ball lost（526） |
| Free Kick | 1801 Accurate（133,836）, 1802 Not accurate（27,640）, 801 High（23,270）, 302 Key pass（2,634）, 1101 Direct（2,119）, 402 Right foot（1,879）, 201 Opportunity（1,506）, 401 Left foot（986） |
| Interruption | 1901 Counter attack（2） |
| Foul | 1702 Yellow card（7,438）, 1701 Red card（174）, 1703 Second yellow card（164）, 1901 Counter attack（48） |
| Shot | 201 Opportunity（30,458）, 1802 Not accurate（27,758）, 402 Right foot（21,961）, 1801 Accurate（15,320）, 401 Left foot（14,130）, 2101 Blocked（10,222）, 403 Head/body（6,987）, 101 Goal（4,492） |
| Save attempt | 1801 Accurate（12,338）, 101 Goal（5,281）, 1802 Not accurate（5,281）, 1201 Position: Goal low center（4,269）, 1203 Position: Goal center（2,627）, 1205 Position: Goal low left（2,593）, 1202 Position: Goal low right（2,265）, 1204 Position: Goal center left（1,462） |
| Offside | 1901 Counter attack（153） |
| Goalkeeper leaving line | 1901 Counter attack（265） |
