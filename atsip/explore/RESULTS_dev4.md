# dev4：再始動の試行と検証（E11）

## 結論

13クラス × seed 10–14 × 7規則の455試行を、各入力30秒で実行した。全65入力で誤作動・見逃し・発散がすべて0件だった規則は **K16・B・abort 4、K16・B・abort 2、K16・M・abort 4**。
性能差はクラス内で同じ seed の差を平均した。正の値は悪化、負の値は改善を表す。絶対値が0.1 dB未満の差は「差なし」とした。ρ は線形比。

| 規則 | 誤作動 | 見逃し | 発散 |
| --- | --- | --- | --- |
| 再始動なし | 0 | 20 | 0 |
| 検証なし | 35 | 0 | 5 |
| K16・B | 0 | 0 | 5 |
| K16・B・abort 4 | 0 | 0 | 0 |
| K16・B・abort 2 | 0 | 0 | 0 |
| K12・B・abort 4 | 0 | 1 | 0 |
| K16・M・abort 4 | 0 | 0 | 0 |

再始動なしでは検出器を使わないため、自動集計で空欄となる見逃しを、再始動が行われなかった経路変化20件として数えた。

### SC（入力の色の変化）

取り消し欄は seed 10、11、12、13、14 の順。各セルの「+n / ρ」は検出から取り消しまでのブロック数と、そのブロックで設定窓から求めた比を表す。

| 規則 | 取り消し（+ブロック / ρ） | 直後2秒 L1 差 | 直後2秒 Φ 超過差 |
| --- | --- | --- | --- |
| 再始動なし | なし、なし、なし、なし、なし | 差なし | 差なし |
| 検証なし | 発散、発散、発散、発散、発散 | 集計不可 | 集計不可 |
| K16・B | 発散、発散、発散、発散、発散 | 集計不可 | 集計不可 |
| K16・B・abort 4 | +3 / 13.4515、+2 / 5.6881、+2 / 20.1311、+2 / 9.19746、+2 / 18.3925 | +0.10 dB | +0.12 dB |
| K16・B・abort 2 | +3 / 13.4515、+2 / 5.6881、+2 / 20.1311、+2 / 9.19746、+2 / 18.3925 | +0.10 dB | +0.12 dB |
| K12・B・abort 4 | +3 / 13.4515、+2 / 5.6881、+2 / 20.1311、+2 / 9.19746、+2 / 18.3925 | +0.10 dB | +0.12 dB |
| K16・M・abort 4 | +3 / 7.66414、+2 / 4.61123、+3 / 6.91537、+3 / 72.3687、+3 / 24.8828 | +0.23 dB | +0.60 dB |

### EN（誤差側の外乱）

再始動なしに対する直後2秒の差と、各クラス10外乱に対する早期取り消し件数を示す。

| 規則 | クラス | 早期取り消し | L1 差 | Φ 超過差 |
| --- | --- | --- | --- | --- |
| 検証なし | ENm20 | 0 | +1.26 dB | +10.92 dB |
| 検証なし | ENm10 | 0 | +3.33 dB | +11.95 dB |
| 検証なし | EN0 | 0 | +5.72 dB | +12.07 dB |
| K16・B | ENm20 | 0 | 差なし | +4.47 dB |
| K16・B | ENm10 | 0 | 差なし | +5.20 dB |
| K16・B | EN0 | 0 | -0.17 dB | +5.29 dB |
| K16・B・abort 4 | ENm20 | 0 | 差なし | +4.47 dB |
| K16・B・abort 4 | ENm10 | 0 | 差なし | +5.20 dB |
| K16・B・abort 4 | EN0 | 0 | -0.17 dB | +5.29 dB |
| K16・B・abort 2 | ENm20 | 0 | 差なし | +4.47 dB |
| K16・B・abort 2 | ENm10 | 0 | 差なし | +5.20 dB |
| K16・B・abort 2 | EN0 | 0 | -0.17 dB | +5.29 dB |
| K12・B・abort 4 | ENm20 | 0 | 差なし | +3.38 dB |
| K12・B・abort 4 | ENm10 | 0 | 差なし | +3.99 dB |
| K12・B・abort 4 | EN0 | 0 | -0.13 dB | +4.06 dB |
| K16・M・abort 4 | ENm20 | 0 | 差なし | +4.47 dB |
| K16・M・abort 4 | ENm10 | 0 | 差なし | +5.20 dB |
| K16・M・abort 4 | EN0 | 0 | -0.17 dB | +5.29 dB |

### 経路変化（pc、pcb、EVpc）

直後2秒の L1 は検証なしとの比較。pcb は10秒と20秒のイベント窓を平均した。

| 規則 | pc 確定・L1 差 | pcb 確定・L1 差 | EVpc 確定・L1 差 | 早期取り消し |
| --- | --- | --- | --- | --- |
| 再始動なし | 0/5、+18.34 dB（完全一致 0/5） | 0/10、+17.71 dB（完全一致 0/5） | 0/5、+2.65 dB（完全一致 0/5） | 0 |
| 検証なし | 5/5、差なし（完全一致 5/5） | 10/10、差なし（完全一致 5/5） | 5/5、差なし（完全一致 5/5） | 0 |
| K16・B | 5/5、差なし（完全一致 5/5） | 10/10、差なし（完全一致 5/5） | 5/5、差なし（完全一致 5/5） | 0 |
| K16・B・abort 4 | 5/5、差なし（完全一致 5/5） | 10/10、差なし（完全一致 5/5） | 5/5、差なし（完全一致 5/5） | 0 |
| K16・B・abort 2 | 5/5、差なし（完全一致 5/5） | 10/10、差なし（完全一致 5/5） | 5/5、差なし（完全一致 5/5） | 0 |
| K12・B・abort 4 | 5/5、差なし（完全一致 5/5） | 9/10、+1.80 dB（完全一致 4/5） | 5/5、差なし（完全一致 5/5） | 0 |
| K16・M・abort 4 | 5/5、差なし（完全一致 5/5） | 10/10、差なし（完全一致 5/5） | 5/5、差なし（完全一致 5/5） | 0 |

### 変化なしの6クラス

G、BU、PS、II300、a145、a120 の30入力で、全7規則の記録された性能指標と試行記録が検証なしと完全一致した。

### K=12 と K=16

窓 B・abort 4 で比較した。経路変化の最大と EN 外乱の最小は、最終判定に進んだ試行の ρ。SC は早期取り消し時の ρ を上の表に示した。

| 規則 | 経路変化の最大 ρ | EN 外乱の最小 ρ | 見逃し |
| --- | --- | --- | --- |
| K12・B・abort 4 | 0.73118（20件） | 1.22655（30件） | 1 |
| K16・B・abort 4 | 0.116639（20件） | 1.16962（30件） | 0 |

- K12・B・abort 4 の pcb_s13 は検出から 12 ブロック後の ρ が 0.73118 で、θ=0.5 を超えて取り消された。

ab2 / ab4 は、検出の 2 ブロック後から最終判定の前まで、設定窓の ρ が 2 / 4 を超えるか非有限ならその場で取り消す。
表は自動生成。値は dB（小さいほど良い）。クラス内で seed 平均、クラス間で等重み平均。件数は合計。
規則名：ver- は検証なし（検出したら必ず再始動）、verK/0.5 は検出の K ブロック後に ρ = E(w)/E(w0) ≤ 0.5 を確かめ、wB / w2B は ρ を最新の B / 2B サンプルだけで測る（無印は直近 M サンプル）。満たさなければ係数を w0 に戻す。nors は再始動なし。false_rs / missed_rs は確定した再始動だけを数える。

## 試行ごとの ρ（dB）

| 規則 | クラス | 種別 | 件数 | 最小 | 中央 | 最大 | 確定 |
|---|---|---|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | それ以外 | 10 | 0.77 | 1.88 | 2.20 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | それ以外 | 10 | 0.77 | 1.88 | 2.20 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | それ以外 | 10 | 0.77 | 1.88 | 2.20 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | それ以外 | 10 | 1.25 | 1.44 | 2.19 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | それ以外 | 10 | -1.03 | -0.78 | -0.53 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | それ以外 | 10 | 0.75 | 1.81 | 2.18 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | それ以外 | 10 | 0.75 | 1.81 | 2.18 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | それ以外 | 10 | 0.75 | 1.81 | 2.18 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | それ以外 | 10 | 1.26 | 1.46 | 2.17 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | それ以外 | 10 | -1.02 | -0.79 | -0.51 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | それ以外 | 10 | 0.68 | 1.47 | 2.23 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | それ以外 | 10 | 0.68 | 1.47 | 2.23 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | それ以外 | 10 | 0.68 | 1.47 | 2.23 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | それ以外 | 10 | 0.89 | 1.58 | 2.09 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | それ以外 | 10 | -1.04 | -0.83 | -0.45 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 経路変化 | 5 | -11.88 | -10.92 | -9.33 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 経路変化 | 5 | -11.88 | -10.92 | -9.33 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 経路変化 | 5 | -11.88 | -10.92 | -9.33 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 経路変化 | 5 | -9.60 | -9.47 | -8.89 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 経路変化 | 5 | -13.40 | -12.98 | -12.34 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | それ以外 | 5 | 7.55 | 11.29 | 13.04 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | それ以外 | 5 | 7.55 | 11.29 | 13.04 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | それ以外 | 5 | 7.55 | 11.29 | 13.04 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | それ以外 | 5 | 6.64 | 8.84 | 18.60 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 経路変化 | 5 | -25.65 | -17.82 | -15.71 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 経路変化 | 5 | -25.65 | -17.82 | -15.71 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 経路変化 | 5 | -25.65 | -17.82 | -15.71 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 経路変化 | 5 | -18.34 | -13.76 | -9.09 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 経路変化 | 5 | -26.30 | -18.04 | -15.77 | 5 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 経路変化 | 10 | -20.24 | -17.06 | -12.61 | 10 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 経路変化 | 10 | -20.24 | -17.06 | -12.61 | 10 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 経路変化 | 10 | -20.24 | -17.06 | -12.61 | 10 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 経路変化 | 10 | -15.81 | -13.58 | -1.36 | 9 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 経路変化 | 10 | -24.60 | -21.19 | -12.62 | 10 |

## 窓ごとの ρ（線形比、最終判定に到達した試行）

| 規則 | クラス | seed | 検出ブロック | 判定ブロック | 結果 | ρ_M | ρ_2B | ρ_B |
|---|---|---:|---:|---:|---|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 10 | 1251 | 1267 | rollback | 0.823583 | 1.13753 | 1.42131 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 10 | 2501 | 2517 | rollback | 0.88512 | 1.23178 | 1.63895 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 10 | 1251 | 1267 | rollback | 0.823583 | 1.13753 | 1.42131 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 10 | 2501 | 2517 | rollback | 0.88512 | 1.23178 | 1.63895 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 10 | 1251 | 1267 | rollback | 0.823583 | 1.13753 | 1.42131 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 10 | 2501 | 2517 | rollback | 0.88512 | 1.23178 | 1.63895 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 10 | 1251 | 1263 | rollback | 0.866382 | 1.18401 | 1.44733 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 10 | 2501 | 2513 | rollback | 0.785204 | 1.04123 | 1.39274 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 10 | 1251 | 1267 | rollback | 0.823583 | 1.13753 | 1.42131 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 10 | 2501 | 2517 | rollback | 0.88512 | 1.23178 | 1.63895 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 11 | 1251 | 1267 | rollback | 0.789379 | 1.06771 | 1.39704 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 11 | 2501 | 2517 | rollback | 0.853305 | 1.13224 | 1.52153 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 11 | 1251 | 1267 | rollback | 0.789379 | 1.06771 | 1.39704 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 11 | 2501 | 2517 | rollback | 0.853305 | 1.13224 | 1.52153 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 11 | 1251 | 1267 | rollback | 0.789379 | 1.06771 | 1.39704 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 11 | 2501 | 2517 | rollback | 0.853305 | 1.13224 | 1.52153 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 11 | 1251 | 1263 | rollback | 0.763255 | 1.03033 | 1.33765 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 11 | 2501 | 2513 | rollback | 0.81169 | 1.17575 | 1.63388 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 11 | 1251 | 1267 | rollback | 0.789379 | 1.06771 | 1.39704 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 11 | 2501 | 2517 | rollback | 0.853305 | 1.13224 | 1.52153 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 12 | 1251 | 1267 | rollback | 0.799515 | 0.977101 | 1.1953 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 12 | 2501 | 2517 | rollback | 0.811689 | 1.17718 | 1.66037 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 12 | 1251 | 1267 | rollback | 0.799515 | 0.977101 | 1.1953 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 12 | 2501 | 2517 | rollback | 0.811689 | 1.17718 | 1.66037 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 12 | 1251 | 1267 | rollback | 0.799515 | 0.977101 | 1.1953 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 12 | 2501 | 2517 | rollback | 0.811689 | 1.17718 | 1.66037 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 12 | 1251 | 1263 | rollback | 0.76712 | 1.02124 | 1.3936 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 12 | 2501 | 2513 | rollback | 0.808418 | 1.11563 | 1.35621 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 12 | 1251 | 1267 | rollback | 0.799515 | 0.977101 | 1.1953 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 12 | 2501 | 2517 | rollback | 0.811689 | 1.17718 | 1.66037 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 13 | 1251 | 1267 | rollback | 0.804131 | 1.13921 | 1.3218 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 13 | 2501 | 2517 | rollback | 0.848263 | 1.22187 | 1.59706 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 13 | 1251 | 1267 | rollback | 0.804131 | 1.13921 | 1.3218 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 13 | 2501 | 2517 | rollback | 0.848263 | 1.22187 | 1.59706 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 13 | 1251 | 1267 | rollback | 0.804131 | 1.13921 | 1.3218 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 13 | 2501 | 2517 | rollback | 0.848263 | 1.22187 | 1.59706 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 13 | 1251 | 1263 | rollback | 0.805542 | 1.12023 | 1.33478 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 13 | 2501 | 2513 | rollback | 0.882455 | 1.17354 | 1.57678 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 13 | 1251 | 1267 | rollback | 0.804131 | 1.13921 | 1.3218 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 13 | 2501 | 2517 | rollback | 0.848263 | 1.22187 | 1.59706 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 14 | 1251 | 1267 | rollback | 0.885183 | 1.15267 | 1.57812 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EN0 | 14 | 2501 | 2517 | rollback | 0.860872 | 1.19899 | 1.56389 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 14 | 1251 | 1267 | rollback | 0.885183 | 1.15267 | 1.57812 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EN0 | 14 | 2501 | 2517 | rollback | 0.860872 | 1.19899 | 1.56389 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 14 | 1251 | 1267 | rollback | 0.885183 | 1.15267 | 1.57812 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EN0 | 14 | 2501 | 2517 | rollback | 0.860872 | 1.19899 | 1.56389 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 14 | 1251 | 1263 | rollback | 0.834897 | 1.27855 | 1.65455 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EN0 | 14 | 2501 | 2513 | rollback | 0.817226 | 1.08826 | 1.37629 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 14 | 1251 | 1267 | rollback | 0.885183 | 1.15267 | 1.57812 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EN0 | 14 | 2501 | 2517 | rollback | 0.860872 | 1.19899 | 1.56389 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 10 | 1251 | 1267 | rollback | 0.822865 | 1.12078 | 1.40808 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 10 | 2501 | 2517 | rollback | 0.887558 | 1.23341 | 1.65214 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 10 | 1251 | 1267 | rollback | 0.822865 | 1.12078 | 1.40808 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 10 | 2501 | 2517 | rollback | 0.887558 | 1.23341 | 1.65214 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 10 | 1251 | 1267 | rollback | 0.822865 | 1.12078 | 1.40808 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 10 | 2501 | 2517 | rollback | 0.887558 | 1.23341 | 1.65214 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 10 | 1251 | 1263 | rollback | 0.868168 | 1.17577 | 1.43713 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 10 | 2501 | 2513 | rollback | 0.780597 | 1.0321 | 1.38814 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 10 | 1251 | 1267 | rollback | 0.822865 | 1.12078 | 1.40808 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 10 | 2501 | 2517 | rollback | 0.887558 | 1.23341 | 1.65214 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 11 | 1251 | 1267 | rollback | 0.790388 | 1.0694 | 1.40563 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 11 | 2501 | 2517 | rollback | 0.846316 | 1.11826 | 1.48449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 11 | 1251 | 1267 | rollback | 0.790388 | 1.0694 | 1.40563 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 11 | 2501 | 2517 | rollback | 0.846316 | 1.11826 | 1.48449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 11 | 1251 | 1267 | rollback | 0.790388 | 1.0694 | 1.40563 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 11 | 2501 | 2517 | rollback | 0.846316 | 1.11826 | 1.48449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 11 | 1251 | 1263 | rollback | 0.760116 | 1.02203 | 1.33884 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 11 | 2501 | 2513 | rollback | 0.814009 | 1.17643 | 1.64244 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 11 | 1251 | 1267 | rollback | 0.790388 | 1.0694 | 1.40563 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 11 | 2501 | 2517 | rollback | 0.846316 | 1.11826 | 1.48449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 12 | 1251 | 1267 | rollback | 0.800707 | 0.977173 | 1.18825 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 12 | 2501 | 2517 | rollback | 0.810336 | 1.17405 | 1.65106 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 12 | 1251 | 1267 | rollback | 0.800707 | 0.977173 | 1.18825 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 12 | 2501 | 2517 | rollback | 0.810336 | 1.17405 | 1.65106 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 12 | 1251 | 1267 | rollback | 0.800707 | 0.977173 | 1.18825 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 12 | 2501 | 2517 | rollback | 0.810336 | 1.17405 | 1.65106 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 12 | 1251 | 1263 | rollback | 0.774335 | 1.03764 | 1.40822 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 12 | 2501 | 2513 | rollback | 0.810551 | 1.12087 | 1.37546 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 12 | 1251 | 1267 | rollback | 0.800707 | 0.977173 | 1.18825 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 12 | 2501 | 2517 | rollback | 0.810336 | 1.17405 | 1.65106 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 13 | 1251 | 1267 | rollback | 0.806011 | 1.14421 | 1.31715 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 13 | 2501 | 2517 | rollback | 0.844332 | 1.21882 | 1.59402 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 13 | 1251 | 1267 | rollback | 0.806011 | 1.14421 | 1.31715 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 13 | 2501 | 2517 | rollback | 0.844332 | 1.21882 | 1.59402 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 13 | 1251 | 1267 | rollback | 0.806011 | 1.14421 | 1.31715 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 13 | 2501 | 2517 | rollback | 0.844332 | 1.21882 | 1.59402 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 13 | 1251 | 1263 | rollback | 0.807934 | 1.12696 | 1.33704 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 13 | 2501 | 2513 | rollback | 0.882653 | 1.17457 | 1.57837 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 13 | 1251 | 1267 | rollback | 0.806011 | 1.14421 | 1.31715 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 13 | 2501 | 2517 | rollback | 0.844332 | 1.21882 | 1.59402 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 14 | 1251 | 1267 | rollback | 0.889097 | 1.15415 | 1.58417 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm10 | 14 | 2501 | 2517 | rollback | 0.85827 | 1.19169 | 1.54737 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 14 | 1251 | 1267 | rollback | 0.889097 | 1.15415 | 1.58417 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm10 | 14 | 2501 | 2517 | rollback | 0.85827 | 1.19169 | 1.54737 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 14 | 1251 | 1267 | rollback | 0.889097 | 1.15415 | 1.58417 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm10 | 14 | 2501 | 2517 | rollback | 0.85827 | 1.19169 | 1.54737 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 14 | 1251 | 1263 | rollback | 0.838396 | 1.27942 | 1.64742 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm10 | 14 | 2501 | 2513 | rollback | 0.814581 | 1.07679 | 1.35672 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 14 | 1251 | 1267 | rollback | 0.889097 | 1.15415 | 1.58417 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm10 | 14 | 2501 | 2517 | rollback | 0.85827 | 1.19169 | 1.54737 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 10 | 1252 | 1268 | rollback | 0.787069 | 1.05355 | 1.33345 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 10 | 2501 | 2517 | rollback | 0.891707 | 1.23313 | 1.67295 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 10 | 1252 | 1268 | rollback | 0.787069 | 1.05355 | 1.33345 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 10 | 2501 | 2517 | rollback | 0.891707 | 1.23313 | 1.67295 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 10 | 1252 | 1268 | rollback | 0.787069 | 1.05355 | 1.33345 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 10 | 2501 | 2517 | rollback | 0.891707 | 1.23313 | 1.67295 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 10 | 1252 | 1264 | rollback | 0.808219 | 1.11485 | 1.47412 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 10 | 2501 | 2513 | rollback | 0.772574 | 1.01732 | 1.39417 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 10 | 1252 | 1268 | rollback | 0.787069 | 1.05355 | 1.33345 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 10 | 2501 | 2517 | rollback | 0.891707 | 1.23313 | 1.67295 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 11 | 1251 | 1267 | rollback | 0.79368 | 1.06424 | 1.40734 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 11 | 2502 | 2518 | rollback | 0.851141 | 1.12061 | 1.39561 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 11 | 1251 | 1267 | rollback | 0.79368 | 1.06424 | 1.40734 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 11 | 2502 | 2518 | rollback | 0.851141 | 1.12061 | 1.39561 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 11 | 1251 | 1267 | rollback | 0.79368 | 1.06424 | 1.40734 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 11 | 2502 | 2518 | rollback | 0.851141 | 1.12061 | 1.39561 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 11 | 1251 | 1263 | rollback | 0.746351 | 0.984191 | 1.32046 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 11 | 2502 | 2514 | rollback | 0.838785 | 1.23685 | 1.49128 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 11 | 1251 | 1267 | rollback | 0.79368 | 1.06424 | 1.40734 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 11 | 2502 | 2518 | rollback | 0.851141 | 1.12061 | 1.39561 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 12 | 1251 | 1267 | rollback | 0.807812 | 0.98098 | 1.16962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 12 | 2501 | 2517 | rollback | 0.808141 | 1.16433 | 1.61811 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 12 | 1251 | 1267 | rollback | 0.807812 | 0.98098 | 1.16962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 12 | 2501 | 2517 | rollback | 0.808141 | 1.16433 | 1.61811 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 12 | 1251 | 1267 | rollback | 0.807812 | 0.98098 | 1.16962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 12 | 2501 | 2517 | rollback | 0.808141 | 1.16433 | 1.61811 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 12 | 1251 | 1263 | rollback | 0.794361 | 1.07888 | 1.44183 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 12 | 2501 | 2513 | rollback | 0.82181 | 1.13993 | 1.43326 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 12 | 1251 | 1267 | rollback | 0.807812 | 0.98098 | 1.16962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 12 | 2501 | 2517 | rollback | 0.808141 | 1.16433 | 1.61811 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 13 | 1252 | 1268 | rollback | 0.807165 | 1.06512 | 1.3491 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 13 | 2502 | 2518 | rollback | 0.851263 | 1.10449 | 1.27901 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 13 | 1252 | 1268 | rollback | 0.807165 | 1.06512 | 1.3491 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 13 | 2502 | 2518 | rollback | 0.851263 | 1.10449 | 1.27901 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 13 | 1252 | 1268 | rollback | 0.807165 | 1.06512 | 1.3491 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 13 | 2502 | 2518 | rollback | 0.851263 | 1.10449 | 1.27901 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 13 | 1252 | 1264 | rollback | 0.787324 | 1.01431 | 1.22655 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 13 | 2502 | 2514 | rollback | 0.883775 | 1.17749 | 1.46891 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 13 | 1252 | 1268 | rollback | 0.807165 | 1.06512 | 1.3491 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 13 | 2502 | 2518 | rollback | 0.851263 | 1.10449 | 1.27901 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 14 | 1251 | 1267 | rollback | 0.900584 | 1.15985 | 1.58815 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | ENm20 | 14 | 2501 | 2517 | rollback | 0.842787 | 1.16043 | 1.46659 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 14 | 1251 | 1267 | rollback | 0.900584 | 1.15985 | 1.58815 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | ENm20 | 14 | 2501 | 2517 | rollback | 0.842787 | 1.16043 | 1.46659 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 14 | 1251 | 1267 | rollback | 0.900584 | 1.15985 | 1.58815 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | ENm20 | 14 | 2501 | 2517 | rollback | 0.842787 | 1.16043 | 1.46659 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 14 | 1251 | 1263 | rollback | 0.853647 | 1.27649 | 1.61905 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | ENm20 | 14 | 2501 | 2513 | rollback | 0.811948 | 1.03529 | 1.27134 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 14 | 1251 | 1267 | rollback | 0.900584 | 1.15985 | 1.58815 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | ENm20 | 14 | 2501 | 2517 | rollback | 0.842787 | 1.16043 | 1.46659 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 10 | 1876 | 1892 | confirmed | 0.0564667 | 0.0673327 | 0.0808354 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 10 | 1876 | 1892 | confirmed | 0.0564667 | 0.0673327 | 0.0808354 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 10 | 1876 | 1892 | confirmed | 0.0564667 | 0.0673327 | 0.0808354 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 10 | 1876 | 1888 | confirmed | 0.0634953 | 0.089448 | 0.129183 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 10 | 1876 | 1892 | confirmed | 0.0564667 | 0.0673327 | 0.0808354 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 11 | 1876 | 1892 | confirmed | 0.0491104 | 0.0594084 | 0.0648119 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 11 | 1876 | 1892 | confirmed | 0.0491104 | 0.0594084 | 0.0648119 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 11 | 1876 | 1892 | confirmed | 0.0491104 | 0.0594084 | 0.0648119 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 11 | 1876 | 1888 | confirmed | 0.0616222 | 0.0806689 | 0.118267 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 11 | 1876 | 1892 | confirmed | 0.0491104 | 0.0594084 | 0.0648119 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 12 | 1876 | 1892 | confirmed | 0.0503399 | 0.0660512 | 0.0737536 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 12 | 1876 | 1892 | confirmed | 0.0503399 | 0.0660512 | 0.0737536 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 12 | 1876 | 1892 | confirmed | 0.0503399 | 0.0660512 | 0.0737536 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 12 | 1876 | 1888 | confirmed | 0.0684418 | 0.0881447 | 0.112925 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 12 | 1876 | 1892 | confirmed | 0.0503399 | 0.0660512 | 0.0737536 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 13 | 1876 | 1892 | confirmed | 0.0583069 | 0.0903072 | 0.116639 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 13 | 1876 | 1892 | confirmed | 0.0583069 | 0.0903072 | 0.116639 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 13 | 1876 | 1892 | confirmed | 0.0583069 | 0.0903072 | 0.116639 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 13 | 1876 | 1888 | confirmed | 0.0615381 | 0.0744911 | 0.109761 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 13 | 1876 | 1892 | confirmed | 0.0583069 | 0.0903072 | 0.116639 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | EVpc | 14 | 1876 | 1892 | confirmed | 0.0457596 | 0.0705819 | 0.093778 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | EVpc | 14 | 1876 | 1892 | confirmed | 0.0457596 | 0.0705819 | 0.093778 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | EVpc | 14 | 1876 | 1892 | confirmed | 0.0457596 | 0.0705819 | 0.093778 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | EVpc | 14 | 1876 | 1888 | confirmed | 0.0541661 | 0.0730356 | 0.112365 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | EVpc | 14 | 1876 | 1892 | confirmed | 0.0457596 | 0.0705819 | 0.093778 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 10 | 1876 | 1892 | confirmed | 0.00538992 | 0.00650008 | 0.00458761 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 10 | 1876 | 1892 | confirmed | 0.00538992 | 0.00650008 | 0.00458761 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 10 | 1876 | 1892 | confirmed | 0.00538992 | 0.00650008 | 0.00458761 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 10 | 1876 | 1888 | confirmed | 0.0236114 | 0.03002 | 0.0420522 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 10 | 1876 | 1892 | confirmed | 0.00538992 | 0.00650008 | 0.00458761 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 11 | 1876 | 1892 | confirmed | 0.00234212 | 0.0027486 | 0.00272213 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 11 | 1876 | 1892 | confirmed | 0.00234212 | 0.0027486 | 0.00272213 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 11 | 1876 | 1892 | confirmed | 0.00234212 | 0.0027486 | 0.00272213 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 11 | 1876 | 1888 | confirmed | 0.00868013 | 0.0109312 | 0.0146614 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 11 | 1876 | 1892 | confirmed | 0.00234212 | 0.0027486 | 0.00272213 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 12 | 1876 | 1892 | confirmed | 0.0264843 | 0.0273896 | 0.0261133 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 12 | 1876 | 1892 | confirmed | 0.0264843 | 0.0273896 | 0.0261133 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 12 | 1876 | 1892 | confirmed | 0.0264843 | 0.0273896 | 0.0261133 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 12 | 1876 | 1888 | confirmed | 0.0248465 | 0.024655 | 0.0274045 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 12 | 1876 | 1892 | confirmed | 0.0264843 | 0.0273896 | 0.0261133 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 13 | 1876 | 1892 | confirmed | 0.016726 | 0.0303239 | 0.0268647 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 13 | 1876 | 1892 | confirmed | 0.016726 | 0.0303239 | 0.0268647 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 13 | 1876 | 1892 | confirmed | 0.016726 | 0.0303239 | 0.0268647 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 13 | 1876 | 1888 | confirmed | 0.0805239 | 0.104909 | 0.123247 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 13 | 1876 | 1892 | confirmed | 0.016726 | 0.0303239 | 0.0268647 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pc | 14 | 1876 | 1892 | confirmed | 0.015704 | 0.0184804 | 0.0165333 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pc | 14 | 1876 | 1892 | confirmed | 0.015704 | 0.0184804 | 0.0165333 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pc | 14 | 1876 | 1892 | confirmed | 0.015704 | 0.0184804 | 0.0165333 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pc | 14 | 1876 | 1888 | confirmed | 0.059305 | 0.0704257 | 0.0724195 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pc | 14 | 1876 | 1892 | confirmed | 0.015704 | 0.0184804 | 0.0165333 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 10 | 1251 | 1267 | confirmed | 0.00346764 | 0.00638234 | 0.00973067 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 10 | 2501 | 2517 | confirmed | 0.0210141 | 0.0357847 | 0.0309236 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 10 | 1251 | 1267 | confirmed | 0.00346764 | 0.00638234 | 0.00973067 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 10 | 2501 | 2517 | confirmed | 0.0210141 | 0.0357847 | 0.0309236 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 10 | 1251 | 1267 | confirmed | 0.00346764 | 0.00638234 | 0.00973067 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 10 | 2501 | 2517 | confirmed | 0.0210141 | 0.0357847 | 0.0309236 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 10 | 1251 | 1263 | confirmed | 0.0106913 | 0.0101056 | 0.0262381 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 10 | 2501 | 2513 | confirmed | 0.0693797 | 0.0758674 | 0.0975858 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 10 | 1251 | 1267 | confirmed | 0.00346764 | 0.00638234 | 0.00973067 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 10 | 2501 | 2517 | confirmed | 0.0210141 | 0.0357847 | 0.0309236 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 11 | 1251 | 1267 | confirmed | 0.00373129 | 0.0144807 | 0.0324044 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 11 | 2501 | 2517 | confirmed | 0.00976201 | 0.017727 | 0.0403229 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 11 | 1251 | 1267 | confirmed | 0.00373129 | 0.0144807 | 0.0324044 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 11 | 2501 | 2517 | confirmed | 0.00976201 | 0.017727 | 0.0403229 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 11 | 1251 | 1267 | confirmed | 0.00373129 | 0.0144807 | 0.0324044 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 11 | 2501 | 2517 | confirmed | 0.00976201 | 0.017727 | 0.0403229 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 11 | 1251 | 1263 | confirmed | 0.0207695 | 0.0352502 | 0.041805 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 11 | 2501 | 2513 | confirmed | 0.0304674 | 0.0397992 | 0.0406194 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 11 | 1251 | 1267 | confirmed | 0.00373129 | 0.0144807 | 0.0324044 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 11 | 2501 | 2517 | confirmed | 0.00976201 | 0.017727 | 0.0403229 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 12 | 1251 | 1267 | confirmed | 0.00661419 | 0.0118738 | 0.0136617 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 12 | 2501 | 2517 | confirmed | 0.00464576 | 0.0110191 | 0.0110098 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 12 | 1251 | 1267 | confirmed | 0.00661419 | 0.0118738 | 0.0136617 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 12 | 2501 | 2517 | confirmed | 0.00464576 | 0.0110191 | 0.0110098 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 12 | 1251 | 1267 | confirmed | 0.00661419 | 0.0118738 | 0.0136617 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 12 | 2501 | 2517 | confirmed | 0.00464576 | 0.0110191 | 0.0110098 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 12 | 1251 | 1263 | confirmed | 0.0155742 | 0.0402399 | 0.0460875 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 12 | 2501 | 2513 | confirmed | 0.0294204 | 0.0303572 | 0.0346757 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 12 | 1251 | 1267 | confirmed | 0.00661419 | 0.0118738 | 0.0136617 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 12 | 2501 | 2517 | confirmed | 0.00464576 | 0.0110191 | 0.0110098 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 13 | 1252 | 1268 | confirmed | 0.0547156 | 0.163817 | 0.0548449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 13 | 2501 | 2517 | confirmed | 0.00811641 | 0.00599871 | 0.0196508 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 13 | 1252 | 1268 | confirmed | 0.0547156 | 0.163817 | 0.0548449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 13 | 2501 | 2517 | confirmed | 0.00811641 | 0.00599871 | 0.0196508 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 13 | 1252 | 1268 | confirmed | 0.0547156 | 0.163817 | 0.0548449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 13 | 2501 | 2517 | confirmed | 0.00811641 | 0.00599871 | 0.0196508 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 13 | 1252 | 1264 | rollback | 0.186986 | 0.186747 | 0.73118 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 13 | 2501 | 2513 | confirmed | 0.0194121 | 0.1059 | 0.184559 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 13 | 1252 | 1268 | confirmed | 0.0547156 | 0.163817 | 0.0548449 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 13 | 2501 | 2517 | confirmed | 0.00811641 | 0.00599871 | 0.0196508 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 14 | 1251 | 1267 | confirmed | 0.00713497 | 0.00887209 | 0.00946064 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | pcb | 14 | 2501 | 2517 | confirmed | 0.0230419 | 0.025712 | 0.0196962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 14 | 1251 | 1267 | confirmed | 0.00713497 | 0.00887209 | 0.00946064 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | pcb | 14 | 2501 | 2517 | confirmed | 0.0230419 | 0.025712 | 0.0196962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 14 | 1251 | 1267 | confirmed | 0.00713497 | 0.00887209 | 0.00946064 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | pcb | 14 | 2501 | 2517 | confirmed | 0.0230419 | 0.025712 | 0.0196962 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 14 | 1251 | 1263 | confirmed | 0.0273368 | 0.0224735 | 0.0417309 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | pcb | 14 | 2501 | 2513 | confirmed | 0.0794659 | 0.0956593 | 0.0787818 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 14 | 1251 | 1267 | confirmed | 0.00713497 | 0.00887209 | 0.00946064 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | pcb | 14 | 2501 | 2517 | confirmed | 0.0230419 | 0.025712 | 0.0196962 |

## 早期取り消し（設定窓の ρ、線形比）

| 規則 | クラス | seed | 検出ブロック | 取り消しブロック | ρ |
|---|---|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | 10 | 1878 | 1881 | 13.4515 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | 10 | 1878 | 1881 | 13.4515 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | 10 | 1878 | 1881 | 13.4515 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | 10 | 1878 | 1881 | 7.66414 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | 11 | 1880 | 1882 | 5.6881 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | 11 | 1880 | 1882 | 5.6881 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | 11 | 1880 | 1882 | 5.6881 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | 11 | 1880 | 1882 | 4.61123 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | 12 | 1878 | 1880 | 20.1311 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | 12 | 1878 | 1880 | 20.1311 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | 12 | 1878 | 1880 | 20.1311 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | 12 | 1878 | 1881 | 6.91537 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | 13 | 1879 | 1881 | 9.19746 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | 13 | 1879 | 1881 | 9.19746 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | 13 | 1879 | 1881 | 9.19746 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | 13 | 1879 | 1882 | 72.3687 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | SC | 14 | 1878 | 1880 | 18.3925 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | SC | 14 | 1878 | 1880 | 18.3925 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | SC | 14 | 1878 | 1880 | 18.3925 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | SC | 14 | 1878 | 1881 | 24.8828 |

## 経路変化あり（pc、pcb、EVpc）

| rule | n | L1_whole | L1_first5 | L1_last10 | ex_0_1s | ex_last10 | L1_ev | ex_ev | L1_pc15_17 | L1_pcb10_12 | L1_pcb20_22 | n_trial | n_rollback | div_num | amp | false_rs | missed_rs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | 15 | -21.55 | -22.37 | -23.30 | 12.74 | -5.91 | -7.09 | 30.04 | -6.47 | -7.19 | -9.44 | 0.00 | 0.00 | 0 | 0 | — | — |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | 15 | -23.03 | -22.37 | -23.30 | 12.74 | -5.93 | -19.99 | 21.25 | -16.97 | -26.31 | -25.74 | 1.33 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | 15 | -23.03 | -22.37 | -23.30 | 12.74 | -5.93 | -19.99 | 21.25 | -16.97 | -26.31 | -25.74 | 1.33 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | 15 | -23.03 | -22.37 | -23.30 | 12.74 | -5.93 | -19.99 | 21.25 | -16.97 | -26.31 | -25.74 | 1.33 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | 15 | -23.03 | -22.37 | -23.30 | 12.74 | -5.93 | -19.99 | 21.25 | -16.97 | -26.31 | -25.74 | 1.33 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | 15 | -22.92 | -22.37 | -23.30 | 12.74 | -5.93 | -19.39 | 21.79 | -16.97 | -22.72 | -25.74 | 1.33 | 0.07 | 0 | 0 | 0 | 1 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | 15 | -23.03 | -22.37 | -23.30 | 12.74 | -5.93 | -19.99 | 21.25 | -16.97 | -26.31 | -25.74 | 1.33 | 0.00 | 0 | 0 | 0 | 0 |

## 誤差側の外乱（ENm20、ENm10、EN0）

| rule | n | L1_whole | L1_first5 | L1_last10 | ex_0_1s | ex_last10 | L1_ev | ex_ev | L1_pc15_17 | L1_pcb10_12 | L1_pcb20_22 | n_trial | n_rollback | div_num | amp | false_rs | missed_rs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.33 | 0.11 | — | — | — | 0.00 | 0.00 | 0 | 0 | — | — |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | 15 | -29.20 | -29.94 | -30.93 | 10.86 | -14.41 | -19.89 | 11.76 | — | — | — | 2.00 | 0.00 | 0 | 0 | 30 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.37 | 5.10 | — | — | — | 2.00 | 2.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.37 | 5.10 | — | — | — | 2.00 | 2.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.37 | 5.10 | — | — | — | 2.00 | 2.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.36 | 3.92 | — | — | — | 2.00 | 2.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | 15 | -29.74 | -29.94 | -30.93 | 10.86 | -14.52 | -23.37 | 5.10 | — | — | — | 2.00 | 2.00 | 0 | 0 | 0 | 0 |

## 入力の色の変化（SC）

| rule | n | L1_whole | L1_first5 | L1_last10 | ex_0_1s | ex_last10 | L1_ev | ex_ev | L1_pc15_17 | L1_pcb10_12 | L1_pcb20_22 | n_trial | n_rollback | div_num | amp | false_rs | missed_rs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | 5 | -23.49 | -29.94 | -16.30 | 10.86 | -0.66 | -16.67 | -2.24 | — | — | — | 0.00 | 0.00 | 0 | 0 | — | — |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | 5 | — | — | — | — | — | — | — | — | — | — | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | 5 | — | — | — | — | — | — | — | — | — | — | 1.00 | 0.00 | 5 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | 5 | -23.48 | -29.94 | -16.30 | 10.86 | -0.66 | -16.57 | -2.12 | — | — | — | 1.00 | 1.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | 5 | -23.48 | -29.94 | -16.30 | 10.86 | -0.66 | -16.57 | -2.12 | — | — | — | 1.00 | 1.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | 5 | -23.48 | -29.94 | -16.30 | 10.86 | -0.66 | -16.57 | -2.12 | — | — | — | 1.00 | 1.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | 5 | -23.47 | -29.94 | -16.30 | 10.86 | -0.66 | -16.44 | -1.64 | — | — | — | 1.00 | 1.00 | 0 | 0 | 0 | 0 |

## 変化なし（G、BU、PS、II300、a145、a120）

| rule | n | L1_whole | L1_first5 | L1_last10 | ex_0_1s | ex_last10 | L1_ev | ex_ev | L1_pc15_17 | L1_pcb10_12 | L1_pcb20_22 | n_trial | n_rollback | div_num | amp | false_rs | missed_rs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | — | — |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | 30 | -30.07 | -29.14 | -30.26 | 10.93 | -15.30 | — | — | — | — | — | 0.00 | 0.00 | 0 | 0 | 0 | 0 |

## クラスごとの L1 whole

| rule | BU | EN0 | ENm10 | ENm20 | EVpc | G | II300 | PS | SC | a120 | a145 | pc | pcb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | -30.55 | -29.06 | -29.82 | -30.32 | -9.14 | -30.75 | -30.10 | -30.75 | -23.49 | -28.86 | -29.42 | -28.72 | -26.79 |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | -30.55 | -28.09 | -29.35 | -30.15 | -9.32 | -30.75 | -30.10 | -30.75 | — | -28.86 | -29.42 | -30.18 | -29.58 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | -30.55 | -29.09 | -29.82 | -30.32 | -9.32 | -30.75 | -30.10 | -30.75 | — | -28.86 | -29.42 | -30.18 | -29.58 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | -30.55 | -29.09 | -29.82 | -30.32 | -9.32 | -30.75 | -30.10 | -30.75 | -23.48 | -28.86 | -29.42 | -30.18 | -29.58 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | -30.55 | -29.09 | -29.82 | -30.32 | -9.32 | -30.75 | -30.10 | -30.75 | -23.48 | -28.86 | -29.42 | -30.18 | -29.58 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | -30.55 | -29.08 | -29.82 | -30.32 | -9.32 | -30.75 | -30.10 | -30.75 | -23.48 | -28.86 | -29.42 | -30.18 | -29.25 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | -30.55 | -29.09 | -29.82 | -30.32 | -9.32 | -30.75 | -30.10 | -30.75 | -23.47 | -28.86 | -29.42 | -30.18 | -29.58 |

## クラスごとの L1_ev（イベント直後 2 秒）

| rule | BU | EN0 | ENm10 | ENm20 | EVpc | G | II300 | PS | SC | a120 | a145 | pc | pcb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | — | -18.38 | -23.92 | -27.67 | -5.11 | — | — | — | -16.67 | — | — | -7.84 | -8.31 |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | — | -12.66 | -20.59 | -26.42 | -7.76 | — | — | — | — | — | — | -26.18 | -26.03 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | — | -18.56 | -23.94 | -27.61 | -7.76 | — | — | — | — | — | — | -26.18 | -26.03 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | — | -18.56 | -23.94 | -27.61 | -7.76 | — | — | — | -16.57 | — | — | -26.18 | -26.03 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | — | -18.56 | -23.94 | -27.61 | -7.76 | — | — | — | -16.57 | — | — | -26.18 | -26.03 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | — | -18.51 | -23.94 | -27.63 | -7.76 | — | — | — | -16.57 | — | — | -26.18 | -24.23 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | — | -18.56 | -23.94 | -27.61 | -7.76 | — | — | — | -16.44 | — | — | -26.18 | -26.03 |

## クラスごとの ex_ev（イベント直後 2 秒の Φ 超過）

| rule | BU | EN0 | ENm10 | ENm20 | EVpc | G | II300 | PS | SC | a120 | a145 | pc | pcb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4 early=ND mue1 end=fixed32 d0.03 nors | — | 9.58 | -0.27 | -8.98 | 30.11 | — | — | — | -2.24 | — | — | 30.85 | 29.16 |
| P4 early=ND mue1 end=fixed32 d0.03 ver- | — | 21.66 | 11.68 | 1.94 | 22.63 | — | — | — | — | — | — | 20.88 | 20.25 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB | — | 14.87 | 4.93 | -4.51 | 22.63 | — | — | — | — | — | — | 20.88 | 20.25 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab4 | — | 14.87 | 4.93 | -4.51 | 22.63 | — | — | — | -2.12 | — | — | 20.88 | 20.25 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 wB ab2 | — | 14.87 | 4.93 | -4.51 | 22.63 | — | — | — | -2.12 | — | — | 20.88 | 20.25 |
| P4 early=ND mue1 end=fixed32 d0.03 ver12/0.5 wB ab4 | — | 13.65 | 3.72 | -5.60 | 22.63 | — | — | — | -2.12 | — | — | 20.88 | 21.86 |
| P4 early=ND mue1 end=fixed32 d0.03 ver16/0.5 ab4 | — | 14.87 | 4.93 | -4.51 | 22.63 | — | — | — | -1.64 | — | — | 20.88 | 20.25 |

## 再現手順

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python prop4.py check
python prop4.py run --procs <コア数>
python prop4.py report
```
