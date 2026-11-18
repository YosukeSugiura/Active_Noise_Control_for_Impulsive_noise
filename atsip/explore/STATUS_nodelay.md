# STATUS

- **テーマ**　SF+g と SF-ND+g の初期ステップ μ_e・長さ n_e の比較
- **実行者**　ChatGPT
- **更新日時**　2026-11-18 14:10:53 JST
- **状態**　`atsip/nodelay` の `615c86068b712ae05d7524e7d51821fec0f97261` 上で、切り替え位置の検証と48規則 × 30入力の1,440試行を完了。
- **結果**　全48規則で数値発散・増幅が0件。各方向の指標別の最良設定を比べた SF-ND − SF の対応差は ex 0–1 s が −3.648 dB、L1 first5 が −0.304 dB、L1 whole が −0.045 dB（差なし）。
- **未解決**　方向ごとに指定された μ_e 格子内での比較。経路変化時と最終評価用 seed での検証は未実施。
- **次の一手**　評価指標と初期ステップ設定を確定し、予約した seed 100–119 で比較する。
- **成果物の場所**　[実装](nodelay_early.py)、[集計](RESULTS_nodelay_early.md)、[試行別CSV](RESULTS_nodelay_early.csv)。既存比較は [RESULTS_nodelay.md](RESULTS_nodelay.md)。中間結果は `out/nodelay_early/`。
