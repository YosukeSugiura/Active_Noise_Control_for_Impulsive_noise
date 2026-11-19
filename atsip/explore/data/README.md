# S0 データの入手と再生成

取得・確認日は 2026-11-19（UTC）。`raw/` に取得物、`prepared/` に音声の生成物を置く。Git で共有する実音は台帳のみ。経路は下記の CC BY 4.0 に従い、変換済みの小さなサブセットを共有する。

## 配布元と利用条件

| データ・版 | 入手・再配布条件、引用 | 取得 |
|---|---|---|
| MeshRIR S32-M441、v1.2、Zenodo 10852693 | CC BY 4.0。著者と配布元、ライセンス、変更点を表記して再配布可。Koyama、Nishida、Kimura、Abe、Ueno、Brunnström, *MeshRIR*, IEEE WASPAA 2021 を引用。 [著者ページ](https://www.sh01.org/MeshRIR/) · [固定版](https://zenodo.org/records/10852693) | 2026-11-19、NPY 版 ZIP |
| MIMII public 1.0、Zenodo 3384388 | Hitachi, Ltd.、CC BY-SA 4.0。クレジット・変更の表示が必要。加工物を公開する場合は継承条件がある。Purohit ほか, *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*, DCASE 2019 を引用。 [配布元](https://zenodo.org/records/3384388) · [論文](https://arxiv.org/abs/1909.09347) | 2026-11-19、`6_dB_valve.zip` の `valve/id_00/normal/00000000.wav` のみ。原著に記載された10秒録音 |
| NOISEX-92、SPIB 系の WAV ミラー、Zenodo 3921794 | ミラーは NOISEX を無料入手と記載し、レコード全体の Rights は Other (Non-Commercial)。元の SPIB 配布ページのアーカイブ同梱写しには再配布許諾の明記がなく、元音声の再配布可否と指定引用は**未確認**。原著として Varga & Steeneken, *Assessment for automatic speech recognition: II. NOISEX-92*, Speech Communication 12(3), 247–251, 1993 を記載する。 [旧配布ページ](https://spib.linse.ufsc.br/noise.html) · [取得ミラー](https://zenodo.org/records/3921794) | 2026-11-19、`NOISEX92.zip`。machinegun、factory1、destroyerengine、volvo を使用 |
| UrbanSound8K 1.0.0、Zenodo 1203745 | Zenodo の Rights は CC BY-NC 4.0。非商用の再配布は表示等の条件付き。Soundata が転載する同梱 README の利用条件には CC BY-NC 3.0 の表記もあり、版の記載に差がある。音源を共有する際は個別の Freesound クレジットと元音源の条件も確認する。配布サイト指定の Salamon、Jacoby、Bello, *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia 2014 を引用。 [著者ページ](https://urbansounddataset.weebly.com/urbansound8k.html) · [固定版](https://zenodo.org/records/1203745) · [同梱文書の転載](https://soundata.readthedocs.io/en/latest/_modules/soundata/datasets/urbansound8k.html) | 2026-11-19、engine_idling と jackhammer を各1クリップ。著者ページによる公開抜粋の上限は4秒 |
| trajectoRIR v1、Zenodo 15564430 | データの Rights は CC BY-NC-SA 4.0。非商用の再配布に表示と、加工版には継承条件。Damiano ほか, *The trajectoRIR Database*, arXiv:2503.23004 を指定。ZIP 内に `audio/`（WAV、静止 RIR と移動中の録音）、`meta/`（CSV）、`tools/`（Python）。将来のローカル配置候補は `raw/trajectoRIR/`。 [配布元](https://zenodo.org/records/15564430) · [形式の説明](https://link.springer.com/article/10.1186/s13636-026-00449-2) | 未取得。配布場所・条件・形式のみ確認 |

## コマンド

`atsip/explore` から Python 3.11 以降とリポジトリの NumPy、SciPy を使う。

```sh
python data/fetch.py mesh mimii noisex urban
python data/make_paths.py
python data/make_noise.py
```

- MeshRIR と NOISEX-92 は配布レコードの MD5 で ZIP 全体を照合する。MeshRIR は約3.53 GB。
- MIMII は HTTP Range で ZIP のディレクトリと選定した1ファイルの圧縮範囲だけを読み、ZIP の CRC-32 と取得時に固定した SHA-256 を照合する。約6.92 GB の ZIP 全体の MD5 は対象外。
- UrbanSound8K は公式 tar.gz を順方向に読み、対象2ファイルを見つけた時点で転送を終了する。各 WAV を MD5 確認済みの [Soundata 1.0 インデックス](https://zenodo.org/records/11176928) と照合する。
- `paths/` の `pair_*.npz` は各組の `fs`、`P`、`S`、`P_5cm`、`S_5cm` などを格納する。CC BY 4.0、Koyama ほか、MeshRIR v1.2（上表）。本サブセットの変更は音源・格子点の選別、48→16 kHz の変換、末尾切り詰め、float32 化。組と変換の詳細は `paths.json` と [DATA_REPORT.md](../DATA_REPORT.md) に記載。
- `prepared/noise.csv` が実音の生成台帳。今回の台帳を [noise.csv](noise.csv) に記録した。元のサンプル区間は開始を含み、終了を含まない。短いクリップは参考値のみ集計する。

## 無音と連続性

各ファイルの先頭から40秒を選び、チャネルの算術平均で単一チャネルにして、ポリフェーズ FIR で16 kHzに変換する。ゲート、音量の正規化、無音の切除、ゼロ埋め、ループ、ファイル結合は行わない。区間内の無音は保存し、全体がゼロの区間は出力対象から外す。40秒未満のファイルについては、そのファイル自体の統計だけを台帳に載せる。
