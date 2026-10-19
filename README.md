# PAI最終課題：2軸アームの強化学習による目標到達タスク

MuJoCo上に自作した2自由度（2軸）の平面アームに、強化学習（PPO）で
「ランダムな位置にある目標へ手先を到達させ、そこに留まる」動作を学習させました。
学習済みモデルの推論でアームを制御し、その様子を動画にしています。

- 動画：YouTube（URLをここに記載）
- 使用技術：MuJoCo / Gymnasium / Stable-Baselines3（PPO, MlpPolicy）/ matplotlib

## ファイル構成
| ファイル | 内容 |
|---|---|
| `arm2_env.py` | 2軸アームのモデル（MJCF）と強化学習環境（観測・行動・報酬） |
| `train.py` | PPOによる学習 |
| `evaluate.py` | 成功率などの定量評価と学習曲線の作成 |
| `make_video.py` | 推論でアームを動かして動画を作成 |
| `PAI_final_colab.ipynb` | Google Colabで一通り実行するためのノートブック |
| `requirements.txt` | 必要なライブラリ |

## 実行方法
```bash
pip install -r requirements.txt
python train.py --steps 500000      # 学習（CPUで十数分〜20分程度）
python evaluate.py --episodes 100   # 評価・学習曲線 → results/
python make_video.py --episodes 20  # 動画（約80秒・等倍速）→ arm2_result.mp4
```
Google Colabでは `PAI_final_colab.ipynb` を開き、上から順に実行してください。
描画はmatplotlibで行うため、GPUの有無に関係なく動作します。

## タスク設計
| 項目 | 内容 |
|---|---|
| ロボット | 長さ0.2mのリンク×2、ヒンジ関節×2（関節2は±150°制限）、重力なしの平面運動 |
| 行動（2次元） | 各関節モータへのトルク指令（−1〜+1） |
| 観測（10次元） | 関節角度のcos・sin、関節角速度、目標位置、手先と目標のズレ |
| 報酬 | −（手先と目標の距離）− 0.01 ×（指令値の二乗和） |
| エピソード | 200ステップ（4秒）。目標は毎回ランダム（半径0.1〜0.38m） |
| 成功判定 | 最終時点の距離が2cm未満 |

## ライセンス
MIT License（`LICENSE` を参照）
