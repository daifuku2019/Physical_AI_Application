"""
PPO（Stable-Baselines3）で2軸アームを学習する
  実行例: python train.py --steps 500000
出力:
  ppo_arm2.zip        学習済みモデル
  logs/monitor/       エピソードごとの報酬ログ（学習曲線の作成に使用）
  logs/tb/            TensorBoard用ログ
"""
import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from arm2_env import Arm2Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000, help="総学習ステップ数")
    p.add_argument("--n-envs", type=int, default=8, help="並列に動かす環境の数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ppo_arm2")
    args = p.parse_args()

    # 環境を並列化。monitor_dir にエピソード報酬が記録される
    env = make_vec_env(Arm2Env, n_envs=args.n_envs, seed=args.seed,
                       monitor_dir="logs/monitor")

    # MlpPolicy：方策・価値関数とも 64ユニット×2層 の多層パーセプトロン（SB3の標準設定）
    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=256,
                seed=args.seed, tensorboard_log="logs/tb", device="cpu")

    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    print(f"学習時間: {(time.time() - t0) / 60:.1f} 分")

    model.save(args.out)
    print(f"保存しました: {args.out}.zip")


if __name__ == "__main__":
    main()
