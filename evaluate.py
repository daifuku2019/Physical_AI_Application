"""
学習済みモデルを定量評価し、学習曲線を描く
  実行例: python evaluate.py --episodes 100
出力:
  コンソール        成功率・最終距離の平均・到達時間の平均
  results/learning_curve.png   学習曲線（レポート用）
  results/eval_result.txt      評価結果（レポート用）
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results
from arm2_env import Arm2Env, EPISODE_STEPS, CONTROL_DT, SUCCESS_DIST


def evaluate(model, n_episodes, seed_offset=1000):
    """学習時と別のseedで評価（初めて見る目標位置でのテスト）"""
    env = Arm2Env()
    final_d, reach_t = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        first_reach = None
        for t in range(EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            if first_reach is None and info["dist"] < SUCCESS_DIST:
                first_reach = (t + 1) * CONTROL_DT
        final_d.append(info["dist"])
        if first_reach is not None:
            reach_t.append(first_reach)
    final_d = np.array(final_d)
    return {
        "episodes": n_episodes,
        "success_rate": float(np.mean(final_d < SUCCESS_DIST) * 100),
        "final_dist_mean_mm": float(final_d.mean() * 1000),
        "final_dist_max_mm": float(final_d.max() * 1000),
        "reach_time_mean_s": float(np.mean(reach_t)) if reach_t else float("nan"),
    }


def plot_learning_curve(log_dir="logs/monitor", out="results/learning_curve.png"):
    df = load_results(log_dir)
    if len(df) == 0:
        print("学習ログが見つかりません（train.py を先に実行してください）")
        return
    steps = df["l"].cumsum()
    smooth = df["r"].rolling(100, min_periods=1).mean()
    plt.figure(figsize=(6, 3.5))
    plt.plot(steps, df["r"], alpha=0.15, label="each episode")
    plt.plot(steps, smooth, lw=2, label="moving average (100 ep)")
    plt.xlabel("training steps"); plt.ylabel("episode reward")
    plt.title("PPO learning curve (2-DoF arm reaching)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"学習曲線を保存しました: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_arm2")
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args()
    os.makedirs("results", exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    r = evaluate(model, args.episodes)
    text = (f"評価エピソード数   : {r['episodes']}\n"
            f"成功率(<{SUCCESS_DIST*100:.0f}cm)    : {r['success_rate']:.1f} %\n"
            f"最終距離の平均     : {r['final_dist_mean_mm']:.1f} mm\n"
            f"最終距離の最大     : {r['final_dist_max_mm']:.1f} mm\n"
            f"到達時間の平均     : {r['reach_time_mean_s']:.2f} s\n")
    print(text)
    with open("results/eval_result.txt", "w", encoding="utf-8") as f:
        f.write(text)

    plot_learning_curve()


if __name__ == "__main__":
    main()
