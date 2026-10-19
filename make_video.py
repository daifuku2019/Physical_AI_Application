"""
学習済みモデルの推論でアームを動かし、提出用の動画(mp4)を作る
  実行例: python make_video.py --episodes 20     → 約80秒（等倍速）
"""
import argparse
import imageio
from stable_baselines3 import PPO
from arm2_env import Arm2Env, EPISODE_STEPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_arm2")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="arm2_result.mp4")
    args = p.parse_args()

    env = Arm2Env(render_mode="rgb_array")
    model = PPO.load(args.model, device="cpu")
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)                 # seedごとに目標位置が変わる
        frames.append(env.render())
        for _ in range(EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            frames.append(env.render())
        print(f"ep{ep:02d} 最終距離: {info['dist']:.3f} m")
    imageio.mimsave(args.out, frames, fps=env.metadata["render_fps"])
    print(f"保存しました: {args.out}（{len(frames) / env.metadata['render_fps']:.0f} 秒）")


if __name__ == "__main__":
    main()
