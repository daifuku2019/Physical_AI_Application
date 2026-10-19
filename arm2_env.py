"""
2自由度（2軸）平面アームの強化学習環境
  - 物理シミュレーション : MuJoCo
  - 環境インターフェース : Gymnasium
  - 可視化               : matplotlib（OpenGLを使わないので、Colabの CPU / GPU どちらでも動く）

タスク: ランダムに置かれた目標（緑の円）にアームの手先を到達させ、そこに留まる。
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")          # 画面のない環境でも描画できる設定
import matplotlib.pyplot as plt

# ============================================================
# ロボットの設計図（MJCF形式のXML）
#   構造：土台 → 関節1(j1) → リンク1 → 関節2(j2) → リンク2 → 手先(tip)
# ============================================================
XML = """
<mujoco model="arm2dof">
  <!-- 物理計算の刻み幅0.01秒、重力0（真上から見た平面内の動きにするため） -->
  <option timestep="0.01" gravity="0 0 0"/>

  <worldbody>
    <geom type="plane" size="1 1 0.1"/>

    <!-- リンク1：z軸まわりに回る関節1（回転制限なし）。damping＝回転の抵抗 -->
    <body name="link1" pos="0 0 0.05">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>

      <!-- リンク2：リンク1の先端に付く。関節2は±150度まで -->
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" limited="true" range="-150 150" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.018"/>
        <site name="tip" pos="0.2 0 0" size="0.02"/>   <!-- 手先の目印 -->
      </body>
    </body>

    <!-- 目標：プログラムから位置を動かせる物体。当たり判定なし -->
    <body name="target" mocap="true" pos="0.2 0.2 0.05">
      <geom type="sphere" size="0.025" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <!-- モータ：各関節に1つずつ。指令値（トルク）は -1〜+1 -->
  <actuator>
    <motor joint="j1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="j2" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

EPISODE_STEPS = 200     # 1エピソード = 200ステップ = 4秒
CONTROL_DT = 0.02       # 1ステップ = 物理計算2回 × 0.01秒
SUCCESS_DIST = 0.02     # 最終距離がこれ未満なら「成功」とみなす [m]


class Arm2Env(gym.Env):
    """2軸アームの目標到達タスク"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_string(XML)   # XMLからロボットを作る
        self.data = mujoco.MjData(self.model)               # 現在の状態の入れ物
        # 行動：2つのモータへの指令値（各 -1〜+1）
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        # 観測：10個の数値（内訳は _obs を参照）
        self.observation_space = spaces.Box(-np.inf, np.inf, (10,), np.float64)
        self.render_mode = render_mode
        self.fig = None
        self.t = 0

    def _obs(self):
        """AIに渡す観測（10個）"""
        q = self.data.qpos                      # 関節角度
        tip = self.data.site("tip").xpos[:2]    # 手先の位置 (x, y)
        tgt = self.data.mocap_pos[0][:2]        # 目標の位置 (x, y)
        return np.concatenate([
            np.cos(q), np.sin(q),   # 角度は cos・sin で表す（359度と1度を近い値にするため）
            self.data.qvel,         # 関節の回転速度
            tgt,                    # 目標位置
            tip - tgt,              # 手先と目標のズレ
        ])

    def reset(self, seed=None, options=None):
        """エピソード開始：アームと目標を初期配置する"""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.np_random.uniform(-0.1, 0.1, 2)   # 初期角度を少しランダムに
        r = self.np_random.uniform(0.1, 0.38)                      # 目標までの距離（腕の届く範囲）
        th = self.np_random.uniform(-np.pi, np.pi)                 # 目標の方向
        self.data.mocap_pos[0] = [r * np.cos(th), r * np.sin(th), 0.05]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        """AIの行動を受け取り、0.02秒進めて結果を返す"""
        self.data.ctrl[:] = a
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        obs = self._obs()
        dist = float(np.linalg.norm(obs[-2:]))           # 手先と目標の距離
        # 報酬：目標に近いほど高く、大きな力を使うと少し減点
        reward = -dist - 0.01 * float(np.sum(np.square(a)))
        truncated = self.t >= EPISODE_STEPS
        return obs, reward, False, truncated, {"dist": dist}

    def render(self):
        """matplotlibで真上から見た図を描き、RGB画像(480×480)として返す"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
        ax = self.ax
        ax.clear()
        ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

        elbow = self.data.body("link2").xpos[:2]   # ひじ（関節2）の位置
        tip = self.data.site("tip").xpos[:2]       # 手先の位置
        tgt = self.data.mocap_pos[0][:2]           # 目標の位置

        ax.add_patch(plt.Circle(tgt, 0.025, color="lime", alpha=0.6))                 # 目標
        ax.plot([0, elbow[0]], [0, elbow[1]], lw=8, color=(.2, .4, .8), solid_capstyle="round")
        ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], lw=7, color=(.8, .4, .2), solid_capstyle="round")
        ax.plot(0, 0, "ko", ms=8)                                                      # 土台
        ax.set_title(f"t = {self.t * CONTROL_DT:.2f} s   dist = {np.linalg.norm(tip - tgt):.3f} m")

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
