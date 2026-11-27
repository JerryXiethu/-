# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""2自由度机械天线（矩形锅盖，转轴在中心）分阶段控制仿真
动作流程（自动检测“彻底转到位”后再计时停止 10 秒）：
- 阶段 0：初始静止 1 秒，方便看清起始位置
- 阶段 1：先水平方向旋转 60°（含加速 -> 过冲 -> 回弹稳住），彻底到位后停止保持 10 秒
- 阶段 2：再垂直方向俯仰旋转 60°（含加速 -> 过冲 -> 回弹稳住），彻底到位后停止保持 10 秒
- 阶段 3：复位回 0°，到位后停 2 秒并循环

运行方式（在激活 .venv 后）：
    python my_antenna.py
"""

import math
import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # ==========================================================
        # 【模型 1：刚体几何与动力学模型（质量 + 转动惯量）】
        # 注：Newton 默认密度为 1000 kg/m³，这个矩形锅盖约重 46 kg，转动惯量较大
        # ==========================================================
        post_hx, post_hy, post_hz = 0.06, 0.06, 0.25   # 垂直支架柱
        dish_hx, dish_hy, dish_hz = 0.03, 0.55, 0.35   # 矩形锅盖（薄板）

        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=post_hx, hy=post_hy, hz=post_hz)

        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=dish_hx, hy=dish_hy, hz=dish_hz)

        # ==========================================================
        # 【模型 2：PD 位置伺服控制模型】 τ = Kp*(q_target - q) - Kd*dq
        # target_ke = 刚度 Kp（拉向目标的力度）
        # target_kd = 阻尼 Kd（刹车力度，越大抖动越小、停得越快）
        # ==========================================================
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 0.0, 1.0),  # 关节 0：绕竖直 Z 轴水平旋转
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.8), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_q=0.0,
            target_ke=180.0,  # 水平刚度 Kp
            target_kd=12.0,   # 水平阻尼 Kd（大幅调大，只轻微过冲一下就立刻稳稳刹住）
            actuator_mode=newton.JointTargetMode.POSITION,
        )

        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),  # 关节 1：绕水平 Y 轴俯仰旋转（转轴在锅盖正中心）
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, post_hz + 0.08), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_q=0.0,
            target_ke=220.0,  # 俯仰刚度 Kp
            target_kd=10.0,   # 俯仰阻尼 Kd
            actuator_mode=newton.JointTargetMode.POSITION,
        )

        builder.add_articulation([j0, j1], label="antenna_2dof")
        builder.add_ground_plane()

        # ==========================================================
        # 【模型 3：物理求解器（SolverMuJoCo）】负责每步积分动力学方程
        # ==========================================================
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, solver="newton")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.2, -2.2, 1.8), pitch=-18.0, yaw=135.0)

        # ==========================================================
        # 【模型 4：分阶段控制状态机】
        # 注：本机 CPU 跑弹窗时仿真时间比现实时间慢一点，设为 4 秒大概等于现实等 6~8 秒
        # ==========================================================
        self.target_angle = math.radians(60.0)  # 60° 对应弧度 ≈ 1.0472
        self.hold_duration = 4.0                # 到位后停止等待秒数（想更久可改成 10.0）
        self.phase = "init"
        self.phase_timer = 0.0
        self.settled_timer = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        print("[状态] 初始化：静止 1 秒，准备开始水平旋转 60°...")

    def _is_settled(self, q, qd):
        """判断两个关节是否已经彻底转到位并基本停稳"""
        pos_err = max(abs(float(q[0]) - self.target_yaw), abs(float(q[1]) - self.target_pitch))
        vel_mag = max(abs(float(qd[0])), abs(float(qd[1])))
        if pos_err < math.radians(1.5) and vel_mag < 0.05:
            self.settled_timer += self.frame_dt
        else:
            self.settled_timer = 0.0
        return self.settled_timer >= 0.3  # 连续 0.3 秒稳定在 1.5° 以内算“彻底转到位”

    def _update_target(self):
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        self.phase_timer += self.frame_dt

        if self.phase == "init":
            if self.phase_timer >= 1.0:
                self.phase = "move_yaw"
                self.phase_timer = 0.0
                self.settled_timer = 0.0
                self.target_yaw = self.target_angle
                self.target_pitch = 0.0
                print(f"[t={self.sim_time:5.1f}s] 阶段 1：开始水平旋转 -> 60°")

        elif self.phase == "move_yaw":
            if self._is_settled(q, qd):
                self.phase = "hold_yaw"
                self.phase_timer = 0.0
                print(f"[t={self.sim_time:5.1f}s] 水平 60° 已彻底转到位，停止等待 {self.hold_duration:.0f} 秒...")

        elif self.phase == "hold_yaw":
            if self.phase_timer >= self.hold_duration:
                self.phase = "move_pitch"
                self.phase_timer = 0.0
                self.settled_timer = 0.0
                self.target_yaw = self.target_angle
                self.target_pitch = -self.target_angle  # 负号为向上抬头 60°
                print(f"[t={self.sim_time:5.1f}s] 阶段 2：开始垂直俯仰旋转 -> 60°")

        elif self.phase == "move_pitch":
            if self._is_settled(q, qd):
                self.phase = "hold_pitch"
                self.phase_timer = 0.0
                print(f"[t={self.sim_time:5.1f}s] 俯仰 60° 已彻底转到位，停止等待 {self.hold_duration:.0f} 秒...")

        elif self.phase == "hold_pitch":
            if self.phase_timer >= self.hold_duration:
                self.phase = "reset"
                self.phase_timer = 0.0
                self.settled_timer = 0.0
                self.target_yaw = 0.0
                self.target_pitch = 0.0
                print(f"[t={self.sim_time:5.1f}s] 阶段 3：复位回 0°...")

        elif self.phase == "reset":
            if self._is_settled(q, qd):
                self.phase = "init"
                self.phase_timer = 0.0
                print(f"[t={self.sim_time:5.1f}s] 已复位，1 秒后重新循环。")

        targets = np.array([self.target_yaw, self.target_pitch], dtype=np.float32)
        self.control.joint_target_q.assign(targets)

    def step(self):
        self._update_target()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
