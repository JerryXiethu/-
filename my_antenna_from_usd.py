"""从 OpenUSD 资产文件直接导入 2 自由度机械天线并运行分阶段 PD 伺服控制
对比 my_antenna.py：
- 机械结构、尺寸、关节轴向、刚度/阻尼全部保存在 assets/antenna_2dof.usda 里
- Python 代码里只需要一行 builder.add_usd("assets/antenna_2dof.usda") 就能导入！

运行方式（激活 .venv 后）：
    python my_antenna_from_usd.py
"""

import math
import os
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
        # 核心亮点：一行代码直接导入 OpenUSD 资产（含立柱、锅盖与两个垂直关节）！
        # ==========================================================
        usd_path = os.path.join(os.path.dirname(__file__), "assets", "antenna_2dof.usda")
        builder.add_usd(usd_path)
        builder.add_ground_plane()

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

        # 分阶段控制状态机
        self.target_angle = math.radians(60.0)
        self.hold_duration = 4.0
        self.phase = "init"
        self.phase_timer = 0.0
        self.settled_timer = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        print(f"[USD 导入成功] {usd_path}")
        print("[状态] 初始化：静止 1 秒，准备开始水平旋转 60°...")

    def _is_settled(self, q, qd):
        pos_err = max(abs(float(q[0]) - self.target_yaw), abs(float(q[1]) - self.target_pitch))
        vel_mag = max(abs(float(qd[0])), abs(float(qd[1])))
        if pos_err < math.radians(1.5) and vel_mag < 0.05:
            self.settled_timer += self.frame_dt
        else:
            self.settled_timer = 0.0
        return self.settled_timer >= 0.3

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
                self.target_pitch = -self.target_angle
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
