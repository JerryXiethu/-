"""Newton 动力学仿真 + Pyrender 渲染 + 伺服控制精细曲线（完整闭环 Demo）

物理层（Newton + SolverMuJoCo）：
- 真实刚体动力学（质量、转动惯量、重力）+ 2自由度 PD 伺服控制
- 飞机在蓝天白云下飞过，天线从初始位置 (0°, 0°) 突然捕获目标 -> 加速追赶 -> 过冲 -> 精细收敛锁住飞机

可视化产出（直接可用于组会 PPT 汇报）：
1. antenna_newton_tracking.gif -> 3D 动画（带橙红色指向射线，肉眼可见惯性滞后、过冲与锁定）
2. servo_dynamics_curves.png   -> 四合一动力学控制分析图（角度跟踪 / 跟踪误差 / 角速度 / 伺服扭矩）

运行方式（激活 .venv 后）：
    python newton_pyrender_servo_demo.py
"""

import math
import numpy as np

if not hasattr(np, "infty"):
    np.infty = np.inf

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import trimesh
import pyrender
import warp as wp
import newton


# 配置中文字体（Windows 微软雅黑），避免图表中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_box(extents, rgba):
    mesh = trimesh.creation.box(extents=extents)
    mesh.visual.vertex_colors = (np.array(rgba) * 255).astype(np.uint8)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def make_cloud(center, scale=1.0):
    offsets = [
        (0.0, 0.0, 0.0, 0.65),
        (-0.55, 0.05, -0.05, 0.48),
        (0.55, -0.05, -0.05, 0.50),
        (0.15, 0.1, 0.35, 0.45),
        (-0.25, -0.05, 0.28, 0.40),
    ]
    meshes = []
    for dx, dy, dz, r in offsets:
        s = trimesh.creation.icosphere(subdivisions=2, radius=r * scale)
        s.apply_translation([center[0] + dx * scale, center[1] + dy * scale, center[2] + dz * scale])
        s.visual.vertex_colors = np.array([250, 252, 255, 255], dtype=np.uint8)
        meshes.append(s)
    return pyrender.Mesh.from_trimesh(trimesh.util.concatenate(meshes), smooth=True)


def make_airplane():
    parts = []
    fuselage = trimesh.creation.cylinder(radius=0.10, height=1.1, sections=16)
    fuselage.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    fuselage.visual.vertex_colors = np.array([245, 245, 245, 255], dtype=np.uint8)
    parts.append(fuselage)

    nose = trimesh.creation.cone(radius=0.10, height=0.28, sections=16)
    nose.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    nose.apply_translation([0.55, 0.0, 0.0])
    nose.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(nose)

    wings = trimesh.creation.box(extents=[0.28, 1.35, 0.035])
    wings.apply_translation([0.05, 0.0, 0.0])
    wings.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(wings)

    h_tail = trimesh.creation.box(extents=[0.16, 0.48, 0.025])
    h_tail.apply_translation([-0.46, 0.0, 0.02])
    h_tail.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(h_tail)

    v_tail = trimesh.creation.box(extents=[0.18, 0.025, 0.28])
    v_tail.apply_translation([-0.46, 0.0, 0.14])
    v_tail.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(v_tail)

    return pyrender.Mesh.from_trimesh(trimesh.util.concatenate(parts), smooth=False)


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def run_newton_dynamics():
    """第 1 部分：用 Newton 跑真实 2 自由度动力学 + PD 伺服追踪仿真"""
    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    total_time = 5.0
    num_frames = int(total_time * fps)

    # 伺服控制参数（调成“轻微过冲 + 快速精细锁定”的实战伺服手感）
    kp_yaw, kd_yaw = 180.0, 16.0
    kp_pitch, kd_pitch = 180.0, 16.0

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    post_hx, post_hy, post_hz = 0.07, 0.07, 0.30
    dish_hx, dish_hy, dish_hz = 0.035, 0.575, 0.375

    link_0 = builder.add_link()
    builder.add_shape_box(link_0, hx=post_hx, hy=post_hy, hz=post_hz)

    link_1 = builder.add_link()
    builder.add_shape_box(link_1, hx=dish_hx, hy=dish_hy, hz=dish_hz)

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.7), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        target_q=0.0,
        target_ke=kp_yaw,
        target_kd=kd_yaw,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    j1 = builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, post_hz + 0.08), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        target_q=0.0,
        target_ke=kp_pitch,
        target_kd=kd_pitch,
        actuator_mode=newton.JointTargetMode.POSITION,
    )

    builder.add_articulation([j0, j1], label="antenna_2dof")
    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model, solver="newton")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    antenna_center = np.array([0.0, 0.0, 1.08])

    history = {
        "t": [],
        "plane_pos": [],
        "target_q": [],
        "actual_q": [],
        "actual_qd": [],
        "torque": [],
    }

    for frame in range(num_frames):
        t = frame * frame_dt
        u = t / total_time

        # 飞机航线：在蓝天中从左向右飞过
        px = -4.2 + 8.4 * u
        py = 3.2 - 0.5 * math.sin(math.pi * u)
        pz = 2.6 + 0.45 * math.sin(math.pi * u)
        plane_pos = np.array([px, py, pz])

        # 前 0.4 秒天线处于待机 (0°, 0°)，0.4 秒时刻雷达突然发现飞机并开始伺服跟踪！
        if t < 0.4:
            target_yaw, target_pitch = 0.0, 0.0
        else:
            rel = plane_pos - antenna_center
            target_yaw = math.atan2(rel[1], rel[0])
            target_pitch = -math.atan2(rel[2], math.hypot(rel[0], rel[1]))

        control.joint_target_q.assign(np.array([target_yaw, target_pitch], dtype=np.float32))

        for _ in range(sim_substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        q = state_0.joint_q.numpy().copy()
        qd = state_0.joint_qd.numpy().copy()
        # 计算对应的 PD 伺服输出扭矩：tau = Kp*(q_target - q) - Kd*qd
        tau_yaw = kp_yaw * (target_yaw - q[0]) - kd_yaw * qd[0]
        tau_pitch = kp_pitch * (target_pitch - q[1]) - kd_pitch * qd[1]

        history["t"].append(t)
        history["plane_pos"].append(plane_pos)
        history["target_q"].append([target_yaw, target_pitch])
        history["actual_q"].append([float(q[0]), float(q[1])])
        history["actual_qd"].append([float(qd[0]), float(qd[1])])
        history["torque"].append([float(tau_yaw), float(tau_pitch)])

    for k in history:
        history[k] = np.array(history[k])
    return history


def plot_dynamics_curves(history):
    """第 2 部分：绘制四合一精细伺服控制曲线图"""
    t = history["t"]
    tgt_deg = np.degrees(history["target_q"])
    act_deg = np.degrees(history["actual_q"])
    err_deg = tgt_deg - act_deg
    qd_deg = np.degrees(history["actual_qd"])
    tau = history["torque"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=150)
    fig.suptitle("2自由度机械天线：Newton 动力学与 PD 伺服控制精细响应过程", fontsize=14, fontweight="bold")

    # 子图 1：角度跟踪（目标 vs 实际）
    ax = axes[0, 0]
    ax.plot(t, tgt_deg[:, 0], "--", color="#1f77b4", alpha=0.7, label="水平目标角度 (Yaw Target)")
    ax.plot(t, act_deg[:, 0], "-", color="#1f77b4", lw=2.0, label="水平实际角度 (Yaw Actual)")
    ax.plot(t, tgt_deg[:, 1], "--", color="#d62728", alpha=0.7, label="俯仰目标角度 (Pitch Target)")
    ax.plot(t, act_deg[:, 1], "-", color="#d62728", lw=2.0, label="俯仰实际角度 (Pitch Actual)")
    ax.axvline(0.4, color="gray", ls=":", lw=1.2, label="t=0.4s 发现目标")
    ax.set_title("1. 角度跟踪响应（可见加速、过冲与锁定）")
    ax.set_ylabel("角度 (°)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 子图 2：跟踪误差
    ax = axes[0, 1]
    ax.plot(t, err_deg[:, 0], color="#1f77b4", lw=1.8, label="水平误差 e_yaw")
    ax.plot(t, err_deg[:, 1], color="#d62728", lw=1.8, label="俯仰误差 e_pitch")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("2. 跟踪误差收敛过程 (Target - Actual)")
    ax.set_ylabel("角度误差 (°)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 子图 3：角速度
    ax = axes[1, 0]
    ax.plot(t, qd_deg[:, 0], color="#1f77b4", lw=1.8, label="水平角速度")
    ax.plot(t, qd_deg[:, 1], color="#d62728", lw=1.8, label="俯仰角速度")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("3. 关节角速度（体现加减速与阻尼制动）")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("角速度 (°/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 子图 4：伺服输出扭矩
    ax = axes[1, 1]
    ax.plot(t, tau[:, 0], color="#1f77b4", lw=1.8, label="水平伺服扭矩 τ_yaw")
    ax.plot(t, tau[:, 1], color="#d62728", lw=1.8, label="俯仰伺服扭矩 τ_pitch")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("4. 底层伺服电机输出扭矩 (τ = Kp·e - Kd·dq)")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("扭矩 (N·m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("servo_dynamics_curves.png")
    plt.close(fig)


def render_with_pyrender(history):
    """第 3 部分：把 Newton 算出来的真实物理角度传给 Pyrender 渲染成动画"""
    scene = pyrender.Scene(
        bg_color=[0.52, 0.80, 0.98, 1.0],
        ambient_light=[0.45, 0.48, 0.55],
    )

    ground_pose = np.eye(4)
    ground_pose[2, 3] = -0.05
    scene.add(make_box([30.0, 30.0, 0.1], [0.38, 0.62, 0.35, 1.0]), pose=ground_pose)

    for pos, sc in [
        ([-3.5, 6.5, 4.6], 1.1),
        ([1.8, 7.5, 5.2], 1.3),
        ([5.5, 6.0, 4.2], 0.95),
        ([-0.8, 8.5, 3.8], 0.85),
    ]:
        scene.add(make_cloud(pos, scale=sc))

    base_pose = np.eye(4)
    base_pose[2, 3] = 0.2
    scene.add(make_box([0.55, 0.55, 0.4], [0.35, 0.38, 0.42, 1.0]), pose=base_pose)

    post_node = scene.add(make_box([0.14, 0.14, 0.6], [0.22, 0.45, 0.72, 1.0]))

    # 锅盖 + 中心探测头 + 一条半透明感橙红色“实际探测波束射线”，直观看出滞后和过冲！
    dish_plate = trimesh.creation.box(extents=[0.07, 1.15, 0.75])
    dish_plate.visual.vertex_colors = np.array([225, 232, 240, 255], dtype=np.uint8)
    beam = trimesh.creation.cylinder(radius=0.018, height=5.5, sections=12)
    beam.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    beam.apply_translation([2.75, 0.0, 0.0])
    beam.visual.vertex_colors = np.array([245, 80, 40, 255], dtype=np.uint8)
    dish_mesh = pyrender.Mesh.from_trimesh(trimesh.util.concatenate([dish_plate, beam]), smooth=False)
    dish_node = scene.add(dish_mesh)

    plane_node = scene.add(make_airplane())

    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.92], intensity=3.2)
    scene.add(sun, pose=look_at([6.0, -5.0, 8.0], [0.0, 0.0, 1.0]))

    camera = pyrender.PerspectiveCamera(yfov=math.radians(48.0))
    scene.add(camera, pose=look_at([-2.8, -4.6, 2.2], [0.4, 1.2, 1.7]))

    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=500)
    frames = []
    antenna_center = np.array([0.0, 0.0, 1.08])

    # 每隔 3 帧抽一帧渲染（60fps -> 20fps GIF，流畅且生成快）
    for i in range(0, len(history["t"]), 3):
        plane_pose = np.eye(4)
        plane_pose[:3, 3] = history["plane_pos"][i]
        scene.set_pose(plane_node, plane_pose)

        # 这里用的是 Newton 算出来的“真实物理角度”（带惯性、加速和过冲！）
        yaw, pitch = history["actual_q"][i]
        R_yaw = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        post_pose = R_yaw.copy()
        post_pose[:3, 3] = [0.0, 0.0, 0.7]
        scene.set_pose(post_node, post_pose)

        R_pitch = trimesh.transformations.rotation_matrix(pitch, [0, 1, 0])
        dish_pose = R_yaw @ R_pitch
        dish_pose[:3, 3] = antenna_center
        scene.set_pose(dish_node, dish_pose)

        color, _ = renderer.render(scene)
        frames.append(color)

    renderer.delete()
    imageio.mimsave("antenna_newton_tracking.gif", frames, fps=20, loop=0)


if __name__ == "__main__":
    print("1/3 正在运行 Newton 动力学 + PD 伺服仿真...")
    hist = run_newton_dynamics()
    print("2/3 正在绘制精细动力学控制曲线 (servo_dynamics_curves.png)...")
    plot_dynamics_curves(hist)
    print("3/3 正在用 Pyrender 渲染带波束射线的追踪动画 (antenna_newton_tracking.gif)...")
    render_with_pyrender(hist)
    print("全部完成！")
