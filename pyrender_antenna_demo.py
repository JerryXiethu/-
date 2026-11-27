"""Pyrender 体验脚本：蓝天白云 + 小飞机飞过 + 2自由度矩形锅盖天线实时追踪
会生成三份文件在当前目录下：
1. antenna_pyrender_demo.gif     -> 动画（天线自动旋转追踪蓝天里飞过的小飞机）
2. antenna_pyrender_snapshot.png -> 高清 RGB 渲染图
3. antenna_pyrender_depth.png    -> 深度图（Depth Map，视觉算法常用输入）

运行方式（激活 .venv 后）：
    python pyrender_antenna_demo.py
"""

import math
import numpy as np

# 兼容 numpy 2.x（pyrender 0.1.45 内部用到了已移除的 np.infty）
if not hasattr(np, "infty"):
    np.infty = np.inf

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import trimesh
import pyrender


def make_box(extents, rgba):
    mesh = trimesh.creation.box(extents=extents)
    mesh.visual.vertex_colors = (np.array(rgba) * 255).astype(np.uint8)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def make_cloud(center, scale=1.0):
    """用几个白色球体拼出一朵蓬松白云"""
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
    cloud_mesh = trimesh.util.concatenate(meshes)
    return pyrender.Mesh.from_trimesh(cloud_mesh, smooth=True)


def make_airplane():
    """拼一架红白配色的小飞机（机身 + 机翼 + 尾翼），默认机头朝 +X 方向"""
    parts = []
    # 机身
    fuselage = trimesh.creation.cylinder(radius=0.10, height=1.1, sections=16)
    fuselage.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    fuselage.visual.vertex_colors = np.array([245, 245, 245, 255], dtype=np.uint8)
    parts.append(fuselage)

    # 机头圆锥
    nose = trimesh.creation.cone(radius=0.10, height=0.28, sections=16)
    nose.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    nose.apply_translation([0.55, 0.0, 0.0])
    nose.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(nose)

    # 主机翼
    wings = trimesh.creation.box(extents=[0.28, 1.35, 0.035])
    wings.apply_translation([0.05, 0.0, 0.0])
    wings.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(wings)

    # 水平尾翼
    h_tail = trimesh.creation.box(extents=[0.16, 0.48, 0.025])
    h_tail.apply_translation([-0.46, 0.0, 0.02])
    h_tail.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(h_tail)

    # 垂直尾翼
    v_tail = trimesh.creation.box(extents=[0.18, 0.025, 0.28])
    v_tail.apply_translation([-0.46, 0.0, 0.14])
    v_tail.visual.vertex_colors = np.array([220, 60, 50, 255], dtype=np.uint8)
    parts.append(v_tail)

    plane_mesh = trimesh.util.concatenate(parts)
    return pyrender.Mesh.from_trimesh(plane_mesh, smooth=False)


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """构造相机位姿矩阵（OpenGL 约定相机朝向 -Z）"""
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


def main():
    # 1. 创建场景：天蓝色背景 + 明亮环境光
    scene = pyrender.Scene(
        bg_color=[0.52, 0.80, 0.98, 1.0],
        ambient_light=[0.45, 0.48, 0.55],
    )

    # 地面（草地绿）
    ground = make_box([30.0, 30.0, 0.1], [0.38, 0.62, 0.35, 1.0])
    ground_pose = np.eye(4)
    ground_pose[2, 3] = -0.05
    scene.add(ground, pose=ground_pose)

    # 几朵漂浮的白云
    for pos, sc in [
        ([-3.5, 6.5, 4.6], 1.1),
        ([1.8, 7.5, 5.2], 1.3),
        ([5.5, 6.0, 4.2], 0.95),
        ([-0.8, 8.5, 3.8], 0.85),
    ]:
        scene.add(make_cloud(pos, scale=sc))

    # 天线固定底座
    base_pose = np.eye(4)
    base_pose[2, 3] = 0.2
    scene.add(make_box([0.55, 0.55, 0.4], [0.35, 0.38, 0.42, 1.0]), pose=base_pose)

    # 天线可旋转立柱（关节 0：水平旋转）
    post_node = scene.add(make_box([0.14, 0.14, 0.6], [0.22, 0.45, 0.72, 1.0]))

    # 天线矩形锅盖（关节 1：中心轴俯仰，亮灰银色 + 中心探测头红色）
    dish_plate = trimesh.creation.box(extents=[0.07, 1.15, 0.75])
    dish_plate.visual.vertex_colors = np.array([225, 232, 240, 255], dtype=np.uint8)
    feed_horn = trimesh.creation.cylinder(radius=0.06, height=0.25, sections=16)
    feed_horn.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    feed_horn.apply_translation([0.14, 0.0, 0.0])
    feed_horn.visual.vertex_colors = np.array([235, 140, 40, 255], dtype=np.uint8)
    dish_mesh = pyrender.Mesh.from_trimesh(trimesh.util.concatenate([dish_plate, feed_horn]), smooth=False)
    dish_node = scene.add(dish_mesh)

    # 小飞机节点
    plane_node = scene.add(make_airplane())

    # 太阳光（平行光）
    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.92], intensity=3.2)
    scene.add(sun, pose=look_at([6.0, -5.0, 8.0], [0.0, 0.0, 1.0]))

    # 旁观相机
    camera = pyrender.PerspectiveCamera(yfov=math.radians(48.0))
    scene.add(camera, pose=look_at([-2.8, -4.6, 2.2], [0.4, 1.2, 1.7]))

    # 2. 离屏渲染 48 帧动画（飞机从左向右飞过，天线自动算出方位/俯仰角实时追踪）
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=500)
    frames = []
    antenna_center = np.array([0.0, 0.0, 1.08])

    num_frames = 48
    snapshot_idx = num_frames // 2

    for i in range(num_frames):
        u = i / (num_frames - 1)
        # 飞机航线：在天线前方蓝天里从左飞到右，带一点高度起伏
        px = -4.0 + 8.0 * u
        py = 3.2 - 0.6 * math.sin(math.pi * u)
        pz = 2.6 + 0.5 * math.sin(math.pi * u)
        plane_pos = np.array([px, py, pz])

        plane_pose = np.eye(4)
        plane_pose[:3, 3] = plane_pos
        scene.set_pose(plane_node, plane_pose)

        # 计算天线瞄准飞机需要的水平角 yaw 和俯仰角 pitch
        rel = plane_pos - antenna_center
        yaw = math.atan2(rel[1], rel[0])
        horiz_dist = math.hypot(rel[0], rel[1])
        pitch = -math.atan2(rel[2], horiz_dist)  # 绕 Y 轴负角为抬头

        # 更新立柱姿态（只绕 Z 轴水平转）
        R_yaw = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        post_pose = R_yaw.copy()
        post_pose[:3, 3] = [0.0, 0.0, 0.7]
        scene.set_pose(post_node, post_pose)

        # 更新锅盖姿态（先水平 yaw，再绕自身 Y 轴俯仰 pitch）
        R_pitch = trimesh.transformations.rotation_matrix(pitch, [0, 1, 0])
        dish_pose = R_yaw @ R_pitch
        dish_pose[:3, 3] = antenna_center
        scene.set_pose(dish_node, dish_pose)

        color, depth = renderer.render(scene)
        frames.append(color)

        if i == snapshot_idx:
            imageio.imwrite("antenna_pyrender_snapshot.png", color)
            # 保存深度图可视化
            valid = depth > 0
            depth_vis = np.zeros_like(depth)
            if np.any(valid):
                d_min, d_max = depth[valid].min(), depth[valid].max()
                depth_vis[valid] = (depth[valid] - d_min) / max(d_max - d_min, 1e-6)
            plt.imsave("antenna_pyrender_depth.png", depth_vis, cmap="viridis")

    renderer.delete()
    imageio.mimsave("antenna_pyrender_demo.gif", frames, fps=16, loop=0)
    print("渲染完成！已生成：")
    print(" - antenna_pyrender_demo.gif")
    print(" - antenna_pyrender_snapshot.png")
    print(" - antenna_pyrender_depth.png")


if __name__ == "__main__":
    main()
