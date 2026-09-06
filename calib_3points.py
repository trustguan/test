# -*- coding: utf-8 -*-
"""
calib_3points.py —— 三点 3D 标定算法（Python / NumPy）

给定同一组 3 个物理点在【相机坐标系】和【机器人坐标系】下的坐标，
返回相机 -> 机器人的 4x4 齐次变换矩阵：

    T_rob_cam = T_rob_obj @ inv(T_cam_obj)

其中 T_*_obj 由 "O/X/Y 三点建系法" 构造：
    O 点 -> 原点；  X 点 -> X 轴；  Y 点 -> Y 轴；
    Z = X × Y，再反算 X = Y × Z 保证严格正交。

用法：
    T = calib_3points(cam_pts, rob_pts)   # 各为 (3,3) 数组，行序 [O, X, Y]
    python calib_3points.py               # 运行自带数值验证测试
"""
import numpy as np


def frame_from_3_points(p0, px, py):
    """由三个点构造正交坐标系，返回 4x4 位姿矩阵。三点共线/重合抛异常。"""
    p0, px, py = map(np.asarray, (p0, px, py))

    rx = px - p0
    ry = py - p0
    if np.linalg.norm(rx) < 1e-12 or np.linalg.norm(ry) < 1e-12:
        raise ValueError("存在重合点，无法建系")
    rx = rx / np.linalg.norm(rx)
    ry = ry / np.linalg.norm(ry)

    rz = np.cross(rx, ry)
    if np.linalg.norm(rz) < 1e-9:
        raise ValueError("三点共线，无法确定坐标系")
    rz = rz / np.linalg.norm(rz)

    rx = np.cross(ry, rz)          # 重新正交化 X 轴（等价 -(rz×ry)）

    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = rx, ry, rz, p0
    return T


def calib_3points(cam_pts, rob_pts):
    """
    三点标定：求相机 -> 机器人变换矩阵。

    参数
    ----
    cam_pts : (3,3) array-like，行序 [O点, X点, Y点]，相机坐标系下
    rob_pts : (3,3) array-like，同一物理点在机器人坐标系下

    返回
    ----
    T : (4,4) ndarray，p_rob = T @ [p_cam; 1]
    """
    cam_pts = np.asarray(cam_pts, dtype=float)
    rob_pts = np.asarray(rob_pts, dtype=float)

    T_cam_obj = frame_from_3_points(cam_pts[0], cam_pts[1], cam_pts[2])
    T_rob_obj = frame_from_3_points(rob_pts[0], rob_pts[1], rob_pts[2])

    return T_rob_obj @ np.linalg.inv(T_cam_obj)


# ----------------------------------------------------------------------------
# 抗噪备选：Umeyama / Kabsch 最小二乘刚体配准（N >= 3 点，非共线）
# 三点建系法无冗余观测，噪声不被平均；点多或噪声大时用这个更稳。
# ----------------------------------------------------------------------------
def calib_points_umeyama(cam_pts, rob_pts):
    """
    最小二乘求解 cam -> rob 刚体变换（Umeyama, 无缩放）。
    cam_pts/rob_pts: (N,3)，同一组 N 个点在两坐标系下的坐标，N >= 3 且非共线。
    """
    P = np.asarray(cam_pts, dtype=float)
    Q = np.asarray(rob_pts, dtype=float)
    assert P.shape == Q.shape and P.shape[0] >= 3

    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)                      # 3x3 协方差
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T            # 保证 det(R) = +1
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = qc - R @ pc
    return T


# ----------------------------------------------------------------------------
# 数值验证测试
# ----------------------------------------------------------------------------
def _random_transform(rng, t_range):
    """生成随机刚体变换：随机旋转轴角 + 随机平移"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)  # Rodrigues
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-t_range, t_range, size=3)
    return T


def _run_test():
    rng = np.random.default_rng(42)

    # 模拟标定物上的三个标记点（标记物自身坐标系下，间距 ~100mm）
    M = np.array([[   0.0,    0.0,   0.0],
                  [100.0,    2.0,  -3.0],
                  [  -1.5, 100.0,   4.0]])

    # 真实工况尺度：相机看 500mm 内的标定物，相机距机器人基座 ~1000mm
    n_trials = 1000
    ok_exact = 0
    max_err_3pt, max_err_ume = 0.0, 0.0
    sum_err_3pt, sum_err_ume = 0.0, 0.0

    for _ in range(n_trials):
        T_rob_cam_true = _random_transform(rng, 1000.0)  # 真值：相机 -> 机器人
        T_cam_obj      = _random_transform(rng, 500.0)   # 相机 -> 标定物
        T_rob_obj      = T_rob_cam_true @ T_cam_obj

        # 同一物理点分别在相机系 / 机器人系下的坐标
        cam_pts = (T_cam_obj[:3, :3] @ M.T).T + T_cam_obj[:3, 3]
        rob_pts = (T_rob_obj[:3, :3] @ M.T).T + T_rob_obj[:3, 3]

        # --- 测试 1：无噪声，三点法应精确恢复 T_rob_cam ---
        T = calib_3points(cam_pts, rob_pts)
        if np.allclose(T, T_rob_cam_true, atol=1e-9):
            ok_exact += 1

        # --- 测试 2：加 0.1mm 测量噪声，对比三点法 vs Umeyama ---
        cam_noisy = cam_pts + rng.normal(0, 0.1, cam_pts.shape)
        rob_noisy = rob_pts + rng.normal(0, 0.1, rob_pts.shape)
        T3 = calib_3points(cam_noisy, rob_noisy)
        Tu = calib_points_umeyama(cam_noisy, rob_noisy)

        # 用恢复的矩阵变换一个 ~500mm 工作距离处的测试点，看位置误差
        p = np.array([300.0, -200.0, 500.0, 1.0])
        e3 = np.linalg.norm((T3 @ p - T_rob_cam_true @ p)[:3])
        eu = np.linalg.norm((Tu @ p - T_rob_cam_true @ p)[:3])
        max_err_3pt, max_err_ume = max(max_err_3pt, e3), max(max_err_ume, eu)
        sum_err_3pt += e3
        sum_err_ume += eu

    print(f"无噪声精确恢复          : {ok_exact}/{n_trials} 通过 (误差 < 1e-9)")
    print(f"0.1mm 噪声下三点建系法  : 平均 {sum_err_3pt/n_trials:.3f} mm, "
          f"最大 {max_err_3pt:.3f} mm  @ 500mm 工作距离")
    print(f"0.1mm 噪声下 Umeyama    : 平均 {sum_err_ume/n_trials:.3f} mm, "
          f"最大 {max_err_ume:.3f} mm  @ 500mm 工作距离")

    assert ok_exact == n_trials, "存在未精确恢复的用例！"
    assert max_err_3pt < 10.0, "三点法噪声误差超出预期！"
    print("全部测试通过 [OK]")


if __name__ == "__main__":
    _run_test()
