#pragma once
// ============================================================================
// Calib3Points.h —— 三点 3D 标定算法（C++ / Eigen）
//
// 原理：
//   给定同一组 3 个物理点在【相机坐标系】和【机器人坐标系】下的坐标，
//   分别用 "O/X/Y 三点建系法" 构造两个坐标系：
//       O 点 -> 坐标系原点
//       X 点 -> X 轴方向 (X - O)
//       Y 点 -> Y 轴方向 (Y - O)
//       Z 轴 = X × Y，再反算 X = Y × Z 保证三轴严格正交
//   得到棋盘(标定物)位姿 T_cam_obj 与 T_rob_obj 后：
//
//       T_rob_cam = T_rob_obj * T_cam_obj^-1
//
//   即相机坐标系 -> 机器人坐标系的 4x4 齐次变换矩阵。
//
//   相比经典 AX = XB 手眼标定（需多组位姿迭代），三点法只需一次测量，
//   代价是精度完全依赖三个点的测量精度，无冗余观测。
// ============================================================================

#include <Eigen/Dense>
#include <cmath>

namespace calib3 {

// ----------------------------------------------------------------------------
// 由三个点构造一个正交的 4x4 坐标系（位姿矩阵）
//   P0: 原点   Px: X 轴参考点   Py: Y 轴参考点（三点不得共线）
// 返回 T = [ rx ry rz | P0 ; 0 0 0 | 1 ]
// 失败（三点共线 / 重合）返回 false，此时 T 保持单位阵
// ----------------------------------------------------------------------------
inline bool BuildFrameFrom3Points(const Eigen::Vector3d& P0,
                                  const Eigen::Vector3d& Px,
                                  const Eigen::Vector3d& Py,
                                  Eigen::Matrix4d& T)
{
    Eigen::Vector3d rx = Px - P0;
    Eigen::Vector3d ry = Py - P0;

    if (rx.norm() < 1e-12 || ry.norm() < 1e-12)
        return false;                       // 点重合

    rx.normalize();
    ry.normalize();

    Eigen::Vector3d rz = rx.cross(ry);
    if (rz.norm() < 1e-9)
        return false;                       // 三点共线，无法确定平面法向
    rz.normalize();

    rx = ry.cross(rz);                      // 重新正交化 X 轴（等价 -(rz×ry)）

    T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 0) = rx;
    T.block<3, 1>(0, 1) = ry;
    T.block<3, 1>(0, 2) = rz;
    T.block<3, 1>(0, 3) = P0;
    return true;
}

// ----------------------------------------------------------------------------
// 三点标定：求相机 -> 机器人的变换矩阵
//
// 输入（每个数组 3 个点，下标 0=原点, 1=X 轴点, 2=Y 轴点）：
//   camPts[3] : 三点在【相机坐标系】下的坐标（如双目三角化 / 深度图取值）
//   robPts[3] : 同三个物理点在【机器人(基座)坐标系】下的坐标（如示教测量）
// 输出：
//   T_rob_cam  : 4x4 齐次矩阵，p_rob = T_rob_cam * p_cam
// 返回 false 表示三点退化（共线/重合）
// ----------------------------------------------------------------------------
inline bool Calib3Points(const Eigen::Vector3d camPts[3],
                         const Eigen::Vector3d robPts[3],
                         Eigen::Matrix4d& T_rob_cam)
{
    Eigen::Matrix4d T_cam_obj, T_rob_obj;

    if (!BuildFrameFrom3Points(camPts[0], camPts[1], camPts[2], T_cam_obj))
        return false;
    if (!BuildFrameFrom3Points(robPts[0], robPts[1], robPts[2], T_rob_obj))
        return false;

    T_rob_cam = T_rob_obj * T_cam_obj.inverse();
    return true;
}

} // namespace calib3
