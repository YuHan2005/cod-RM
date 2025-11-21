// Created by Labor 2023.8.25
// Maintained by Chengfu Zou, Labor
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "armor_detector/ba_solver.hpp"
// std
#include <memory>
// 3rd party
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
// project
#include "armor_detector/graph_optimizer.hpp"
#include "armor_detector/types.hpp"
#include "rm_utils/logger/log.hpp"
#include "rm_utils/math/utils.hpp"

namespace fyt::auto_aim {
G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(std::array<double, 9> &camera_matrix, std::vector<double> &dist_coeffs) {
  cam_internal_k_ = CameraInternalK{
    .fx = camera_matrix[0], .fy = camera_matrix[4], .cx = camera_matrix[2], .cy = camera_matrix[5]};

  // Optimization information
  optimizer_.setVerbose(false);
  // Optimization method
  optimizer_.setAlgorithm(
    g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solver_property_));
  // Initial step size
  lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
    const_cast<g2o::OptimizationAlgorithm *>(optimizer_.algorithm()));
  lm_algorithm_->setUserLambdaInit(0.1);
}



//AI提供的BA优化函数
bool BaSolver::solveBa(const std::deque<Armor> &armors, cv::Mat &rmat) noexcept {
  if (armors.empty()) {
    return true;
  }

  size_t MAX_WINDOW_SIZE = 1;
  // ====== 1. 限制小窗口：只用最近 N 帧 ======
  if(armors[armors.size()-1].Close_Center==true)
  {
      MAX_WINDOW_SIZE=5;
  }
  const size_t total_frames = armors.size();
  const size_t used_frames = std::min(total_frames, MAX_WINDOW_SIZE);

  auto it_begin = armors.end() - used_frames;  // 从队尾往前取 used_frames 帧

  // ====== 2. 重置优化器 ======
  optimizer_.clear();

  // 装甲板尺寸：这里假设窗口里类型一致，用最后一帧的类型也可以
  const Armor &ref_armor = armors.back();
  Eigen::Vector2d armor_size =
    (ref_armor.type == ArmorType::SMALL)
      ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
      : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);

  // ====== 3. 用最后一帧算一个初始 yaw，作为共享顶点的初值 ======
  {
    Eigen::Matrix3d camera2armor = utils::cvToEigen(ref_armor.rmat);
    Eigen::Matrix3d imu2camera   = ref_armor.imu2camera;
    Eigen::Matrix3d imu2armor    = imu2camera * camera2armor;

    auto theta_by_sin = std::asin(-imu2armor(0, 1));
    auto theta_by_cos = std::acos(imu2armor(1, 1));

    double initial_yaw;
    if (std::abs(theta_by_sin) > 1e-5) {
      initial_yaw = (theta_by_sin > 0) ? theta_by_cos : -theta_by_cos;
    } else {
      initial_yaw = (imu2armor(1, 1) > 0) ? 0.0 : CV_PI;
    }

    // ====== 4. 创建“一个共享 yaw 顶点” ======
    VertexYaw *v_yaw = new VertexYaw();
    v_yaw->setId(0);  // 只有一个 yaw 顶点，id 设成 0 即可
    Eigen::Matrix<double, 1, 1> yaw_vec;
    yaw_vec << initial_yaw;
    v_yaw->setEstimate(yaw_vec);
    optimizer_.addVertex(v_yaw);
  }

  // 取出刚刚添加的 yaw 顶点
  auto *v_yaw_shared = dynamic_cast<VertexYaw *>(optimizer_.vertex(0));

  // ====== 5. 为窗口内每一帧添加一个投影边（都连到同一个 yaw 顶点） ======
  int edge_id = 1;
  for (auto it = it_begin; it != armors.end(); ++it) {
    const Armor &armor = *it;

    // 坐标系转换
    Eigen::Matrix3d camera2armor = utils::cvToEigen(armor.rmat);
    Eigen::Matrix3d imu2camera   = armor.imu2camera;
    Eigen::Matrix3d camera2imu   = imu2camera.transpose();

    // 装甲板中心在相机坐标系下的位置（PnP 给的 tvec）
    Eigen::Vector3d armor_position_3d = utils::cvToEigen(armor.tvec);

    // pitch：仍然用你原来的 outpost 特殊逻辑
    double armor_pitch =
      (armor.number == "outpost" ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD);

    // 2D 角点观测打包成向量
    Eigen::Matrix<double, Armor::N_LANDMARKS_2, 1> armor_landmarks_2d;
    auto landmarks = armor.landmarks();
    for (size_t i = 0; i < Armor::N_LANDMARKS; ++i) {
      armor_landmarks_2d(2 * i)     = landmarks[i].x;
      armor_landmarks_2d(2 * i + 1) = landmarks[i].y;
    }

    // 边：同一个 yaw，多个相机位姿 / tvec / 角点观测
    EdgeProjection *edge = new EdgeProjection(
      Sophus::SO3d(camera2imu),
      armor_position_3d,
      cam_internal_k_,
      armor_size,
      armor_pitch);

    edge->setId(edge_id++);
    edge->setVertex(0, v_yaw_shared);  // 关键：所有边都连到同一个 yaw 顶点
    edge->setMeasurement(armor_landmarks_2d);
    edge->setInformation(EdgeProjection::InfoMatrixType::Identity());

    // 鲁棒核保持不变
    g2o::RobustKernel *robustKernel =
      g2o::RobustKernelFactory::instance()->construct("Fair");
    dynamic_cast<g2o::RobustKernelFair *>(robustKernel)->setDelta(2);
    edge->setRobustKernel(robustKernel);

    optimizer_.addEdge(edge);
  }

  // ====== 6. 开始优化 ======
  optimizer_.initializeOptimization();
  optimizer_.optimize(20);

  // ====== 7. 取出优化后的 yaw（共享顶点） ======
  double yaw_optimized = v_yaw_shared->estimate()(0);

  // ====== 8. 用“最后一帧”的 imu2camera + 优化 yaw + 固定 pitch 生成最终 R ======
  double pitch_optimized =
    (ref_armor.number == "outpost" ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD);

  Eigen::Vector3d euler_in_imu_frame(0.0, pitch_optimized, yaw_optimized);
  Eigen::Matrix3d imu2armor_optimized =
    utils::eulerToMatrix(euler_in_imu_frame, utils::EulerOrder::XYZ);

  Eigen::Matrix3d rmat_optimized =
    ref_armor.imu2camera.transpose() * imu2armor_optimized;

  rmat = utils::eigenToCv(rmat_optimized);

  return true;
}

//AI说这段写得有问题，但是暂时看不懂，尝试使用AI提供的代码
/*
bool BaSolver::solveBa(const std::deque<Armor> &armors, cv::Mat &rmat) noexcept {
  if (armors.empty()) {
    return true;
  }

  // Reset optimizer
  optimizer_.clear();

  auto initial_armor_size = armors.front().type == ArmorType::SMALL
                              ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
                              : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);

  int optimized_frame_number = armors.size();
  int id_counter = 0;
  for (const auto &armor : armors) {
    // Essential coordinate system transformation
    Eigen::Matrix3d camera2armor = utils::cvToEigen(armor.rmat);
    Eigen::Matrix3d imu2camera = armor.imu2camera;
    Eigen::Matrix3d imu2armor = imu2camera * camera2armor;
    Eigen::Matrix3d camera2imu = imu2camera.transpose();

    // Compute the initial yaw from rotation matrix
    Eigen::Vector<double, 1> initial_armor_yaw;
    auto theta_by_sin = std::asin(-imu2armor(0, 1));
    auto theta_by_cos = std::acos(imu2armor(1, 1));

    double initial_armor_pitch =
      armor.number == "outpost" ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;

    if (std::abs(theta_by_sin) > 1e-5) {
      initial_armor_yaw = Eigen::Vector<double, 1>(theta_by_sin > 0 ? theta_by_cos : -theta_by_cos);
    } else {
      initial_armor_yaw = Eigen::Vector<double, 1>(imu2armor(1, 1) > 0 ? 0 : CV_PI);
    }

    auto armor_position_3d = utils::cvToEigen(armor.tvec);

    Eigen::Matrix<double, Armor::N_LANDMARKS_2, 1> armor_landmarks_2d;
    auto landmarks = armor.landmarks();
    for (size_t i = 0; i < Armor::N_LANDMARKS; i++) {
      armor_landmarks_2d(2 * i) = landmarks[i].x;
      armor_landmarks_2d(2 * i + 1) = landmarks[i].y;
    }

    VertexYaw *v_yaw = new VertexYaw();
    v_yaw->setId(id_counter);
    v_yaw->setEstimate(initial_armor_yaw);
    optimizer_.addVertex(v_yaw);

    EdgeProjection *edge = new EdgeProjection(Sophus::SO3d(camera2imu),
                                              armor_position_3d,
                                              cam_internal_k_,
                                              initial_armor_size,
                                              initial_armor_pitch);
    edge->setId(id_counter + optimized_frame_number);
    edge->setVertex(0, v_yaw);
    edge->setMeasurement(armor_landmarks_2d);
    edge->setInformation(EdgeProjection::InfoMatrixType::Identity());

    // Kernel function selection : "Fair" "Huber"(and threshold value)
    g2o::RobustKernel *robustKernel;
    robustKernel = g2o::RobustKernelFactory::instance()->construct("Fair");
    dynamic_cast<g2o::RobustKernelFair *>(robustKernel)->setDelta(2);
    edge->setRobustKernel(robustKernel);
    optimizer_.addEdge(edge);

    id_counter++;
  }

  // Start optimizing
  optimizer_.initializeOptimization();
  optimizer_.optimize(20);

  // Get yaw angle after optimization
  double yaw_optimized =
    dynamic_cast<VertexYaw *>(optimizer_.vertex(id_counter - 1))->estimate()(0);

  // Get rotation under the camera coordinate system
  double pitch_optimized =
    armors.back().number == "outpost" ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;
  auto armor_euler_in_fixed_frame = Eigen::Vector3d(0, pitch_optimized, yaw_optimized);
  Eigen::Matrix3d imu2armor =
    utils::eulerToMatrix(armor_euler_in_fixed_frame, utils::EulerOrder::XYZ);

  Eigen::Matrix3d rmat_optimized = armors.back().imu2camera.transpose() * imu2armor;

  rmat = utils::eigenToCv(rmat_optimized);

  return true;
}
  */

}  // namespace fyt::auto_aim