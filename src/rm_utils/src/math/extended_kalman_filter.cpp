// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under Apache License 2.0.
//
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


#include "rm_utils/math/extended_kalman_filter.hpp"

namespace fyt {
ExtendedKalmanFilter::ExtendedKalmanFilter(const VecVecFunc &f,
                                           const VecVecFunc &h,
                                           const VecMatFunc &j_f,
                                           const VecMatFunc &j_h,
                                           const VoidMatFunc &u_q,
                                           const VecMatFunc &u_r,
                                           const Eigen::MatrixXd &P0)
: f(f)
, h(h)
, jacobian_f(j_f)
, jacobian_h(j_h)
, update_Q(u_q)
, update_R(u_r)
, P_post(P0)
, n(P0.rows())
, I(Eigen::MatrixXd::Identity(n, n))
, x_pri(n)
, x_post(n) {}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd &x0) noexcept { x_post = x0;}

Eigen::MatrixXd ExtendedKalmanFilter::predict() noexcept {
  F = jacobian_f(x_post), Q = update_Q();

  x_pri = f(x_post);
  P_pri = F * P_post * F.transpose() + Q;

  // handle the case when there will be no measurement before the next predict
  x_post = x_pri;
  P_post = P_pri;

  return x_pri;
}

/*
Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd &z) noexcept {
  H = jacobian_h(x_pri), R = update_R(z);

  K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();
  x_post = x_pri + K * (z - h(x_pri));
  P_post = (I - K * H) * P_pri;

  return x_post;
}*/

//添加了NIS门控的update函数

Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd &z) noexcept
{
  total_updates_++;
  const int m = z.size();  // 观测维度

  // 1) 基于先验 x_pri 计算 H、R、预测量测
  H = jacobian_h(x_pri);
  R = update_R(z);

  Eigen::VectorXd z_pred = h(x_pri);      // h(x_pri)
  Eigen::VectorXd y      = z - z_pred;    // 残差 y = z - h(x_pri)

  // 2) 残差协方差 S = H P_pri H^T + R
  Eigen::MatrixXd S = H * P_pri * H.transpose() + R;

  // 用 LDLT 分解，后面既算 S^{-1}y，又算 S^{-1}
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);

  // 3) 计算 NIS = y^T S^{-1} y
  Eigen::VectorXd S_inv_y = ldlt.solve(y);    // S^{-1} y
  last_nis_ = y.dot(S_inv_y);                 // NIS

  last_nis_ok_ = true;
  int fail_flag = 0;  // 这次 update 是否算失败（NIS 超门）0/1

  // ==== NIS 门控 ====
  if (nis_threshold_ > 0.0 && last_nis_ > nis_threshold_) {
    // 本次观测在统计意义上“不一致”，拒绝更新
    last_nis_ok_ = false;
    fail_flag    = 1;
    nis_fail_count_++;

    // 先验 -> 后验
    x_post = x_pri;
    P_post = P_pri;

    // 更新滑动窗口统计
    recent_nis_failures_.push_back(fail_flag);
    if (recent_nis_failures_.size() > nis_window_size_) {
      recent_nis_failures_.pop_front();
    }
    int sum_fail = std::accumulate(
      recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
    recent_fail_ratio_ = recent_nis_failures_.empty()
                           ? 0.0
                           : static_cast<double>(sum_fail) / recent_nis_failures_.size();

    FYT_DEBUG("ekf",
              "NIS = {:.2f}, fail_ratio_recent = {:.1f}%",
              last_nis_, 100.0 * recent_fail_ratio_);

    return x_post;
  }
  // ==================

  // 4) 计算 Kalman 增益 K = P_pri H^T S^{-1}
  Eigen::MatrixXd PHt   = P_pri * H.transpose();
  Eigen::MatrixXd I_m   = Eigen::MatrixXd::Identity(m, m);
  Eigen::MatrixXd S_inv = ldlt.solve(I_m);    // S^{-1}
  K = PHt * S_inv;

  // 5) 状态更新 x_post = x_pri + K y
  x_post = x_pri + K * y;

  // 6) 用 Joseph 形式更新协方差，数值更稳定
  //    P_post = (I - K H) P_pri (I - K H)^T + K R K^T
  Eigen::MatrixXd I_n  = I;  // 或 Eigen::MatrixXd::Identity(P_pri.rows(), P_pri.cols());
  Eigen::MatrixXd I_KH = I_n - K * H;
  P_post = I_KH * P_pri * I_KH.transpose() + K * R * K.transpose();

  // 对称化
  P_post = 0.5 * (P_post + P_post.transpose());

  // ---- 通过门控的情况，同样更新滑动窗口 ----
  recent_nis_failures_.push_back(fail_flag);  // 这里 fail_flag = 0
  if (recent_nis_failures_.size() > nis_window_size_) {
    recent_nis_failures_.pop_front();
  }
  int sum_fail = std::accumulate(
    recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
  recent_fail_ratio_ = recent_nis_failures_.empty()
                         ? 0.0
                         : static_cast<double>(sum_fail) / recent_nis_failures_.size();

  FYT_DEBUG("ekf",
            "NIS = {:.2f}, fail_ratio_recent = {:.1f}%",
            last_nis_, 100.0 * recent_fail_ratio_);

  return x_post;
}



}  // namespace fyt
