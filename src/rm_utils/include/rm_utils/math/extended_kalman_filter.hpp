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


#ifndef RM_UTILS_KALMAN_FILTER_HPP_
#define RM_UTILS_KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>
#include "rm_utils/logger/log.hpp"
#include <deque>
#include <numeric>  // std::accumulate

namespace fyt {

class ExtendedKalmanFilter {
public:
  ExtendedKalmanFilter() = default;

  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  explicit ExtendedKalmanFilter(const VecVecFunc &f,
                                const VecVecFunc &h,
                                const VecMatFunc &j_f,
                                const VecMatFunc &j_h,
                                const VoidMatFunc &u_q,
                                const VecMatFunc &u_r,
                                const Eigen::MatrixXd &P0);

  // Set the initial state
  void setState(const Eigen::VectorXd &x0) noexcept;

  // Compute a predicted state
  Eigen::MatrixXd predict() noexcept;

  // Update the estimated state based on measurement
  Eigen::MatrixXd update(const Eigen::VectorXd &z) noexcept;

  //NIS门控  
  double lastNis()    const noexcept { return last_nis_; }
  bool   lastNisOk()  const noexcept { return last_nis_ok_; } 
  size_t total_updates_;
  size_t nis_fail_count_;
  std::deque<int> recent_nis_failures_;   


private:
  // Process nonlinear vector function
  VecVecFunc f;
  // Observation nonlinear vector function
  VecVecFunc h;
  // Jacobian of f()
  VecMatFunc jacobian_f;
  Eigen::MatrixXd F;
  // Jacobian of h()
  VecMatFunc jacobian_h;
  Eigen::MatrixXd H;
  // Process noise covariance matrix
  VoidMatFunc update_Q;
  Eigen::MatrixXd Q;
  // Measurement noise covariance matrix
  VecMatFunc update_R;
  Eigen::MatrixXd R;

  // Priori error estimate covariance matrix
  Eigen::MatrixXd P_pri;
  // Posteriori error estimate covariance matrix
  Eigen::MatrixXd P_post;

  // Kalman gain
  Eigen::MatrixXd K;

  // System dimensions
  int n;

  // N-size identity
  Eigen::MatrixXd I;

  // Priori state
  Eigen::VectorXd x_pri;
  // Posteriori state
  Eigen::VectorXd x_post;

  //NIS门控
  double last_nis_    = 0.0;
  bool   last_nis_ok_ = true;
  double nis_threshold_ = 9.5;   // 由外部设置，如卡方阈值
  std::size_t  nis_window_size_ = 100;  // 比如最近 100 次
  double recent_fail_ratio_ = 0.0;


};

}  // namespace fyt

#endif  // RM_UTILS_KALMAN_FILTER_HPP_
