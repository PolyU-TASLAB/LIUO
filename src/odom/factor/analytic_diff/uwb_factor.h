/*
 * LIUO: Outlier-Aware LiDAR/IMU/UWB Integration with NLOS Mitigation and Anchor Self-Calibration in Complex Scenarios
 * Copyright (C) 2026 Xiao Naigui
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <ceres/ceres.h>
#include <odom/msg_manager.h>
#include <spline/trajectory.h>
#include <utils/parameter_struct.h>

#include "split_spline_view.h"

namespace liuo {
namespace analytic_derivative {

class UWBFactorNURBS : public ceres::CostFunction {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  UWBFactorNURBS(int64_t time_uwb_ns,
                 const std::pair<const int, double>& dist_pair,
                 const std::pair<int, Eigen::Vector3d>& anchor_pos_pair,
                 const std::pair<int, double>& su,
                 const Eigen::Matrix4d& blending_matrix,
                 const Eigen::Matrix4d& cumulative_blending_matrix,
                 const SO3d& S_GtoM, const Vec3d& p_GinM, const SO3d& S_UtoI,
                 const Vec3d& p_UinI, double weight)
      : time_uwb_ns_(time_uwb_ns),
        dist_pair_(dist_pair),
        anchor_pos_pair_(anchor_pos_pair),
        su_(su),
        blending_matrix_(blending_matrix),
        cumulative_blending_matrix_(cumulative_blending_matrix),
        S_GtoM_(S_GtoM),
        p_GinM_(p_GinM),
        S_UtoI_(S_UtoI),
        p_UinI_(p_UinI),
        weight_(weight) {
    // LOG(INFO) << "UWB Factor Constructor";
    set_num_residuals(1);

    size_t kont_num = 4;
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
  // TODO: Core implementation available upon acceptance.

    return true;
  }

 private:
  int64_t time_uwb_ns_;
  std::pair<const int, double> dist_pair_;
  std::pair<int, Eigen::Vector3d> anchor_pos_pair_;
  // UwbData uwb_data_;
  std::pair<int, double> su_;
  Eigen::Matrix4d blending_matrix_;
  Eigen::Matrix4d cumulative_blending_matrix_;
  Eigen::Vector3d uwb_measurement_;
  Eigen::Matrix3d K_;
  SO3d S_UtoI_;
  Vec3d p_UinI_;
  SO3d S_GtoM_;
  Vec3d p_GinM_;
  double weight_;
};

class UWBFactorNURBSAutoDiff : public ceres::CostFunction {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  UWBFactorNURBSAutoDiff(int64_t time_uwb_ns, const UwbData& uwb_data,
                         const std::pair<int, double>& su,
                         const Eigen::Matrix4d& blending_matrix,
                         const Eigen::Matrix4d& cumulative_blending_matrix,
                         const SO3d& S_GtoM, const Vec3d& p_GinM,
                         const SO3d& S_UtoI, const Vec3d& p_UinI, double weight)
      : time_uwb_ns_(time_uwb_ns),
        uwb_data_(uwb_data),
        su_(su),
        blending_matrix_(blending_matrix),
        cumulative_blending_matrix_(cumulative_blending_matrix),
        S_GtoM_(S_GtoM),
        p_GinM_(p_GinM),
        S_UtoI_(S_UtoI),
        p_UinI_(p_UinI),
        weight_(weight) {
    // LOG(INFO) << "UWB Factor Constructor";
    set_num_residuals(1);

    size_t kont_num = 4;
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);
    }
    for (size_t i = 0; i < kont_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);
    }
  }

  virtual bool operator()(double const* const* parameters, double* residuals,
                          double** jacobians) const {
   // TODO: Core implementation available upon acceptance.

    return true;
  }

 private:
  int64_t time_uwb_ns_;
  UwbData uwb_data_;
  std::pair<int, double> su_;
  Eigen::Matrix4d blending_matrix_;
  Eigen::Matrix4d cumulative_blending_matrix_;
  Eigen::Vector3d uwb_measurement_;
  Eigen::Matrix3d K_;
  SO3d S_UtoI_;
  Vec3d p_UinI_;
  SO3d S_GtoM_;
  Vec3d p_GinM_;
  double weight_;
};

class UWBFactorWithAnchorOptNURBS : public ceres::CostFunction {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vec3d = Eigen::Matrix<double, 3, 1>;
  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Mat3d = Eigen::Matrix<double, 3, 3>;
  using SO3d = Sophus::SO3<double>;

  UWBFactorWithAnchorOptNURBS(int64_t time_uwb_ns,
                              const std::pair<const int, double>& dist_pair,
                              const std::pair<int, double>& su,
                              const Eigen::Matrix4d& blending_matrix,
                              const Eigen::Matrix4d& cumulative_blending_matrix,
                              const SO3d& S_GtoM, const Vec3d& p_GinM,
                              const SO3d& S_UtoI, const Vec3d& p_UinI,
                              double weight)
      : time_uwb_ns_(time_uwb_ns),
        dist_pair_(dist_pair),
        su_(su),
        blending_matrix_(blending_matrix),
        cumulative_blending_matrix_(cumulative_blending_matrix),
        S_GtoM_(S_GtoM),
        p_GinM_(p_GinM),
        S_UtoI_(S_UtoI),
        p_UinI_(p_UinI),
        weight_(weight) {
    set_num_residuals(1);

    // 4 rotation control points + 4 position control points + 1 anchor position
    size_t knot_num = 4;
    for (size_t i = 0; i < knot_num; ++i) {
      mutable_parameter_block_sizes()->push_back(4);  // rotation knots
    }
    for (size_t i = 0; i < knot_num; ++i) {
      mutable_parameter_block_sizes()->push_back(3);  // position knots
    }
    mutable_parameter_block_sizes()->push_back(3);  // anchor position
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
  // TODO: Core implementation available upon acceptance.

    return true;
  }

 private:
  int64_t time_uwb_ns_;
  std::pair<const int, double> dist_pair_;
  std::pair<int, double> su_;
  Eigen::Matrix4d blending_matrix_;
  Eigen::Matrix4d cumulative_blending_matrix_;
  SO3d S_UtoI_;
  Vec3d p_UinI_;
  SO3d S_GtoM_;
  Vec3d p_GinM_;
  double weight_;
};

class UWBAnchorPriorFactor : public ceres::CostFunction {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vec3d = Eigen::Matrix<double, 3, 1>;

  UWBAnchorPriorFactor(const Vec3d& prior_pos, double weight)
      : prior_pos_(prior_pos), weight_(weight) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(3);
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
  // TODO: Core implementation available upon acceptance.


    return true;
  }

 private:
  const Vec3d prior_pos_;
  const double weight_;
};

}  // namespace analytic_derivative

// namespace auto_diff {

// }  // namespace auto_diff

}  // namespace liuo