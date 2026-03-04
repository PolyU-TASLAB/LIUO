/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera
 * Odometry using Non-Uniform B-spline 
 * Copyright (C) 2023 Xiaolei Lang
 *
 * LIUO: Outlier-Aware LiDAR/IMU/UWB Integration with NLOS Mitigation 
 * and Anchor Self-Calibration in Complex Scenarios
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

#include "odom/trajectory_manager.h"
#include <boost/filesystem.hpp>

#include <ros/assert.h>
#include <fstream>
#include <iomanip>
#include <random>

#include "lidar/lidar_handler.h"
#include "odom/odometry_viewer.hpp"
// #include "odom/factor/analytic_diff/imu_factor.h"
// #include "odom/factor/analytic_diff/reprojection_factor.h"
#include "odom/factor/analytic_diff/trajectory_value_factor.h"
#include "odom/factor/analytic_diff/uwb_factor.h"
#include "utils/log_utils.h"
std::fstream myfile_t_ba;
namespace liuo {

TrajectoryManager::TrajectoryManager(const YAML::Node &node,
                                     Trajectory::Ptr trajectory,
                                     const std::string &config_file_path)
    : verbose(false), cur_img_time_(-1), process_cur_img_(false),
      opt_weight_(OptWeight(node)), trajectory_(trajectory),
      lidar_marg_info(nullptr), cam_marg_info(nullptr),
      odometry_viewer_(nullptr) {
  
  // Resolve config path relative to the config file itself
  boost::filesystem::path config_file_p(config_file_path);
  std::string config_path = config_file_p.parent_path().string();
  
  if (!config_path.empty() && config_path.back() != '/') {
      config_path += "/";
  }

  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  std::string imu_yaml_path = config_path + imu_yaml;
  // std::cout << "Loading IMU config: " << imu_yaml_path << std::endl;
  YAML::Node imu_node = YAML::LoadFile(imu_yaml_path);
  imu_state_estimator_ = std::make_shared<ImuStateEstimator>(imu_node);

  if_use_init_bg_ = imu_node["if_use_init_bg"].as<bool>();

  lidar_prior_ctrl_id = std::make_pair(0, 0);

  InitFactorInfo(trajectory_->GetSensorEP(LiDARSensor),
                 opt_weight_.local_velocity_info_vec);

  division_ = 0;
  use_marg_ = true;

  opt_cnt = 0;
  t_opt_sum = 0.0;

  v_points_.clear();
  px_obss_.clear();

  // Initialize LidarHandler pointer as null
  lidar_handler_ = nullptr;

  // Initialize UWB NLOS detector
  nlos_detector_ = std::make_shared<UWBNLOSDetector>(node);
  LOG(INFO) << "UWB NLOS Detector initialized for "
               "LiDAR-IMU-UWB fusion.";

  // Initialize UWB anchor positions from config if available
  if (node["uwb_yaml"]) {
    std::string uwb_yaml = node["uwb_yaml"].as<std::string>();
    std::string uwb_yaml_path = config_path + uwb_yaml;
    // std::cout << "Loading UWB config: " << uwb_yaml_path << std::endl;
    YAML::Node uwb_node = YAML::LoadFile(uwb_yaml_path);
    if (uwb_node["AnchorId"] && uwb_node["AnchorPos"] &&
        uwb_node["Extrinsics"]) {
      std::vector<int> anchor_ids = uwb_node["AnchorId"].as<std::vector<int>>();
      std::vector<double> anchor_positions =
          uwb_node["AnchorPos"].as<std::vector<double>>();
      Eigen::Vector3d trans_(uwb_node["Extrinsics"]["Trans"][0].as<double>(),
                             uwb_node["Extrinsics"]["Trans"][1].as<double>(),
                             uwb_node["Extrinsics"]["Trans"][2].as<double>());
      Eigen::Matrix3d rot_ = Eigen::Matrix3d::Identity();
      Eigen::Isometry3d extrinsics_ = Eigen::Isometry3d::Identity();
      rot_ << uwb_node["Extrinsics"]["Rot"][0].as<double>(),
          uwb_node["Extrinsics"]["Rot"][1].as<double>(),
          uwb_node["Extrinsics"]["Rot"][2].as<double>(),
          uwb_node["Extrinsics"]["Rot"][3].as<double>(),
          uwb_node["Extrinsics"]["Rot"][4].as<double>(),
          uwb_node["Extrinsics"]["Rot"][5].as<double>(),
          uwb_node["Extrinsics"]["Rot"][6].as<double>(),
          uwb_node["Extrinsics"]["Rot"][7].as<double>(),
          uwb_node["Extrinsics"]["Rot"][8].as<double>();
      extrinsics_.rotate(rot_);
      extrinsics_.pretranslate(trans_);
      // Read noise configuration if available
      double noise_std_x = 0.0, noise_std_y = 0.0, noise_std_z = 0.0;
      bool add_noise = false;
      if (uwb_node["anchor_noise_std"]) {
        if (uwb_node["anchor_noise_std"].IsScalar()) {
          // Single value for all axes
          double noise_std = uwb_node["anchor_noise_std"].as<double>();
          if (noise_std > 0.0) {
            noise_std_x = noise_std;
            noise_std_y = noise_std;
            noise_std_z = noise_std;
            add_noise = true;
          }
        } else if (uwb_node["anchor_noise_std"].IsSequence() &&
                   uwb_node["anchor_noise_std"].size() == 3) {
          // Three values for x, y, z axes
          noise_std_x = uwb_node["anchor_noise_std"][0].as<double>();
          noise_std_y = uwb_node["anchor_noise_std"][1].as<double>();
          noise_std_z = uwb_node["anchor_noise_std"][2].as<double>();
          add_noise =
              (noise_std_x > 0.0 || noise_std_y > 0.0 || noise_std_z > 0.0);
        }
      }

      std::map<int, Eigen::Vector3d> initial_positions;
      for (size_t i = 0; i < anchor_ids.size(); i++) {
        Eigen::Vector3d pos(anchor_positions[i * 3],
                            anchor_positions[i * 3 + 1],
                            anchor_positions[i * 3 + 2]);
        Eigen::Vector3d pos_body = extrinsics_ * pos;

        // Add Gaussian noise if configured
        if (add_noise) {
          Eigen::Vector3d pos_body_original = pos_body;
          prior_anchor_positions_[anchor_ids[i]] = pos_body_original;
          pos_body = AddGaussianNoiseToAnchorPosition(pos_body, noise_std_x,
                                                      noise_std_y, noise_std_z);
          LOG(INFO) << "UWB anchor position initialized with noise. anchor_id: "
                    << anchor_ids[i]
                    << ", original: " << pos_body_original.transpose()
                    << ", noisy: " << pos_body.transpose() << ", noise std: ["
                    << noise_std_x << ", " << noise_std_y << ", " << noise_std_z
                    << "]";
        } else {
          LOG(INFO) << "UWB anchor position initialized. anchor_id: "
                    << anchor_ids[i] << ", position: " << pos_body.transpose();
        }

        initial_positions[anchor_ids[i]] = pos_body;
      }
      InitializeUWBAnchorPositions(initial_positions);
    }
    // Load single anchor optimization settings from uwb_node
    if (uwb_node["optimize_one_uwb_anchor"]) {
      opt_weight_.optimize_one_uwb_anchor =
          uwb_node["optimize_one_uwb_anchor"].as<bool>();
      if (opt_weight_.optimize_one_uwb_anchor &&
          uwb_node["target_uwb_anchor_id"]) {
        opt_weight_.target_uwb_anchor_id =
            uwb_node["target_uwb_anchor_id"].as<int>();
      }
    }
  }

  // Integrity Monitoring Configuration Removed
    enable_residual_block_logging_ =
      node["enable_residual_block_logging"]
        ? node["enable_residual_block_logging"].as<bool>()
        : false;
    solve_num_threads_ =
      node["solve_num_threads"]
        ? std::max(1, node["solve_num_threads"].as<int>())
        : 1;
    solve_function_tolerance_ =
      node["solve_function_tolerance"]
        ? node["solve_function_tolerance"].as<double>()
        : 1e-4;
    solve_gradient_tolerance_ =
      node["solve_gradient_tolerance"]
        ? node["solve_gradient_tolerance"].as<double>()
        : 1e-10;
    solve_parameter_tolerance_ =
      node["solve_parameter_tolerance"]
        ? node["solve_parameter_tolerance"].as<double>()
        : 1e-8;

    LOG(INFO) << "Solver tuning | residual_log="
        << (enable_residual_block_logging_ ? "on" : "off")
      << ", solve_num_threads=" << solve_num_threads_
        << ", function_tolerance=" << solve_function_tolerance_
        << ", gradient_tolerance=" << solve_gradient_tolerance_
        << ", parameter_tolerance=" << solve_parameter_tolerance_;

    LOG(INFO) << "========================================";
    LOG(INFO) << "";

}

void TrajectoryManager::InitFactorInfo(
    const ExtrinsicParam &Ep_LtoI,
    const Eigen::Vector3d &local_velocity_weight) {
  analytic_derivative::LoamFeatureOptMapPoseFactor::SetParam(Ep_LtoI.so3,
                                                             Ep_LtoI.p);
  analytic_derivative::RalativeLoamFeatureFactor::SetParam(Ep_LtoI.so3,
                                                           Ep_LtoI.p);
}

void TrajectoryManager::SetSystemState(const SystemState &sys_state,
                                       double distance0) {
  gravity_ = sys_state.g;

  SetOriginalPose(sys_state.q, sys_state.p);

  trajectory_->AddKntNs(0.0 * S_TO_NS); // add knot t3   （t0、t1、t2 have been
                                        // added in the constructor）
  trajectory_->AddKntNs(distance0 * S_TO_NS); // add knot t4
  trajectory_->SetMaxTimeNsNURBS(trajectory_->knts.back());

  SO3d R0(sys_state.q);
  for (size_t i = 0; i < trajectory_->numKnots(); i++) {
    trajectory_->setKnotSO3(R0, i);
  }
  LOG(INFO) << "[debug numKnots] " << trajectory_->numKnots(); // 4

  tparam_.last_bias_time = trajectory_->maxTimeNsNURBS();
  tparam_.cur_bias_time = trajectory_->maxTimeNsNURBS();
  LOG(INFO) << "[SetSystemState] [tparam_.last_bias_time] "
            << tparam_.last_bias_time << " [tparam_.cur_bias_time] "
            << tparam_.cur_bias_time;

  // TODO
  all_imu_bias_[tparam_.last_bias_time] = sys_state.bias;
  if (!if_use_init_bg_) {
    all_imu_bias_[tparam_.last_bias_time].gyro_bias = Eigen::Vector3d::Zero();
    all_imu_bias_[tparam_.last_bias_time].accel_bias = Eigen::Vector3d::Zero();
  }
}

void TrajectoryManager::SetOriginalPose(Eigen::Quaterniond q,
                                        Eigen::Vector3d p) {
  original_pose_.orientation.setQuaternion(q);
  original_pose_.position = p;
}

void TrajectoryManager::AddIMUData(const IMUData &data) {
  if (trajectory_->GetDataStartTime() < 0) {
    trajectory_->SetDataStartTime(data.timestamp);
  }
  imu_data_.emplace_back(data);
  imu_data_.back().timestamp -= trajectory_->GetDataStartTime();

  imu_state_estimator_->FeedIMUData(imu_data_.back());
}

void TrajectoryManager::AddUWBData(const UwbData &data) {
  uwb_data_.emplace_back(data);
  // LOG(INFO) << "[AddUWBData] " << uwb_data_.back().timestamp << " "
  //           << trajectory_->GetDataStartTime();
  uwb_data_.back().timestamp -= trajectory_->GetDataStartTime();
}

void TrajectoryManager::AddPoseData(const PoseData &data) {
  pose_data_.emplace_back(data);
  pose_data_.back().timestamp -= trajectory_->GetDataStartTime();
}

void TrajectoryManager::RemoveIMUData(int64_t t_window_min) {
  if (t_window_min < 0)
    return;

  // https://stackoverflow.com/questions/991335/
  // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < t_window_min) {
      iter = imu_data_.erase(iter);
    } else {
      break;
    }
  }
}

void TrajectoryManager::RemoveUWBData(int64_t t_window_min) {
  if (t_window_min < 0)
    return;

  // https://stackoverflow.com/questions/991335/
  // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
  for (auto iter = uwb_data_.begin(); iter != uwb_data_.end();) {
    if (iter->timestamp < t_window_min) {
      iter = uwb_data_.erase(iter);
    } else {
      break;
    }
  }
}

void TrajectoryManager::RemovePoseData(int64_t t_window_min) {
  if (t_window_min < 0)
    return;

  // https://stackoverflow.com/questions/991335/
  // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
  for (auto iter = pose_data_.begin(); iter != pose_data_.end();) {
    if (iter->timestamp < t_window_min) {
      iter = pose_data_.erase(iter);
    } else {
      break;
    }
  }
}

void TrajectoryManager::UpdateIMUInlio() {
  int64_t t_min = opt_min_t_ns;
  int64_t t_max = opt_max_t_ns;

  for (auto iter = imu_data_.begin(); iter != imu_data_.end(); ++iter) {
    if (iter->timestamp >= t_min) {
      if (iter->timestamp >= t_max) {
        continue;
      }
      tparam_.lio_imu_idx[0] = std::distance(imu_data_.begin(), iter);
      tparam_.lio_imu_time[0] = iter->timestamp;
      break;
    }
  }

  for (auto rter = imu_data_.rbegin(); rter != imu_data_.rend(); ++rter) {
    if (rter->timestamp < t_max) {
      tparam_.lio_imu_idx[1] =
          std::distance(imu_data_.begin(), rter.base()) - 1;
      tparam_.lio_imu_time[1] = rter->timestamp;
      break;
    }
  }
}

void TrajectoryManager::PredictTrajectory(int64_t scan_time_min,
                                          int64_t scan_time_max,
                                          int64_t traj_max_time_ns,
                                          int knot_add_num, bool non_uniform) {
  if (imu_data_.empty() || imu_data_.size() == 1) {
    LOG(ERROR) << "[AppendWithIMUData] IMU data empty! ";
    return;
  }

  /// newly added interval：[opt_min_t_ns, opt_max_t_ns)
  opt_min_t_ns = trajectory_->maxTimeNsNURBS();

  /// extend trajectory by adding control points
  trajectory_->SetMaxTimeNsNURBS(traj_max_time_ns);
  opt_max_t_ns = trajectory_->maxTimeNsNURBS();
  SE3d last_knot = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(knot_add_num, last_knot);

  ////// color control point for visualization
  int intensity = 0;
  if (knot_add_num == 1) {
    intensity = 100;
  } else if (knot_add_num == 2) {
    intensity = 200;
  } else if (knot_add_num == 3) {
    intensity = 300;
  } else if (knot_add_num == 4) {
    intensity = 400;
  }
  for (int i = 0; i < knot_add_num; i++) {
    trajectory_->intensity_map[trajectory_->numKnots() + i] = intensity;
  }
  ////// color control point for visualization

  LOG(INFO) << "[max_time_ns] " << opt_max_t_ns;
  LOG(INFO) << "[numKnots aft extension] " << trajectory_->numKnots();

  tparam_.last_bias_time = tparam_.cur_bias_time; // opt_min_t_ns
  tparam_.cur_bias_time = opt_max_t_ns;
  LOG(INFO) << "[last_bias_time] " << tparam_.last_bias_time << " "
            << "[cur_bias_time] " << tparam_.cur_bias_time;
  LOG(INFO) << "[PredictTrajectory] traj_max_time_ns input: "
            << traj_max_time_ns << ", trajectory_->maxTimeNsNURBS(): "
            << trajectory_->maxTimeNsNURBS()
            << ", opt_max_t_ns: " << opt_max_t_ns;
  tparam_.UpdateCurScan(scan_time_min, scan_time_max);
  UpdateIMUInlio(); // determine the imu data involved in this optimization

  /// optimization
  InitTrajWithPropagation();
}

void TrajectoryManager::InitTrajWithPropagation() {
  TrajectoryEstimatorOptions option;
  option.lock_ab = true;
  option.lock_wb = true;
  option.lock_g = true;
  option.lock_tran = false; // note
  option.show_residual_summary = verbose;
  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "Init Traj"));

  // // Configure integrity monitoring from YAML config


  estimator->SetFixedIndex(3);

  // [0] prior factor
  if (true && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // [1] imu factor
  double *para_bg = all_imu_bias_.rbegin()->second.gyro_bias.data();
  double *para_ba = all_imu_bias_.rbegin()->second.accel_bias.data();
  for (int i = tparam_.lio_imu_idx[0]; i <= tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i), para_bg, para_ba,
                                              gravity_.data(), //(0, 0, 9.8)
                                              opt_weight_.imu_info_vec);
  }

  ceres::Solver::Summary summary =
      estimator->Solve(50, false, solve_num_threads_);
  static int init_cnt = 0;
  init_cnt++;
  LOG(INFO) << init_cnt << " TrajInitSolver " << summary.BriefReport();
  LOG(INFO) << init_cnt << " TrajInit Successful/Unsuccessful steps: "
            << summary.num_successful_steps << "/"
            << summary.num_unsuccessful_steps;
}

bool TrajectoryManager::UpdateTrajectoryWithLIC(
    int lidar_iter, int64_t img_time_stamp,
    const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
    const Eigen::aligned_vector<Eigen::Vector3d> &pnp_3ds,
    const Eigen::aligned_vector<Eigen::Vector2d> &pnp_2ds,
    const int iteration) {
  if (point_corrs.empty() || imu_data_.empty() || imu_data_.size() == 1) {
    LOG(WARNING) << " input empty data " << point_corrs.size() << ", "
                 << imu_data_.size();
    return false;
  }

  LOG(INFO) << "[point_corrs_size]: " << point_corrs.size();
  // LOG(INFO) << "[opt_domain]: "
  //           << "[" << opt_min_t_ns * NS_TO_S << ", " << opt_max_t_ns *
  //           NS_TO_S
  //           << ")";

  IMUBias last_bias = all_imu_bias_.rbegin()->second;
  all_imu_bias_[tparam_.cur_bias_time] = last_bias;
  std::map<int, double *> para_bg_vec;
  std::map<int, double *> para_ba_vec;
  {
    auto &bias0 = all_imu_bias_[tparam_.last_bias_time]; // bi
    para_bg_vec[0] = bias0.gyro_bias.data();
    para_ba_vec[0] = bias0.accel_bias.data();

    auto &bias1 = all_imu_bias_[tparam_.cur_bias_time]; // bj
    para_bg_vec[1] = bias1.gyro_bias.data();
    para_ba_vec[1] = bias1.accel_bias.data();
  }

  TrajectoryEstimatorOptions option;
  option.lock_ab = false;
  option.lock_wb = false;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  option.enable_residual_block_logging = enable_residual_block_logging_;
  option.function_tolerance = solve_function_tolerance_;
  option.gradient_tolerance = solve_gradient_tolerance_;
  option.parameter_tolerance = solve_parameter_tolerance_;
  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "Before LIO"));

  estimator->SetFixedIndex(3);

  // [0] prior factor
  if (true && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // [1] lidar factor
  SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

  for (const auto &v : point_corrs) {
    if (v.t_point < opt_min_t_ns)
      continue;
    if (v.t_point >= opt_max_t_ns)
      continue;
    if (v.t_point < tparam_.last_scan[1])
      continue;
    if (use_lidar_scale) {
      estimator->AddLoamMeasurementAnalyticNURBS(
          v, S_GtoM, p_GinM, S_LtoI, p_LinI,
          opt_weight_.lidar_weight * v.scale);
    } else {
      estimator->AddLoamMeasurementAnalyticNURBS(
          v, S_GtoM, p_GinM, S_LtoI, p_LinI, opt_weight_.lidar_weight);
    }
  }

  // [2] imu factor
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i), para_bg_vec[0],
                                              para_ba_vec[0], gravity_.data(),
                                              opt_weight_.imu_info_vec);
  }

  /// [3] bias factor
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> noise_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  noise_covariance.block<3, 3>(0, 0) =
      (opt_weight_.imu_noise.sigma_wb_discrete *
       opt_weight_.imu_noise.sigma_wb_discrete) *
      Eigen::Matrix3d::Identity();
  noise_covariance.block<3, 3>(3, 3) =
      (opt_weight_.imu_noise.sigma_ab_discrete *
       opt_weight_.imu_noise.sigma_ab_discrete) *
      Eigen::Matrix3d::Identity();
  for (int i = tparam_.lio_imu_idx[0] + 1; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i - 1).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    double dt = (imu_data_[i].timestamp - imu_data_[i - 1].timestamp) * NS_TO_S;
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
    G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt;
    G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt;
    covariance =
        F * covariance * F.transpose() + G * noise_covariance * G.transpose();
  }

  Eigen::Matrix<double, 6, 6> sqrt_info_mat =
      Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse())
          .matrixL()
          .transpose();
  sqrt_info_ << sqrt_info_mat(0, 0), sqrt_info_mat(1, 1), sqrt_info_mat(2, 2),
      sqrt_info_mat(3, 3), sqrt_info_mat(4, 4), sqrt_info_mat(5, 5);
  // Only add bias factor if we have different bias times (avoid duplicate
  // parameter blocks)
  if (tparam_.last_bias_time != tparam_.cur_bias_time) {
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                             para_ba_vec[1], 1, sqrt_info_);
  } else {
    // In realtime mode, same bias time is common at initialization
    // We can skip the bias factor or use a regularization approach
    LOG(INFO) << "Skipping bias factor: same bias time "
              << tparam_.last_bias_time << " (common in realtime mode)";

    // Alternative: Add a weak regularization factor to prevent bias drift
    // This helps maintain bias stability when we don't have temporal
    // constraints
    Eigen::Matrix<double, 6, 1> weak_info =
        sqrt_info_ * 0.1; // 10x weaker constraint
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[0], para_ba_vec[0],
                             para_ba_vec[0], 0.01, weak_info); // Very small dt
  }

  TicToc t_opt;
  static int loam_cnt = 0;
  ceres::Solver::Summary summary =
      estimator->Solve(iteration, false, solve_num_threads_);
  double opt_time = t_opt.toc();
  LOG(INFO) << "[t_opt] " << opt_time << std::endl;
  LOG(INFO) << "LoamSolver " << summary.BriefReport();
  LOG(INFO) << ++loam_cnt << " UpdateLio Successful/Unsuccessful steps: "
            << summary.num_successful_steps << "/"
            << summary.num_unsuccessful_steps;

  opt_cnt++;
  t_opt_sum += opt_time;

  LOG(INFO) << "[gyro_bias_new] "
            << all_imu_bias_.rbegin()->second.gyro_bias.x() << " "
            << all_imu_bias_.rbegin()->second.gyro_bias.y() << " "
            << all_imu_bias_.rbegin()->second.gyro_bias.z();
  LOG(INFO) << "[acce_bias_new] "
            << all_imu_bias_.rbegin()->second.accel_bias.x() << " "
            << all_imu_bias_.rbegin()->second.accel_bias.y() << " "
            << all_imu_bias_.rbegin()->second.accel_bias.z();

  return true;
}

bool TrajectoryManager::UpdateTrajectoryWithLICU(
    int lidar_iter, int64_t img_time_stamp, int64_t uwb_time_stamp,
    const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
    const Eigen::aligned_vector<Eigen::Vector3d> &pnp_3ds,
    const Eigen::aligned_vector<Eigen::Vector2d> &pnp_2ds,
    const int iteration) {
  if (point_corrs.empty() || imu_data_.empty() || imu_data_.size() == 1) {
    LOG(WARNING) << " input empty data " << point_corrs.size() << ", "
                 << imu_data_.size();
    return false;
  }

  // ===== Scheme 2: Update Point Cloud Map before UWB optimization (for NLOS detection) =====
  if (nlos_detector_ && lidar_handler_ && !uwb_data_.empty()) {
    static int update_counter = 0;
    if (++update_counter % 3 == 0) { // Update once every 3 optimizations
      PosCloud::Ptr current_map = BuildPointCloudMapForNLOS();
      if (current_map && !current_map->empty()) {
        nlos_detector_->UpdatePointCloudMap(current_map);
        LOG(INFO) << "[UpdateTrajectoryWithLICU] Updated NLOS detector map "
                  << "with " << current_map->size() << " points";
      }
    }
  }
  // ===== End Point Cloud Map Update =====

  LOG(INFO) << "[point_corrs size] " << point_corrs.size();
  LOG(INFO) << "[opt_domain]: "
            << "[" << opt_min_t_ns * NS_TO_S << ", " << opt_max_t_ns * NS_TO_S
            << ")";

  IMUBias last_bias = all_imu_bias_.rbegin()->second;
  all_imu_bias_[tparam_.cur_bias_time] = last_bias;
  std::map<int, double *> para_bg_vec;
  std::map<int, double *> para_ba_vec;
  {
    auto &bias0 = all_imu_bias_[tparam_.last_bias_time]; // bi
    para_bg_vec[0] = bias0.gyro_bias.data();
    para_ba_vec[0] = bias0.accel_bias.data();

    auto &bias1 = all_imu_bias_[tparam_.cur_bias_time]; // bj
    para_bg_vec[1] = bias1.gyro_bias.data();
    para_ba_vec[1] = bias1.accel_bias.data();
  }

  TrajectoryEstimatorOptions option;
  option.lock_ab = false;
  option.lock_wb = false;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  option.enable_residual_block_logging = enable_residual_block_logging_;
  option.function_tolerance = solve_function_tolerance_;
  option.gradient_tolerance = solve_gradient_tolerance_;
  option.parameter_tolerance = solve_parameter_tolerance_;
  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "Before LIO"));

  estimator->SetFixedIndex(3);

  // [0] prior factor
  if (true && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // [1] lidar factor
  SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

  for (const auto &v : point_corrs) {
    if (v.t_point < opt_min_t_ns)
      continue;
    if (v.t_point >= opt_max_t_ns)
      continue;
    if (v.t_point < tparam_.last_scan[1])
      continue;
    if (use_lidar_scale) {
      estimator->AddLoamMeasurementAnalyticNURBS(
          v, S_GtoM, p_GinM, S_LtoI, p_LinI,
          opt_weight_.lidar_weight * v.scale);
    } else {
      estimator->AddLoamMeasurementAnalyticNURBS(
          v, S_GtoM, p_GinM, S_LtoI, p_LinI, opt_weight_.lidar_weight);
    }
  }

  // [2] imu factor
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i), para_bg_vec[0],
                                              para_ba_vec[0], gravity_.data(),
                                              opt_weight_.imu_info_vec);
  }

  // [3] bias factor
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> noise_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  noise_covariance.block<3, 3>(0, 0) =
      (opt_weight_.imu_noise.sigma_wb_discrete *
       opt_weight_.imu_noise.sigma_wb_discrete) *
      Eigen::Matrix3d::Identity();
  noise_covariance.block<3, 3>(3, 3) =
      (opt_weight_.imu_noise.sigma_ab_discrete *
       opt_weight_.imu_noise.sigma_ab_discrete) *
      Eigen::Matrix3d::Identity();

  for (int i = tparam_.lio_imu_idx[0] + 1; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i - 1).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    double dt = (imu_data_[i].timestamp - imu_data_[i - 1].timestamp) * NS_TO_S;
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
    G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt;
    G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt;
    covariance =
        F * covariance * F.transpose() + G * noise_covariance * G.transpose();
  }

  Eigen::Matrix<double, 6, 6> sqrt_info_mat =
      Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse())
          .matrixL()
          .transpose();
  sqrt_info_ << sqrt_info_mat(0, 0), sqrt_info_mat(1, 1), sqrt_info_mat(2, 2),
      sqrt_info_mat(3, 3), sqrt_info_mat(4, 4), sqrt_info_mat(5, 5);
  // Only add bias factor if we have different bias times (avoid duplicate
  // parameter blocks)
  if (tparam_.last_bias_time != tparam_.cur_bias_time) {
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                             para_ba_vec[1], 1, sqrt_info_);
  } else {
    // In realtime mode, same bias time is common at initialization
    // We can skip the bias factor or use a regularization approach
    LOG(INFO) << "Skipping bias factor: same bias time "
              << tparam_.last_bias_time << " (common in realtime mode)";

    // Alternative: Add a weak regularization factor to prevent bias drift
    // This helps maintain bias stability when we don't have temporal
    // constraints
    Eigen::Matrix<double, 6, 1> weak_info =
        sqrt_info_ * 0.1; // 10x weaker constraint
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[0], para_ba_vec[0],
                             para_ba_vec[0], 0.01, weak_info); // Very small dt
  }

  // [5] uwb factor
  LOG(INFO) << "===== uwb_data_ size: " << uwb_data_.size();
  if (!uwb_data_.empty()) {
    LOG(INFO) << " building UWB Factor... ";
    SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
    Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;

    // Pass NLOS detector to estimator (LiDAR-IMU-UWB fusion)
    if (nlos_detector_) {
      estimator->SetNLOSDetector(nlos_detector_);
      LOG(INFO) << "===== NLOS Detector enabled for LiDAR-IMU-UWB optimization";
    }

    // Set up trajectory estimator options for UWB anchor optimization
    estimator->options.optimize_uwb_anchors = opt_weight_.optimize_uwb_anchors;
    estimator->options.lock_uwb_anchors = !opt_weight_.optimize_uwb_anchors;
    estimator->options.optimize_one_uwb_anchor =
        opt_weight_.optimize_one_uwb_anchor;
    estimator->options.target_uwb_anchor_id = opt_weight_.target_uwb_anchor_id;

    // If anchor optimization is enabled, first collect visible anchors
    std::set<int> visible_anchor_ids;
    if (opt_weight_.optimize_uwb_anchors ||
        opt_weight_.optimize_one_uwb_anchor) {
      // Collect all anchors that have measurements in current optimization
      // window
      for (const auto &uwb_meas : uwb_data_) {
        if (uwb_meas.timestamp < opt_min_t_ns)
          continue;
        if (uwb_meas.timestamp >= opt_max_t_ns)
          continue;
        for (const auto &dist_pair : uwb_meas.anchor_distances) {
          visible_anchor_ids.insert(dist_pair.first);
        }
      }

      LOG(INFO) << "Visible anchors in current optimization window: "
                << visible_anchor_ids.size();

      double timestamp = uwb_data_.back().timestamp * NS_TO_S;
      SaveUWBAnchorPositions(visible_anchor_ids, timestamp);

      int target_id = -1;
      if (opt_weight_.optimize_one_uwb_anchor) {
        target_id = opt_weight_.target_uwb_anchor_id;
      }

      // Only add parameter blocks for visible anchors
      for (int anchor_id : visible_anchor_ids) {
        // Only add the target anchor if single anchor optimization is enabled
        if (opt_weight_.optimize_one_uwb_anchor && anchor_id != target_id) {
          continue;
        }

        auto anchor_it = uwb_anchor_positions_.find(anchor_id);
        if (anchor_it == uwb_anchor_positions_.end()) {
          LOG(WARNING) << "Anchor " << anchor_id
                       << " not found in anchor_positions map";
          continue;
        }

        Eigen::Vector3d &anchor_pos = anchor_it->second;

        // Add anchor with prior factor if weight is positive
        auto prior_it = prior_anchor_positions_.find(anchor_id);
        if (prior_it != prior_anchor_positions_.end() &&
            opt_weight_.anchor_prior_weight > 0) {
          const Eigen::Vector3d &prior_pos = prior_it->second;
          estimator->AddUWBAnchorPositionParameter(
              anchor_id, anchor_pos.data(), prior_pos,
              opt_weight_.anchor_prior_weight);
          LOG(INFO) << "Added UWB anchor " << anchor_id
                    << " for optimization with prior factor (weight="
                    << opt_weight_.anchor_prior_weight
                    << ") at position: " << anchor_pos.transpose()
                    << ", prior: " << prior_pos.transpose();
        } else {
          // Fallback to version without prior if no initial position or zero
          // weight
          estimator->AddUWBAnchorPositionParameter(anchor_id,
                                                   anchor_pos.data());
          LOG(INFO) << "Added UWB anchor " << anchor_id
                    << " for optimization at position: "
                    << anchor_pos.transpose();
        }
      }
    }

    for (const auto &uwb_meas : uwb_data_) {
      if (uwb_meas.timestamp < opt_min_t_ns) {
        // LOG(INFO) << "===== uwb_meas.timestamp < opt_min_t_ns: "
        //           << "uwb_meas.timestamp: " << uwb_meas.timestamp
        //           << " opt_min_t_ns: " << opt_min_t_ns;
        continue;
      }
      if (uwb_meas.timestamp >= opt_max_t_ns) {
        continue;
      }
      // TODO : opt_weight_.uwb_weight = AutoAdjustWeight();
      if (opt_weight_.optimize_one_uwb_anchor) {
        estimator->AddUWBMeasurementWithOneAnchorOptNURBS(
            uwb_meas, opt_weight_.target_uwb_anchor_id, S_GtoM, p_GinM, S_UtoI,
            p_UinI, opt_weight_.uwb_weight);
      } else {
        estimator->AddUWBMeasurementAnalyticNURBS(
            uwb_meas, S_GtoM, p_GinM, S_UtoI, p_UinI, opt_weight_.uwb_weight);
      }

      // estimator->AddUWBMeasurementAutoDiffNURBS(
      // uwb_meas, S_GtoM, p_GinM, S_UtoI, p_UinI, opt_weight_.uwb_weight);
    }
  }

  TicToc t_opt;
  static int loam_cnt = 0;
  ceres::Solver::Summary summary =
      estimator->Solve(iteration, false, solve_num_threads_);
  // ceres::Solver::Summary summary = estimator->Solve(iteration, true);  //
  // Debug

  double opt_time = t_opt.toc();
  LOG(INFO) << "[t_opt] " << opt_time << std::endl;
  LOG(INFO) << "LICUSolver " << summary.BriefReport();
  LOG(INFO) << ++loam_cnt << " UpdateLio Successful/Unsuccessful steps: "
            << summary.num_successful_steps << "/"
            << summary.num_unsuccessful_steps;
  // LOG(INFO) << summary.FullReport();

  opt_cnt++;
  t_opt_sum += opt_time;

  LOG(INFO) << "[gyro_bias_new] "
            << all_imu_bias_.rbegin()->second.gyro_bias.x() << " "
            << all_imu_bias_.rbegin()->second.gyro_bias.y() << " "
            << all_imu_bias_.rbegin()->second.gyro_bias.z();
  LOG(INFO) << "[acce_bias_new] "
            << all_imu_bias_.rbegin()->second.accel_bias.x() << " "
            << all_imu_bias_.rbegin()->second.accel_bias.y() << " "
            << all_imu_bias_.rbegin()->second.accel_bias.z();

  // Publish UWB anchor visualization after optimization (LIU mode)
  if (opt_weight_.optimize_uwb_anchors && estimator) {
    std::set<int> visible_anchors;
    // Collect visible anchors from the estimator
    for (const auto &anchor_pair : uwb_anchor_positions_) {
      if (estimator->IsAnchorVisible(anchor_pair.first)) {
        visible_anchors.insert(anchor_pair.first);
      }
    }
    // Publish visualization
    PublishUWBAnchorVisualization(visible_anchors);
    // double timestamp = uwb_data_.back().timestamp * NS_TO_S;
    // SaveUWBAnchorPositions(visible_anchors, timestamp);
    LOG(INFO) << "UWB anchor visualization: " << visible_anchors.size()
              << " visible anchors";
  }

  return true;
}

void TrajectoryManager::SaveUWBAnchorPositions(
    const std::set<int> &visible_anchors, double timestamp) {
  std::ofstream anchor_out(
      "/home/dev/ws_LIU/src/LIUO-UWB/data/uwb_anchors.txt", std::ios::app);
  if (!anchor_out.is_open()) {
    return;
  }
  const bool filter = !visible_anchors.empty();
  for (const auto &pair_id_pos : uwb_anchor_positions_) {
    // for (const auto &pair_id_pos : initial_anchor_positions_) {
    const int anchor_id = pair_id_pos.first;
    if (filter && visible_anchors.find(anchor_id) == visible_anchors.end()) {
      continue;
    }
    const Eigen::Vector3d &p = pair_id_pos.second;
    anchor_out << std::setprecision(12) << timestamp << " " << anchor_id << " "
               << p.x() << " " << p.y() << " " << p.z() << std::endl;
  }
}

void TrajectoryManager::UpdateLiDARAttribute(double scan_time_min,
                                             double scan_time_max) {
  if (trajectory_->maxTimeNsNURBS() > 25 * S_TO_NS) {
    int64_t t = trajectory_->maxTimeNsNURBS() - 15 * S_TO_NS;
    RemoveIMUData(t);
    RemoveUWBData(t);
    RemovePoseData(t);
  }
}

void TrajectoryManager::UpdateLICPrior(
    const Eigen::aligned_vector<PointCorrespondence> &point_corrs) {
  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;

  // Configure Integrity Monitoring through options


  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option)); // AddControlPoint

  // construct a new prior
  MarginalizationInfo *marginalization_info = new MarginalizationInfo();

  // prepare the control points and biases to be marginalized
  int lhs_idx = trajectory_->numKnots() - 1 - division_ -
                2; // retain the last 3 control points in this optimization;
                   // remember, cubic spline is adopted
  int rhs_idx = trajectory_->numKnots() - 4;

  auto &last_bias =
      all_imu_bias_[tparam_.last_bias_time]; // marginalize the bias bi
  auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
  std::vector<double *> drop_param;
  for (int i = lhs_idx; i <= rhs_idx; i++) {
    drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
    drop_param.emplace_back(trajectory_->getKnotPos(i).data());
  }
  drop_param.emplace_back(last_bias.gyro_bias.data());
  drop_param.emplace_back(last_bias.accel_bias.data());

  // [0] prior factor marginalization
  if (lidar_marg_info) {
    std::vector<int> drop_set;
    for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++) {
      for (auto const &dp : drop_param) {
        if (lidar_marg_parameter_blocks[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      MarginalizationFactor *cost_function =
          new MarginalizationFactor(lidar_marg_info);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                lidar_marg_parameter_blocks, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [1] imu factor marginalization
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    int64_t time_ns = imu_data_.at(i).timestamp;
    std::pair<int, double> su; // i u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      ceres::CostFunction *cost_function =
          new analytic_derivative::IMUFactorNURBS(
              time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec,
              trajectory_->knts, su, blending_matrix,
              cumulative_blending_matrix);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_IMU, cost_function, NULL, vec, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [2] lidar factor marginalization
  SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();
  for (const auto &v : point_corrs) {
    if (v.t_point < opt_min_t_ns)
      continue;
    if (v.t_point >= opt_max_t_ns)
      continue;
    if (v.t_point < tparam_.last_scan[1])
      continue;
    int64_t time_ns = v.t_point;
    std::pair<int, double> su; // i and u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      double weight = opt_weight_.lidar_weight;
      if (use_lidar_scale) {
        weight *= v.scale;
      }
      ceres::CostFunction *cost_function =
          new analytic_derivative::LoamFeatureFactorNURBS(
              time_ns, v, su, blending_matrix, cumulative_blending_matrix,
              S_GtoM, p_GinM, S_LtoI, p_LinI, weight);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          RType_LiDAR, cost_function, NULL, vec, drop_set);
      int num_residuals = cost_function->num_residuals();
      Eigen::MatrixXd residuals;
      residuals.setZero(num_residuals, 1);
      cost_function->Evaluate(vec.data(), residuals.data(), nullptr);
      double dist = (residuals / weight).norm();
      if (dist < 0.05)
      // if (dist < 0.01)
      {
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }
  }

  // [3] bias factor marginalization
  std::vector<double *> vec;
  vec.emplace_back(last_bias.gyro_bias.data());
  vec.emplace_back(cur_bias.gyro_bias.data());
  vec.emplace_back(last_bias.accel_bias.data());
  vec.emplace_back(cur_bias.accel_bias.data());

  std::vector<int> drop_set;
  drop_set.emplace_back(0); // bgi
  drop_set.emplace_back(2); // bai

  analytic_derivative::BiasFactor *cost_function =
      new analytic_derivative::BiasFactor(1, sqrt_info_);
  ResidualBlockInfo *residual_block_info =
      new ResidualBlockInfo(RType_Bias, cost_function, NULL, vec, drop_set);
  marginalization_info->addResidualBlockInfo(residual_block_info);

  marginalization_info->preMarginalize();
  marginalization_info->marginalize();
  if (lidar_marg_info) {
    lidar_marg_info = nullptr;
  }
  lidar_marg_info.reset(marginalization_info);
  lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
}

void TrajectoryManager::UpdateLICUPrior(
    const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
    const std::deque<UwbData> &uwb_measurements) {
  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;

  // Configure Integrity Monitoring through options


  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option)); // AddControlPoint

  // construct a new prior
  MarginalizationInfo *marginalization_info = new MarginalizationInfo();

  // prepare the control points and biases to be marginalized
  int lhs_idx = trajectory_->numKnots() - 1 - division_ -
                2; // retain the last 3 control points in this optimization;
                   // remember, cubic spline is adopted
  int rhs_idx = trajectory_->numKnots() - 4;

  auto &last_bias =
      all_imu_bias_[tparam_.last_bias_time]; // marginalize the bias bi
  auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
  std::vector<double *> drop_param;
  for (int i = lhs_idx; i <= rhs_idx; i++) {
    drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
    drop_param.emplace_back(trajectory_->getKnotPos(i).data());
  }
  drop_param.emplace_back(last_bias.gyro_bias.data());
  drop_param.emplace_back(last_bias.accel_bias.data());

  // [0] prior factor marginalization
  if (lidar_marg_info) {
    std::vector<int> drop_set;
    for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++) {
      for (auto const &dp : drop_param) {
        if (lidar_marg_parameter_blocks[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      MarginalizationFactor *cost_function =
          new MarginalizationFactor(lidar_marg_info);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                lidar_marg_parameter_blocks, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [1] imu factor marginalization
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    int64_t time_ns = imu_data_.at(i).timestamp;
    std::pair<int, double> su; // i u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      ceres::CostFunction *cost_function =
          new analytic_derivative::IMUFactorNURBS(
              time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec,
              trajectory_->knts, su, blending_matrix,
              cumulative_blending_matrix);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_IMU, cost_function, NULL, vec, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [2] lidar factor marginalization
  SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();
  for (const auto &v : point_corrs) {
    if (v.t_point < opt_min_t_ns)
      continue;
    if (v.t_point >= opt_max_t_ns)
      continue;
    if (v.t_point < tparam_.last_scan[1])
      continue;
    int64_t time_ns = v.t_point;
    std::pair<int, double> su; // i and u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      double weight = opt_weight_.lidar_weight;
      if (use_lidar_scale) {
        weight *= v.scale;
      }
      ceres::CostFunction *cost_function =
          new analytic_derivative::LoamFeatureFactorNURBS(
              time_ns, v, su, blending_matrix, cumulative_blending_matrix,
              S_GtoM, p_GinM, S_LtoI, p_LinI, weight);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          RType_LiDAR, cost_function, NULL, vec, drop_set);
      int num_residuals = cost_function->num_residuals();
      Eigen::MatrixXd residuals;
      residuals.setZero(num_residuals, 1);
      cost_function->Evaluate(vec.data(), residuals.data(), nullptr);
      double dist = (residuals / weight).norm();
      if (dist < 0.05)
      // if (dist < 0.01)
      {
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }
  }

  // [3] bias factor marginalization
  std::vector<double *> vec;
  vec.emplace_back(last_bias.gyro_bias.data());
  vec.emplace_back(cur_bias.gyro_bias.data());
  vec.emplace_back(last_bias.accel_bias.data());
  vec.emplace_back(cur_bias.accel_bias.data());

  std::vector<int> drop_set;
  drop_set.emplace_back(0); // bgi
  drop_set.emplace_back(2); // bai

  analytic_derivative::BiasFactor *cost_function =
      new analytic_derivative::BiasFactor(1, sqrt_info_);
  ResidualBlockInfo *residual_block_info =
      new ResidualBlockInfo(RType_Bias, cost_function, NULL, vec, drop_set);
  marginalization_info->addResidualBlockInfo(residual_block_info);

  // [5] uwb factor marginalization
  SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
  Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;

  for (const auto &uwb_meas : uwb_measurements) {
    if (uwb_meas.timestamp < opt_min_t_ns)
      continue;
    if (uwb_meas.timestamp >= opt_max_t_ns)
      continue;
    std::pair<int, double> su; // i and u
    trajectory_->GetIdxT(uwb_meas.timestamp, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];

    for (const auto &dist_pair : uwb_meas.anchor_distances) {
      auto anchor_it = uwb_meas.anchor_positions.find(dist_pair.first);
      using Functor = analytic_derivative::UWBFactorNURBS;
      ceres::CostFunction *cost_function =
          new Functor(uwb_meas.timestamp, dist_pair, *anchor_it, su,
                      blending_matrix, cumulative_blending_matrix, S_GtoM,
                      p_GinM, S_UtoI, p_UinI, opt_weight_.uwb_weight);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_UWB, cost_function, NULL, vec, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  marginalization_info->preMarginalize();
  marginalization_info->marginalize();
  if (lidar_marg_info) {
    lidar_marg_info = nullptr;
  }
  lidar_marg_info.reset(marginalization_info);
  lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
}

void TrajectoryManager::UpdateTrajectoryWithUIO(
    int iter, int64_t uwb_timestamp,
    const std::deque<UwbData> &uwb_measurements) {
  IMUBias last_bias = all_imu_bias_.rbegin()->second;
  all_imu_bias_[tparam_.cur_bias_time] = last_bias;
  std::map<int, double *> para_bg_vec;
  std::map<int, double *> para_ba_vec;
  {
    auto &bias0 = all_imu_bias_[tparam_.last_bias_time]; // bi
    para_bg_vec[0] = bias0.gyro_bias.data();
    para_ba_vec[0] = bias0.accel_bias.data();

    auto &bias1 = all_imu_bias_[tparam_.cur_bias_time]; // bj
    para_bg_vec[1] = bias1.gyro_bias.data();
    para_ba_vec[1] = bias1.accel_bias.data();
  }

  // Define time bounds
  int64_t t_min = opt_min_t_ns;
  int64_t t_max = opt_max_t_ns;

  TrajectoryEstimatorOptions option;
  option.lock_ab = false;
  option.lock_wb = false;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option, "UWB-IMU Optimization"));

  estimator->SetFixedIndex(3);

  // [0] prior factor
  if (true && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // Add UWB factors
  SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
  Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;

  for (const auto &uwb_data : uwb_measurements) {
    if (uwb_data.timestamp < t_min || uwb_data.timestamp >= t_max)
      continue;

    // Add UWB measurement using trajectory estimator
    estimator->AddUWBMeasurementAnalyticNURBS(uwb_data, SO3d(),
                                              Eigen::Vector3d::Zero(), S_UtoI,
                                              p_UinI, opt_weight_.uwb_weight);
  }

  // Solve the optimization problem using estimator
  TicToc t_solver;
  ceres::Solver::Summary summary =
      estimator->Solve(5, false, solve_num_threads_);

  if (verbose) {
    std::cout << summary.BriefReport() << std::endl;
  }
}

void TrajectoryManager::UpdateTrajectoryWithUICO(
    int iter, int64_t uwb_timestamp, int64_t image_timestamp,
    const std::deque<UwbData> &uwb_measurements,
    const Eigen::aligned_vector<Eigen::Vector3d> &v_points,
    const Eigen::aligned_vector<Eigen::Vector2d> &px_obss) {
  // Define time bounds
  int64_t t_min = opt_min_t_ns;
  int64_t t_max = opt_max_t_ns;

  TrajectoryEstimatorOptions option;
  option.lock_ab = false;
  option.lock_wb = false;
  option.lock_g = true;
  option.show_residual_summary = verbose;
  TrajectoryEstimator::Ptr estimator(new TrajectoryEstimator(
      trajectory_, option, "UWB-IMU-Camera Optimization"));

  estimator->SetFixedIndex(3);

  // [0] prior factor
  if (use_marg_ && lidar_marg_info) {
    estimator->AddMarginalizationFactor(lidar_marg_info,
                                        lidar_marg_parameter_blocks);
  }

  // Add IMU preintegration factors
  for (auto iter = imu_data_.begin(); iter != imu_data_.end(); ++iter) {
    if (iter->timestamp >= t_min && iter->timestamp < t_max) {
      IMUBias last_bias = all_imu_bias_[tparam_.last_bias_time];
      IMUBias cur_bias = all_imu_bias_[tparam_.cur_bias_time];

      estimator->AddIMUMeasurementAnalyticNURBS(
          *iter, last_bias.gyro_bias.data(), last_bias.accel_bias.data(),
          gravity_.data(), sqrt_info_, false);
    }
  }

  // Add UWB factors
  SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
  Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;

  for (const auto &uwb_data : uwb_measurements) {
    if (uwb_data.timestamp < t_min || uwb_data.timestamp >= t_max)
      continue;

    // Add UWB measurement using trajectory estimator
    estimator->AddUWBMeasurementAnalyticNURBS(uwb_data, SO3d(),
                                              Eigen::Vector3d::Zero(), S_UtoI,
                                              p_UinI, opt_weight_.uwb_weight);
  }

  // Add visual factors (skip for now as ReprojectionFactorNURBS is not
  // available)
  // TODO: Implement visual factor addition if needed

  // Solve the optimization problem using estimator
  TicToc t_solver;
  ceres::Solver::Summary summary =
      estimator->Solve(5, false, solve_num_threads_);

  if (verbose) {
    std::cout << summary.BriefReport() << std::endl;
  }
}

void TrajectoryManager::UpdateUIOPrior() {
  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;

  // Configure Integrity Monitoring through options


  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option)); // AddControlPoint

  // construct a new prior
  MarginalizationInfo *marginalization_info = new MarginalizationInfo();

  // prepare the control points and biases to be marginalized
  int lhs_idx = trajectory_->numKnots() - 1 - division_ -
                2; // retain the last 3 control points in this optimization;
                   // remember, cubic spline is adopted
  int rhs_idx = trajectory_->numKnots() - 4;

  auto &last_bias =
      all_imu_bias_[tparam_.last_bias_time]; // marginalize the bias bi
  auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
  std::vector<double *> drop_param;
  for (int i = lhs_idx; i <= rhs_idx; i++) {
    drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
    drop_param.emplace_back(trajectory_->getKnotPos(i).data());
  }
  drop_param.emplace_back(last_bias.gyro_bias.data());
  drop_param.emplace_back(last_bias.accel_bias.data());

  // [0] prior factor marginalization
  if (lidar_marg_info) {
    std::vector<int> drop_set;
    for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++) {
      for (auto const &dp : drop_param) {
        if (lidar_marg_parameter_blocks[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      MarginalizationFactor *cost_function =
          new MarginalizationFactor(lidar_marg_info);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                lidar_marg_parameter_blocks, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [1] imu factor marginalization
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    int64_t time_ns = imu_data_.at(i).timestamp;
    std::pair<int, double> su; // i u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      ceres::CostFunction *cost_function =
          new analytic_derivative::IMUFactorNURBS(
              time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec,
              trajectory_->knts, su, blending_matrix,
              cumulative_blending_matrix);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_IMU, cost_function, NULL, vec, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [2] bias factor marginalization
  std::vector<double *> vec;
  vec.emplace_back(last_bias.gyro_bias.data());
  vec.emplace_back(cur_bias.gyro_bias.data());
  vec.emplace_back(last_bias.accel_bias.data());
  vec.emplace_back(cur_bias.accel_bias.data());

  std::vector<int> drop_set;
  drop_set.emplace_back(0); // bgi
  drop_set.emplace_back(2); // bai

  analytic_derivative::BiasFactor *cost_function =
      new analytic_derivative::BiasFactor(1, sqrt_info_);
  ResidualBlockInfo *residual_block_info =
      new ResidualBlockInfo(RType_Bias, cost_function, NULL, vec, drop_set);
  marginalization_info->addResidualBlockInfo(residual_block_info);

  // [3] uwb factor marginalization
  SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
  Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

  for (const auto &uwb_meas : uwb_data_) {
    if (uwb_meas.timestamp < opt_min_t_ns)
      continue;
    if (uwb_meas.timestamp >= opt_max_t_ns)
      continue;
    std::pair<int, double> su; // i and u
    trajectory_->GetIdxT(uwb_meas.timestamp, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      for (const auto &dist_pair : uwb_meas.anchor_distances) {
        auto anchor_it = uwb_meas.anchor_positions.find(dist_pair.first);
        using Functor = analytic_derivative::UWBFactorNURBS;
        ceres::CostFunction *cost_function =
            new Functor(uwb_meas.timestamp, dist_pair, *anchor_it, su,
                        blending_matrix, cumulative_blending_matrix, S_GtoM,
                        p_GinM, S_UtoI, p_UinI, opt_weight_.uwb_weight);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            RType_UWB, cost_function, NULL, vec, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }
  }

  marginalization_info->preMarginalize();
  marginalization_info->marginalize();
  if (lidar_marg_info) {
    lidar_marg_info = nullptr;
  }
  lidar_marg_info.reset(marginalization_info);
  lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
}

void TrajectoryManager::UpdateUICOPrior() {
  TrajectoryEstimatorOptions option;
  option.is_marg_state = true;

  // Configure Integrity Monitoring through options


  TrajectoryEstimator::Ptr estimator(
      new TrajectoryEstimator(trajectory_, option)); // AddControlPoint

  // construct a new prior
  MarginalizationInfo *marginalization_info = new MarginalizationInfo();

  // prepare the control points and biases to be marginalized
  int lhs_idx = trajectory_->numKnots() - 1 - division_ -
                2; // retain the last 3 control points in this optimization;
                   // remember, cubic spline is adopted
  int rhs_idx = trajectory_->numKnots() - 4;

  auto &last_bias =
      all_imu_bias_[tparam_.last_bias_time]; // marginalize the bias bi
  auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
  std::vector<double *> drop_param;
  for (int i = lhs_idx; i <= rhs_idx; i++) {
    drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
    drop_param.emplace_back(trajectory_->getKnotPos(i).data());
  }
  drop_param.emplace_back(last_bias.gyro_bias.data());
  drop_param.emplace_back(last_bias.accel_bias.data());

  // [0] prior factor marginalization
  if (lidar_marg_info) {
    std::vector<int> drop_set;
    for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++) {
      for (auto const &dp : drop_param) {
        if (lidar_marg_parameter_blocks[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      MarginalizationFactor *cost_function =
          new MarginalizationFactor(lidar_marg_info);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                lidar_marg_parameter_blocks, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [1] imu factor marginalization
  for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i) {
    if (imu_data_.at(i).timestamp < opt_min_t_ns)
      continue;
    if (imu_data_.at(i).timestamp >= opt_max_t_ns)
      continue;
    int64_t time_ns = imu_data_.at(i).timestamp;
    std::pair<int, double> su; // i u
    trajectory_->GetIdxT(time_ns, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      ceres::CostFunction *cost_function =
          new analytic_derivative::IMUFactorNURBS(
              time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec,
              trajectory_->knts, su, blending_matrix,
              cumulative_blending_matrix);
      ResidualBlockInfo *residual_block_info =
          new ResidualBlockInfo(RType_IMU, cost_function, NULL, vec, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }
  }

  // [2] bias factor marginalization
  std::vector<double *> vec;
  vec.emplace_back(last_bias.gyro_bias.data());
  vec.emplace_back(cur_bias.gyro_bias.data());
  vec.emplace_back(last_bias.accel_bias.data());
  vec.emplace_back(cur_bias.accel_bias.data());

  std::vector<int> drop_set;
  drop_set.emplace_back(0); // bgi
  drop_set.emplace_back(2); // bai

  analytic_derivative::BiasFactor *cost_function =
      new analytic_derivative::BiasFactor(1, sqrt_info_);
  ResidualBlockInfo *residual_block_info =
      new ResidualBlockInfo(RType_Bias, cost_function, NULL, vec, drop_set);
  marginalization_info->addResidualBlockInfo(residual_block_info);

  // [3] uwb factor marginalization
  SO3d S_UtoI = trajectory_->GetSensorEP(UWBSensor).so3;
  Eigen::Vector3d p_UinI = trajectory_->GetSensorEP(UWBSensor).p;
  SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
  Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

  for (const auto &uwb_meas : uwb_data_) {
    if (uwb_meas.timestamp < opt_min_t_ns)
      continue;
    if (uwb_meas.timestamp >= opt_max_t_ns)
      continue;
    std::pair<int, double> su; // i and u
    trajectory_->GetIdxT(uwb_meas.timestamp, su);
    Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
    Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[su.first - 3];
    std::vector<double *> vec;
    estimator->AddControlPointsNURBS(su.first - 3, vec);
    estimator->AddControlPointsNURBS(su.first - 3, vec, true);

    std::vector<int> drop_set;
    for (int i = 0; i < vec.size(); i++) {
      for (auto const &dp : drop_param) {
        if (vec[i] == dp) {
          drop_set.emplace_back(i);
          break;
        }
      }
    }

    if (!drop_set.empty()) {
      for (const auto &dist_pair : uwb_meas.anchor_distances) {
        auto anchor_it = uwb_meas.anchor_positions.find(dist_pair.first);
        using Functor = analytic_derivative::UWBFactorNURBS;
        ceres::CostFunction *cost_function =
            new Functor(uwb_meas.timestamp, dist_pair, *anchor_it, su,
                        blending_matrix, cumulative_blending_matrix, S_GtoM,
                        p_GinM, S_UtoI, p_UinI, opt_weight_.uwb_weight);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            RType_UWB, cost_function, NULL, vec, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }
  }

  marginalization_info->preMarginalize();
  marginalization_info->marginalize();
  if (lidar_marg_info) {
    lidar_marg_info = nullptr;
  }
  lidar_marg_info.reset(marginalization_info);
  lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
}

void TrajectoryManager::UpdateUWBAttribute(int64_t uwb_timestamp) {
  // Update time window
  tparam_.cur_bias_time = uwb_timestamp;

  // Remove old IMU data
  RemoveIMUData(uwb_timestamp - 1.0 * S_TO_NS);

  // Remove old UWB data
  RemoveUWBData(uwb_timestamp - 1.0 * S_TO_NS);

  // Update IMU bias
  // UpdateIMUBias();
}

// UWB anchor position management methods implementation
void TrajectoryManager::InitializeUWBAnchorPositions(
    const std::map<int, Eigen::Vector3d> &initial_positions) {
  initial_anchor_positions_ = initial_positions;
  uwb_anchor_positions_ = initial_positions;
  LOG(INFO) << "Initialized " << initial_positions.size()
            << " UWB anchor positions";
}

void TrajectoryManager::SetUWBAnchorPosition(int anchor_id,
                                             const Eigen::Vector3d &position) {
  uwb_anchor_positions_[anchor_id] = position;
}

Eigen::Vector3d TrajectoryManager::GetUWBAnchorPosition(int anchor_id) const {
  auto it = uwb_anchor_positions_.find(anchor_id);
  if (it != uwb_anchor_positions_.end()) {
    return it->second;
  }
  // Return zero vector if anchor not found
  return Eigen::Vector3d::Zero();
}

Eigen::Vector3d TrajectoryManager::AddGaussianNoiseToAnchorPosition(
    const Eigen::Vector3d &position, double noise_std_x, double noise_std_y,
    double noise_std_z) {
  // Static random number generator (seeded once for consistency)
  static std::random_device rd;
  static std::mt19937 gen(rd());

  // Generate Gaussian noise for each axis
  std::normal_distribution<double> dist_x(0.0, noise_std_x);
  std::normal_distribution<double> dist_y(0.0, noise_std_y);
  std::normal_distribution<double> dist_z(0.0, noise_std_z);

  Eigen::Vector3d noisy_position = position;
  noisy_position.x() += dist_x(gen);
  noisy_position.y() += dist_y(gen);
  noisy_position.z() += dist_z(gen);

  return noisy_position;
}

void TrajectoryManager::UpdateUWBAnchorPositions() {
  // This method can be called after optimization to update anchor positions
  // The actual positions are updated through the optimization process
  LOG(INFO) << "UWB anchor positions updated after optimization";
}

bool TrajectoryManager::IsUWBAnchorOptimizationEnabled() const {
  // Check if UWB anchor optimization is enabled in trajectory options
  // This would depend on the specific implementation of trajectory options
  return opt_weight_
      .optimize_uwb_anchors; // Assuming this field exists in OptWeight
}

void TrajectoryManager::SetOdometryViewer(OdometryViewer *viewer) {
  odometry_viewer_ = viewer;
}

void TrajectoryManager::PublishUWBAnchorVisualization(
    const std::set<int> &visible_anchors) {
  if (!odometry_viewer_)
    return;

  // Publish anchor positions with visualization
  odometry_viewer_->PublishUWBAnchorPositions(
      uwb_anchor_positions_, initial_anchor_positions_, visible_anchors);

  // Print statistics if there are anchors
  if (!uwb_anchor_positions_.empty()) {
    odometry_viewer_->PublishUWBAnchorStats(
        uwb_anchor_positions_, initial_anchor_positions_, visible_anchors);
  }
}

// ConfigureIntegrityMonitoring removed


void TrajectoryManager::UpdateNLOSDetectorPointCloudMap(
    const PosCloud::Ptr &cloud_map) {
  if (!nlos_detector_) {
    LOG(WARNING) << "[TrajectoryManager] NLOS detector not initialized!";
    return;
  }

  if (!cloud_map || cloud_map->empty()) {
    LOG(WARNING) << "[TrajectoryManager] Point cloud map is empty!";
    return;
  }

  nlos_detector_->UpdatePointCloudMap(cloud_map);
  LOG(INFO) << "[TrajectoryManager] Updated NLOS detector with "
            << cloud_map->size() << " points for LiDAR-IMU-UWB fusion.";
}

PosCloud::Ptr TrajectoryManager::BuildPointCloudMapForNLOS() {
  PosCloud::Ptr map(new PosCloud);

  if (!lidar_handler_) {
    LOG(WARNING) << "[BuildPointCloudMapForNLOS] LidarHandler not set!";
    return map;
  }

  // Cast void pointer back to LidarHandler
  // Note: This requires including lidar/lidar_handler.h
  auto *lidar_handler = static_cast<liuo::LidarHandler *>(lidar_handler_);

  // Get local feature container (keyframes)
  const auto &feature_container =
      lidar_handler->GetLocalFeatureContainerDownsampled();

  if (feature_container.empty()) {
    LOG(WARNING) << "[BuildPointCloudMapForNLOS] Feature container is empty!";
    return map;
  }

  // Extract points from recent keyframes (last 10 frames)
  size_t count = 0;
  size_t max_keyframes = 10;

  for (auto it = feature_container.rbegin();
       it != feature_container.rend() && count < max_keyframes; ++it, ++count) {
    int64_t timestamp = it->first;
    const LiDARFeature &feature = it->second;

    // Use surface features (most representative of environment)
    if (feature.surface_features && !feature.surface_features->empty()) {
      // Transform to global coordinate system
      SE3d pose = trajectory_->GetLidarPoseNURBS(timestamp);
      PosCloud::Ptr transformed_cloud(new PosCloud);
      pcl::transformPointCloud(*feature.surface_features, *transformed_cloud,
                               pose.matrix());
      *map += *transformed_cloud;
    }
  }

  // Downsample to reduce computational cost
  if (!map->empty()) {
    VoxelFilter<PosPoint> voxel_filter;
    voxel_filter.SetResolution(0.2); // Match nlos_voxel_resolution
    voxel_filter.SetInputCloud(map);
    PosCloud::Ptr downsampled_map(new PosCloud);
    voxel_filter.Filter(downsampled_map);

    LOG(INFO) << "[BuildPointCloudMapForNLOS] Built map with "
              << downsampled_map->size() << " points from " << count
              << " keyframes (original: " << map->size() << " points)";

    return downsampled_map;
  }

  return map;
}

} // namespace liuo
