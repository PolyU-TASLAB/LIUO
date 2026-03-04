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

#pragma once

#include <imu/imu_initializer.h>
#include <imu/imu_state_estimator.h>
#include <lidar/lidar_handler.h>
#include <odom/msg_manager.h>
#include <odom/odometry_viewer.hpp>
#include <odom/trajectory_manager.h>
#include <ros/ros.h>
#include <uwb/uwb_handler.h>

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace liuo {

enum KnotDensity {
  gear1 = 1, // 0.1
  gear2 = 2, // 0.05
  gear3 = 3, // 0.033
  gear4 = 4  // 0.025
};
class OdometryManager {
public:
  struct RunBagTimingRecord {
    int64_t frame_idx = -1;
    std::string status = "ok";
    int64_t traj_max_time_ns_next = -1;
    int64_t lidar_t_min_ns = -1;
    int64_t lidar_t_max_ns = -1;
    int64_t uwb_t_ns = -1;
    int scan_num = -1;
    int cp_add_num_cur = -1;
    int has_uwb = 0;

    double bag_spin_ms = -1.0;
    double prepare_msgs_ms = -1.0;
    double update_one_seg_ms = -1.0;
    double solve_total_ms = -1.0;
    double process_liuo_ms = -1.0;
    double prior_ms = -1.0;
    double attr_ms = -1.0;
    double frame_total_ms = -1.0;
  };

  OdometryManager(const YAML::Node &node, ros::NodeHandle &nh,
                  const std::string &config_file_path);

  void RunBag();

  void RunInSubscribeMode();

  double SaveOdometry();

  void SavePointCloudMap(const std::string &descri);

  std::vector<int> cp_num_vec;

  int GetKnotDensity(double gyro_norm, double acce_norm) {
    if (gyro_norm < 0.0 || acce_norm < 0.0) {
      std::cout << RED << "gyro_norm/acce_norm is wrong!" << RESET << std::endl;
    }

    int gyro_density = -1, acce_density = -1;

    acce_norm = std::abs(acce_norm - gravity_norm_);
    LOG(INFO) << "[acce_norm] " << acce_norm;
    if (acce_norm < 0.5) { // [0, 0.5)
      acce_density = KnotDensity::gear1;
    } else if (acce_norm < 1.0) { // [0.5, 1.0)
      acce_density = KnotDensity::gear2;
    } else if (acce_norm < 5.0) { // [1.0, 5.0)
      acce_density = KnotDensity::gear3;
    } else { // [5.0, -)
      acce_density = KnotDensity::gear4;
    }

    LOG(INFO) << "[gyro_norm] " << gyro_norm;
    if (gyro_norm < 0.5) { // [0, 0.5)
      gyro_density = KnotDensity::gear1;
    } else if (gyro_norm < 1.0) { // [0.5, 1.0)
      gyro_density = KnotDensity::gear2;
    } else if (gyro_norm < 5.0) { // [1.0, 5.0)
      gyro_density = KnotDensity::gear3;
    } else { // [5.0, -)
      gyro_density = KnotDensity::gear4;
    }

    return std::max(gyro_density, acce_density);
  };

protected:
  bool CreateCacheFolder(const std::string &config_path,
                         const std::string &bag_path);

  void SolveLIO();

  void ProcessLICData();

  void SolveLIUO();

  void SolveLICUO();

  void ProcessLIUOData();

  void ProcessLICUOData();

  void ProcessImageData();

  bool PrepareTwoSegMsgs(int seg_idx);

  void UpdateTwoSeg();

  bool PrepareMsgs();

  void UpdateOneSeg();

  void SetInitialState();

  void PublishCloudAndTrajectory();

  void GetUWBPosInit();

  void InitRunBagTimingConfig(const YAML::Node &node);

  bool InitRunBagTimingFileIfNeeded();

  void AppendRunBagTimingRecord(const RunBagTimingRecord &record);

  std::string BuildSaveFilePath(const std::string &output_dir,
                                const std::string &descri,
                                const std::string &suffix,
                                const std::string &extension) const;

protected:
  OdometryMode odometry_mode_;

  MsgManager::Ptr msg_manager_;

  bool is_initialized_;
  IMUInitializer::Ptr imu_initializer_;

  Trajectory::Ptr trajectory_;
  TrajectoryManager::Ptr trajectory_manager_;

  LidarHandler::Ptr lidar_handler_;

  UWBHandler::Ptr uwb_handler_;

  int64_t t_begin_add_cam_; //

  OdometryViewer odom_viewer_;

  int update_every_k_knot_;

  /// [nurbs]
  double t_add_;
  int64_t t_add_ns_;
  bool non_uniform_;
  double distance0_;

  int cp_add_num_coarse_;
  int cp_add_num_refine_;

  int lidar_iter_;
  bool use_lidar_scale_;

  std::string cache_path_;
  bool save_trajectory_enable_;
  bool save_pcd_enable_;
  std::string trajectory_output_dir_;
  std::string pcd_output_dir_;
  std::string save_runtime_timestamp_;
  std::string save_file_prefix_;

  double pasue_time_;

  TimeStatistics time_summary_;

  struct SysTimeOffset {
    SysTimeOffset(double t1, double t2, double t3, double t4)
        : timestamp(t1), t_lidar(t2), t_cam(t3), t_imu(t4) {}
    double timestamp = 0;
    double t_lidar = 0;
    double t_cam = 0;
    double t_imu = 0;
    double t_uwb = 0;
  };
  std::vector<SysTimeOffset> sys_t_offset_vec_;

private:
  double gravity_norm_;

  int64_t traj_max_time_ns_cur;
  int64_t traj_max_time_ns_next;
  int64_t traj_max_time_ns_next_next;

  int cp_add_num_cur;
  int cp_add_num_next;
  int cp_add_num_next_next;

  bool is_evo_viral_;

  double ave_r_thresh_;
  double ave_a_thresh_;

  VPointCloud sub_map_cur_frame_point_;

  Eigen::aligned_vector<Eigen::Vector3d> v_points_;
  Eigen::aligned_vector<Eigen::Vector2d> px_obss_;

  Eigen::Matrix3d K_;

  bool timing_enable_;
  std::string timing_output_file_;
  std::string timing_file_path_;
  bool timing_header_written_;
  int64_t timing_frame_idx_;
  double last_process_liuo_ms_;
  double last_prior_ms_;
  double last_attr_ms_;
};

} // namespace liuo
