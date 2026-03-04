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

#include <eigen_conversions/eigen_msg.h>
#include <odom/odometry_manager.h>
#include <utils/cloud_tool.h>

#include <boost/filesystem.hpp>

#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <ctime>

std::fstream rgb_file;
std::fstream img_file;

namespace liuo {

OdometryManager::OdometryManager(const YAML::Node &node, ros::NodeHandle &nh,
                                 const std::string &config_file_path)
    : odometry_mode_(LIO), is_initialized_(false), traj_max_time_ns_cur(0),
      traj_max_time_ns_next(0), traj_max_time_ns_next_next(0),
  cp_add_num_cur(0), cp_add_num_next(0), cp_add_num_next_next(0),
  timing_enable_(false), timing_output_file_("runbag_timing.csv"),
  timing_header_written_(false), timing_frame_idx_(0),
  last_process_liuo_ms_(-1.0), last_prior_ms_(-1.0),
  last_attr_ms_(-1.0), save_trajectory_enable_(true),
  save_pcd_enable_(true), trajectory_output_dir_("logs"),
  pcd_output_dir_("logs") {
  
  // Resolve config path relative to the config file itself
  boost::filesystem::path config_file_p(config_file_path);
  std::string config_path = config_file_p.parent_path().string();
  
  // Check if config_path ends with /
  if (!config_path.empty() && config_path.back() != '/') {
      config_path += "/";
  }
  
  // Attempt to use YAML provided paths if they are valid absolute paths
  // Otherwise fallback to relative paths
  
  std::string lidar_yaml = node["lidar_yaml"].as<std::string>();
  std::string lidar_yaml_path = config_path + lidar_yaml;
  
  std::cout << "Loading Lidar config: " << lidar_yaml_path << std::endl;
  YAML::Node lidar_node = YAML::LoadFile(lidar_yaml_path);
  
  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  std::string imu_yaml_path = config_path + imu_yaml;
  std::cout << "Loading IMU config: " << imu_yaml_path << std::endl;
  YAML::Node imu_node = YAML::LoadFile(imu_yaml_path);
  
  std::string uwb_yaml_rel = node["uwb_yaml"].as<std::string>();
  std::string uwb_yaml = config_path + uwb_yaml_rel;
  std::cout << "Loading UWB config: " << uwb_yaml << std::endl;
  YAML::Node uwb_node = YAML::LoadFile(uwb_yaml);
  odometry_mode_ = OdometryMode(node["odometry_mode"].as<int>());
  std::cout << "\n Odometry Mode: ";
  if (odometry_mode_ == LICO) {
    std::cout << "LiDAR-Inertial-Camera Odometry" << std::endl;
  } else if (odometry_mode_ == LIO) {
    std::cout << "LiDAR-Inertial Odometry" << std::endl;
  } else if (odometry_mode_ == LIUO) {
    std::cout << "LiDAR-Inertial-UWB Odometry" << std::endl;
  } else if (odometry_mode_ == LICUO) {
    std::cout << "LiDAR-Inertial-Camera-UWB Odometry" << std::endl;
  }

  // extrinsic: sensor to imu
  ExtrinsicParam EP_LtoI, EP_ItoI, EP_MtoI;
  EP_LtoI.Init(lidar_node["lidar0"]["Extrinsics"]);
  if (node["IMUExtrinsics"])
    EP_ItoI.Init(imu_node["IMUExtrinsics"]);
  EP_MtoI.Init(imu_node["MarkerExtrinsics"]);

  trajectory_ = std::make_shared<Trajectory>(-1, 0);
  trajectory_->SetSensorExtrinsics(SensorType::LiDARSensor, EP_LtoI);
  trajectory_->SetSensorExtrinsics(SensorType::IMUSensor, EP_ItoI);
  trajectory_->SetSensorExtrinsics(SensorType::Marker, EP_MtoI);
  trajectory_->SetSensorExtrinsics(SensorType::UWBSensor, EP_ItoI);

  // non-uniform b-spline
  t_add_ = node["t_add"].as<double>();
  t_add_ns_ = t_add_ * S_TO_NS;
  non_uniform_ = node["non_uniform"].as<bool>();
  distance0_ = node["distance0"].as<double>();

  // lidar
  lidar_iter_ = node["lidar_iter"].as<int>();
  use_lidar_scale_ = node["use_lidar_scale"].as<bool>();
  lidar_handler_ = std::make_shared<LidarHandler>(lidar_node, trajectory_);
  std::cout << "\n🍺 The number of multiple LiDARs is "
            << lidar_node["num_lidars"].as<int>() << "." << std::endl;

  // imu
  imu_initializer_ = std::make_shared<IMUInitializer>(imu_node);
  gravity_norm_ = imu_initializer_->GetGravity().norm();

  // uwb
  uwb_handler_ = std::make_shared<UWBHandler>(uwb_node);
  std::cout << "\n🍺 The number of UWB is " << uwb_node["num_uwb"].as<int>()
            << "." << std::endl;

  // trajectory parameterized by b-spline
  trajectory_manager_ = std::make_shared<TrajectoryManager>(node, trajectory_, config_file_path);
  trajectory_manager_->use_lidar_scale = use_lidar_scale_;
  trajectory_manager_->SetIntrinsic(K_);

  trajectory_manager_->SetLidarHandler(lidar_handler_.get());

  trajectory_manager_->SetOdometryViewer(&odom_viewer_);

  int division_coarse = node["division_coarse"].as<int>();
  cp_add_num_coarse_ = division_coarse;
  trajectory_manager_->SetDivisionParam(division_coarse, -1);

  odom_viewer_.SetPublisher(nh);

  msg_manager_ = std::make_shared<MsgManager>(node, nh, config_file_path); // load rosbag
  msg_manager_->anchor_id_positions = uwb_handler_->anchor_id_positions_;

  bool verbose;
  nh.param<double>("pasue_time", pasue_time_, -1);
  nh.param<bool>("verbose", verbose, false);
  trajectory_manager_->verbose = verbose;

  is_evo_viral_ = node["is_evo_viral"].as<bool>();

  if (node["save_trajectory"]) {
    const auto traj_node = node["save_trajectory"];
    save_trajectory_enable_ =
        traj_node["enable"] ? traj_node["enable"].as<bool>() : true;
    if (traj_node["output_dir"]) {
      trajectory_output_dir_ = traj_node["output_dir"].as<std::string>();
    }
  }
  if (node["save_pcd"]) {
    const auto pcd_node = node["save_pcd"];
    save_pcd_enable_ =
        pcd_node["enable"] ? pcd_node["enable"].as<bool>() : true;
    if (pcd_node["output_dir"]) {
      pcd_output_dir_ = pcd_node["output_dir"].as<std::string>();
    }
  }

  std::time_t now = std::time(nullptr);
  std::tm tm_now = *std::localtime(&now);
  std::ostringstream ts_stream;
  ts_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
  save_runtime_timestamp_ = ts_stream.str();

  InitRunBagTimingConfig(node);
  CreateCacheFolder(config_path, msg_manager_->bag_path_);

  std::cout << std::fixed << std::setprecision(4);
  LOG(INFO) << std::fixed << std::setprecision(4);
}

bool OdometryManager::CreateCacheFolder(const std::string &config_path,
                                        const std::string &bag_path) {
  namespace fs = boost::filesystem;

  boost::filesystem::path path_cfg(config_path);
  boost::filesystem::path path_bag(bag_path);
  std::string bag_name_ = "run";
  if (!path_bag.empty()) {
    bag_name_ = path_bag.stem().string();
  }
  if (bag_name_.empty()) {
    bag_name_ = "run";
  }

  std::string cache_path_parent_ = path_cfg.parent_path().string();
  cache_path_ = cache_path_parent_ + "/data/" + bag_name_;

  fs::path pkg_root = path_cfg.parent_path().parent_path();
  fs::path trajectory_output_path(trajectory_output_dir_);
  if (trajectory_output_path.is_relative()) {
    trajectory_output_path = pkg_root / trajectory_output_path;
  }
  if (!fs::exists(trajectory_output_path)) {
    fs::create_directories(trajectory_output_path);
  }
  trajectory_output_dir_ = trajectory_output_path.string();

  fs::path pcd_output_path(pcd_output_dir_);
  if (pcd_output_path.is_relative()) {
    pcd_output_path = pkg_root / pcd_output_path;
  }
  if (!fs::exists(pcd_output_path)) {
    fs::create_directories(pcd_output_path);
  }
  pcd_output_dir_ = pcd_output_path.string();

  save_file_prefix_ = bag_name_ + "_" + save_runtime_timestamp_;

  LOG(INFO) << "Trajectory save: "
            << (save_trajectory_enable_ ? "enabled" : "disabled")
            << ", dir=" << trajectory_output_dir_;
  LOG(INFO) << "PCD save: " << (save_pcd_enable_ ? "enabled" : "disabled")
            << ", dir=" << pcd_output_dir_;

  return true;
}

std::string OdometryManager::BuildSaveFilePath(const std::string &output_dir,
                                               const std::string &descri,
                                               const std::string &suffix,
                                               const std::string &extension) const {
  return output_dir + "/" + save_file_prefix_ + "_" + descri + suffix +
         extension;
}

void OdometryManager::InitRunBagTimingConfig(const YAML::Node &node) {
  if (!node["timing"]) {
    return;
  }

  const auto timing_node = node["timing"];
  timing_enable_ = timing_node["enable"] ? timing_node["enable"].as<bool>()
                                           : false;

  if (timing_node["output_file"]) {
    timing_output_file_ = timing_node["output_file"].as<std::string>();
  }

  if (timing_output_file_.empty()) {
    timing_output_file_ = "runbag_timing.csv";
  }
}

bool OdometryManager::InitRunBagTimingFileIfNeeded() {
  if (!timing_enable_) {
    return false;
  }

  if (timing_header_written_) {
    return true;
  }

  std::string log_dir = FLAGS_log_dir;
  if (log_dir.empty()) {
    log_dir = ".";
  }

  std::string file_name = timing_output_file_;
  std::string base_name = file_name;
  std::string extension;
  const auto dot_pos = file_name.find_last_of('.');
  if (dot_pos != std::string::npos && dot_pos != 0 && dot_pos + 1 < file_name.size()) {
    base_name = file_name.substr(0, dot_pos);
    extension = file_name.substr(dot_pos);
  }

  std::time_t now = std::time(nullptr);
  std::tm tm_now = *std::localtime(&now);
  std::ostringstream ts_stream;
  ts_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

  std::string runtime_file_name = base_name + "_" + ts_stream.str() + extension;
  timing_file_path_ = log_dir + "/" + runtime_file_name;

  std::ifstream check_file(timing_file_path_);
  bool need_header = !check_file.good() || check_file.peek() == std::ifstream::traits_type::eof();
  check_file.close();

  std::ofstream out_file(timing_file_path_, std::ios::app);
  if (!out_file.is_open()) {
    LOG(WARNING) << "Failed to open timing log file: " << timing_file_path_;
    return false;
  }

  if (need_header) {
    out_file << "frame_idx,status,traj_max_time_ns_next,lidar_t_min_ns,lidar_t_max_ns,uwb_t_ns,"
             << "scan_num,cp_add_num_cur,has_uwb,bag_spin_ms,prepare_msgs_ms,update_one_seg_ms,"
             << "solve_total_ms,process_liuo_ms,prior_ms,attr_ms,frame_total_ms"
             << std::endl;
  }

  out_file.close();
  timing_header_written_ = true;
  LOG(INFO) << "RunBag timing logging enabled: " << timing_file_path_;
  return true;
}

void OdometryManager::AppendRunBagTimingRecord(
    const RunBagTimingRecord &record) {
  if (!timing_enable_) {
    return;
  }

  if (!InitRunBagTimingFileIfNeeded()) {
    return;
  }

  std::ofstream out_file(timing_file_path_, std::ios::app);
  if (!out_file.is_open()) {
    LOG(WARNING) << "Failed to append timing log file: " << timing_file_path_;
    return;
  }

  out_file << record.frame_idx << "," << record.status << ","
           << record.traj_max_time_ns_next << "," << record.lidar_t_min_ns
           << "," << record.lidar_t_max_ns << "," << record.uwb_t_ns << ","
           << record.scan_num << "," << record.cp_add_num_cur << ","
           << record.has_uwb << "," << std::fixed << std::setprecision(6)
           << record.bag_spin_ms << "," << record.prepare_msgs_ms << ","
           << record.update_one_seg_ms << "," << record.solve_total_ms << ","
           << record.process_liuo_ms << "," << record.prior_ms << ","
           << record.attr_ms << "," << record.frame_total_ms << std::endl;

  out_file.close();
}

void OdometryManager::RunBag() {
  while (ros::ok()) {
    TicToc t_loop;

    /// [1] process a newly arrived frame of data: lidar or imu or camera
    TicToc t_spin;
    msg_manager_->SpinBagOnce();
    double bag_spin_ms = t_spin.toc();
    if (!msg_manager_->has_valid_msg_) {
      std::cout << "\n🍺 No valid msg.\n";
      break;
    }

    /// [2] static initialization, do not move at the begging!
    if (!is_initialized_) {
      while (!msg_manager_->imu_buf_.empty()) {
        imu_initializer_->FeedIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }

      if (imu_initializer_->StaticInitialIMUState()) {
        SetInitialState();
        std::cout << "\n🍺 Static initialization succeeds.\n";
        std::cout << "\n🍺 Trajectory start time: "
                  << trajectory_->GetDataStartTime() << " ns.\n";
      } else {
        continue;
      }
    }

    /// [3] prepare data for the latest time interval delta_t
    static bool is_two_seg_prepared = false;
    static int seg_msg_cnt = 0;
    if (!is_two_seg_prepared) {
      if (PrepareTwoSegMsgs(seg_msg_cnt)) // prepare interval0 and interval1
      {
        seg_msg_cnt++;
      }
      if (seg_msg_cnt == 2) // if interval0 and interval1 are ready
      {
        is_two_seg_prepared = true;
        UpdateTwoSeg();
        trajectory_->InitBlendMat(); // blending matrix is computed by knots of
                                     // b-spline
      } else {
        continue;
      }
    }

    /// [4] update trajectory segment in the latest time interval delta_t
    TicToc t_prepare;
    bool has_msg = PrepareMsgs();
    double prepare_msgs_ms = t_prepare.toc();
    if (has_msg) {
      // decide control point placement in the time interval delta_t by imu
      TicToc t_update_one_seg;
      UpdateOneSeg();
      double update_one_seg_ms = t_update_one_seg.toc();
      int offset = cp_add_num_cur + cp_add_num_next + cp_add_num_next_next;
      for (int i = 0; i < cp_add_num_cur; i++) {
        trajectory_->AddBlendMat(
            offset - i); // blending matrix is computed by knots of b-spline
      }
      trajectory_manager_->SetDivision(cp_add_num_cur);
      trajectory_->startIdx = trajectory_->knts.size() - 1 - offset -
                              2; // 2 serves as a margin or tolerance
      if (trajectory_->startIdx < 0) {
        trajectory_->startIdx = 0;
      }
      TicToc t_update;
      last_process_liuo_ms_ = -1.0;
      last_prior_ms_ = -1.0;
      last_attr_ms_ = -1.0;
      // fusing lidar-imu-camera to update the trajectory
      if (odometry_mode_ == LICO) {
        SolveLIO();
      } else if (odometry_mode_ == LIO) {
        SolveLIO();
      } else if (odometry_mode_ == LIUO) {
        SolveLIUO();
      } else if (odometry_mode_ == LICUO) {
        SolveLICUO();
      }
      double solve_total_ms = t_update.toc();
      LOG(INFO) << "[Update time]: " << solve_total_ms << " ms.";

      if (timing_enable_) {
        RunBagTimingRecord record;
        record.frame_idx = timing_frame_idx_++;
        record.status = "ok";
        record.traj_max_time_ns_next = traj_max_time_ns_next;
        record.lidar_t_min_ns = msg_manager_->cur_msgs.lidar_timestamp;
        record.lidar_t_max_ns = msg_manager_->cur_msgs.lidar_max_timestamp;
        record.uwb_t_ns = msg_manager_->cur_msgs.uwb_timestamp;
        record.scan_num = msg_manager_->cur_msgs.scan_num;
        record.cp_add_num_cur = cp_add_num_cur;
        record.has_uwb = msg_manager_->cur_msgs.if_have_uwb ? 1 : 0;
        record.bag_spin_ms = bag_spin_ms;
        record.prepare_msgs_ms = prepare_msgs_ms;
        record.update_one_seg_ms = update_one_seg_ms;
        record.solve_total_ms = solve_total_ms;
        record.process_liuo_ms = last_process_liuo_ms_;
        record.prior_ms = last_prior_ms_;
        record.attr_ms = last_attr_ms_;
        record.frame_total_ms = t_loop.toc();
        AppendRunBagTimingRecord(record);
      }

      // deep copy
      msg_manager_->cur_msgs = NextMsgs();
      msg_manager_->cur_msgs = msg_manager_->next_msgs;
      msg_manager_->cur_msgs.image = msg_manager_->next_msgs.image.clone();
      msg_manager_->cur_msgs.uwb_msg = msg_manager_->next_msgs.uwb_msg;
      msg_manager_->next_msgs = NextMsgs();
      msg_manager_->next_msgs = msg_manager_->next_next_msgs;
      msg_manager_->next_msgs.image =
          msg_manager_->next_next_msgs.image.clone();
      msg_manager_->next_msgs.uwb_msg = msg_manager_->next_next_msgs.uwb_msg;
      msg_manager_->next_next_msgs = NextMsgs();

      traj_max_time_ns_cur = traj_max_time_ns_next;
      traj_max_time_ns_next = traj_max_time_ns_next_next;
      cp_add_num_cur = cp_add_num_next;
      cp_add_num_next = cp_add_num_next_next;

      while (msg_manager_->image_buf_.size() > 10) {
        msg_manager_->image_buf_.pop_front();
      }
    }
  }
}

void OdometryManager::RunInSubscribeMode() {
  ros::NodeHandle nh("~");
  msg_manager_->InitSubscribers(nh);

  LOG(INFO) << "Starting realtime subscribe mode...";
  std::cout << "\n🥥 Running in realtime subscribe mode 🥥" << std::endl;

  ros::Rate rate(100); // 100Hz

  while (ros::ok()) {
    ros::spinOnce();

    /// [2] static initialization, do not move at the begging!
    if (!is_initialized_) {
      while (!msg_manager_->imu_buf_.empty()) {
        imu_initializer_->FeedIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }

      if (imu_initializer_->StaticInitialIMUState()) {
        SetInitialState();
        std::cout << "\n🍺 Static initialization succeeds.\n";
        std::cout << "\n🍺 Trajectory start time: "
                  << trajectory_->GetDataStartTime() << " ns.\n";
      } else {
        rate.sleep();
        continue;
      }
    }

    /// [3] prepare data for the latest time interval delta_t
    static bool is_two_seg_prepared = false;
    static int seg_msg_cnt = 0;
    if (!is_two_seg_prepared) {
      if (PrepareTwoSegMsgs(seg_msg_cnt)) // prepare interval0 and interval1
      {
        seg_msg_cnt++;
      }
      if (seg_msg_cnt == 2) // if interval0 and interval1 are ready
      {
        is_two_seg_prepared = true;
        UpdateTwoSeg();
        trajectory_->InitBlendMat(); // blending matrix is computed by knots of
                                     // b-spline
      } else {
        rate.sleep();
        continue;
      }
    }

    /// [4] update trajectory segment in the latest time interval delta_t
    bool has_new_msgs = PrepareMsgs();
    LOG(INFO) << "[Main loop] has_new_msgs: " << has_new_msgs
              << ", traj_max_time_ns_next_next: " << traj_max_time_ns_next_next;
    if (has_new_msgs) {
      // decide control point placement in the time interval delta_t by imu
      UpdateOneSeg();
      int offset = cp_add_num_cur + cp_add_num_next + cp_add_num_next_next;
      for (int i = 0; i < cp_add_num_cur; i++) {
        trajectory_->AddBlendMat(
            offset - i); // blending matrix is computed by knots of b-spline
      }
      trajectory_manager_->SetDivision(cp_add_num_cur);
      trajectory_->startIdx = trajectory_->knts.size() - 1 - offset -
                              2; // 2 serves as a margin or tolerance
      if (trajectory_->startIdx < 0) {
        trajectory_->startIdx = 0;
      }

      /// [5] process lidar and camera data
      if (odometry_mode_ == LICO) {
        ProcessLICData();
        SolveLIO();
      } else if (odometry_mode_ == LIO) {
        ProcessLICData();
        SolveLIO();
      } else if (odometry_mode_ == LIUO) {
        ProcessLIUOData();
        SolveLIUO();
      } else if (odometry_mode_ == LICUO) {
        ProcessLICUOData();
        SolveLICUO();
      }

      /// [6] publish trajectory and point cloud
      PublishCloudAndTrajectory();

      /// [7] remove old data in the buffer
      msg_manager_->RemoveBeginData(trajectory_->minTimeNs());

      /// [8] remove old image data
      while (!msg_manager_->image_buf_.empty()) {
        if (msg_manager_->image_buf_.front().timestamp <
            trajectory_->minTimeNs()) {
          msg_manager_->image_buf_.pop_front();
        } else {
          break;
        }
      }
    }

    /// [9] update time progression - only when we have processed data
    // Only update time progression when we have actually processed new messages
    // This prevents infinite time progression without real data
    if (has_new_msgs && traj_max_time_ns_next_next > 0) {
      LOG(INFO) << "[Before time update] traj_max_time_ns_cur: "
                << traj_max_time_ns_cur
                << ", traj_max_time_ns_next: " << traj_max_time_ns_next
                << ", traj_max_time_ns_next_next: "
                << traj_max_time_ns_next_next;

      traj_max_time_ns_cur = traj_max_time_ns_next;
      traj_max_time_ns_next = traj_max_time_ns_next_next;
      cp_add_num_cur = cp_add_num_next;
      cp_add_num_next = cp_add_num_next_next;

      LOG(INFO) << "[After time update] traj_max_time_ns_cur: "
                << traj_max_time_ns_cur
                << ", traj_max_time_ns_next: " << traj_max_time_ns_next;
    } else {
      LOG(INFO) << "[Time update skipped] has_new_msgs: " << has_new_msgs
                << ", traj_max_time_ns_next_next: "
                << traj_max_time_ns_next_next;
    }

    rate.sleep();
  }

  LOG(INFO) << "Realtime subscribe mode ended.";
}

void OdometryManager::SolveLIO() {
  msg_manager_->LogInfo();
  if (msg_manager_->cur_msgs.lidar_timestamp < 0) {
    LOG(INFO) << "CANT SolveLIO!";
  }
  TicToc t_solve;
  // lic optimization
  ProcessLICData();
  double t_process = t_solve.toc();
  TicToc t_prior_update;
  // prior update
  trajectory_manager_->UpdateLICPrior(lidar_handler_->GetPointCorrespondence());
  double t_prior = t_prior_update.toc();

  TicToc t_attribute_update;
  // remove old imu data
  auto &msg = msg_manager_->cur_msgs;
  trajectory_manager_->UpdateLiDARAttribute(msg.lidar_timestamp,
                                            msg.lidar_max_timestamp);
  double t_attribute = t_attribute_update.toc();
}

void OdometryManager::SolveLIUO() {
  msg_manager_->LogInfo();
  if (msg_manager_->cur_msgs.lidar_timestamp < 0) {
    LOG(INFO) << "CANT SolveLIUO!";
  }
  TicToc t_solve;
  // lic optimization with uwb
  ProcessLIUOData();
  double t_process = t_solve.toc();
  TicToc t_prior_update;

  // prior update
  trajectory_manager_->UpdateLICPrior(lidar_handler_->GetPointCorrespondence());
  double t_prior = t_prior_update.toc();

  TicToc t_attribute_update;
  // remove old imu data
  auto &msg = msg_manager_->cur_msgs;
  trajectory_manager_->UpdateLiDARAttribute(msg.lidar_timestamp,
                                            msg.lidar_max_timestamp);
  double t_attribute = t_attribute_update.toc();

  last_process_liuo_ms_ = t_process;
  last_prior_ms_ = t_prior;
  last_attr_ms_ = t_attribute;
}

void OdometryManager::SolveLICUO() {
  msg_manager_->LogInfo();
  if (msg_manager_->cur_msgs.lidar_timestamp < 0) {
    LOG(INFO) << "CANT SolveLICUO!";
  }

  // lic optimization with uwb
  ProcessLICUOData();

  // prior update
  trajectory_manager_->UpdateLICPrior(lidar_handler_->GetPointCorrespondence());

  // remove old imu data
  auto &msg = msg_manager_->cur_msgs;
  trajectory_manager_->UpdateLiDARAttribute(msg.lidar_timestamp,
                                            msg.lidar_max_timestamp);
}

void OdometryManager::ProcessLICData() {
  auto &msg =
      msg_manager_->cur_msgs; // fake points with timestamp -1 exist up to now
  // msg.CheckData();

  bool process_image =
      msg.if_have_image && msg.image_timestamp > t_begin_add_cam_;
  if (process_image) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; image_time: " << msg.image_timestamp * NS_TO_S;
  } else {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]";
  }

  /// [1] transform the format of lidar pointcloud ->
  /// feature_cur_、feature_cur_ds_
  if (!msg.lidar_corner_cloud->empty() && !msg.lidar_surf_cloud->empty() &&
      !msg.lidar_raw_cloud->empty()) {
    lidar_handler_->FeatureCloudHandler(
        msg.lidar_timestamp, msg.lidar_max_timestamp, msg.lidar_corner_cloud,
        msg.lidar_surf_cloud, msg.lidar_raw_cloud); // fake points are removed
  } else {
    ROS_WARN("Lidar cloud is empty", msg.lidar_corner_cloud->size(),
             msg.lidar_surf_cloud->size(), msg.lidar_raw_cloud->size());
  }

  /// [2] coarsely optimize trajectory based on prior、imu（served as good
  /// initial values）
  trajectory_manager_->PredictTrajectory(
      msg.lidar_timestamp, msg.lidar_max_timestamp, traj_max_time_ns_cur,
      cp_add_num_cur, non_uniform_);

  /// [3] update lidar local map
  int active_idx = trajectory_->numKnots() - 1 - cp_add_num_cur - 2;
  trajectory_->SetActiveTime(trajectory_->knts[active_idx]);
  lidar_handler_->UpdateLidarSubMap();

  /// TODO: Add UWB handler
  /// TODO: [5] finely optimize trajectory based on prior、lidar、imu、camera
  for (int iter = 0; iter < lidar_iter_; ++iter) {
    lidar_handler_->GetLoamFeatureAssociation();
    /// TODO: Add Ceres residule here!!
    if (process_image) {
      trajectory_manager_->UpdateTrajectoryWithLIC(
          iter, msg.image_timestamp, lidar_handler_->GetPointCorrespondence(),
          v_points_, px_obss_, 8);
    } else {
      trajectory_manager_->UpdateTrajectoryWithLIC(
          iter, msg.image_timestamp, lidar_handler_->GetPointCorrespondence(),
          {}, {}, 8);
      trajectory_manager_->SetProcessCurImg(false);
    }
  }
  PublishCloudAndTrajectory();

  /// [6] update visual global map
  PosCloud::Ptr cloud_undistort = PosCloud::Ptr(new PosCloud);
  auto latest_feature_before_active_time = lidar_handler_->GetFeatureCurrent();
  PosCloud::Ptr cloud_distort =
      latest_feature_before_active_time.surface_features;
  if (cloud_distort->size() != 0) {
    trajectory_->UndistortScanInG(*cloud_distort,
                                  latest_feature_before_active_time.timestamp,
                                  *cloud_undistort);
  }

  /// [8] visualize tf in rviz
  auto pose = trajectory_->GetLidarPoseNURBS(msg.lidar_timestamp);
  odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "lidar",
                         "map");
  odom_viewer_.PublishTF(trajectory_manager_->GetGlobalFrame(),
                         Eigen::Vector3d::Zero(), "map", "global");
}

void OdometryManager::ProcessLIUOData() {
  auto &msg = msg_manager_->cur_msgs;
  // msg.CheckData();

  bool process_image =
      msg.if_have_image && msg.image_timestamp > t_begin_add_cam_;
  bool process_uwb = msg.if_have_uwb;
  // std::cout << "process_image: " << process_image << std::endl;
  // std::cout << "process_uwb: " << process_uwb << std::endl;
  if (process_image && process_uwb) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; image_time: " << msg.image_timestamp * NS_TO_S
              << "; uwb_time: " << msg.uwb_timestamp * NS_TO_S;
  } else if (process_image) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; image_time: " << msg.image_timestamp * NS_TO_S;
  } else if (process_uwb) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; uwb_time: " << msg.uwb_timestamp * NS_TO_S;
  } else {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]";
  }

  /// [1] transform the format of lidar pointcloud ->
  /// feature_cur_、feature_cur_ds_
  if (!msg.lidar_corner_cloud->empty() && !msg.lidar_surf_cloud->empty() &&
      !msg.lidar_raw_cloud->empty()) {
    lidar_handler_->FeatureCloudHandler(
        msg.lidar_timestamp, msg.lidar_max_timestamp, msg.lidar_corner_cloud,
        msg.lidar_surf_cloud, msg.lidar_raw_cloud);
  } else {
    ROS_WARN("Lidar cloud is empty", msg.lidar_corner_cloud->size(),
             msg.lidar_surf_cloud->size(), msg.lidar_raw_cloud->size());
  }
  /// [2] coarsely optimize trajectory based on prior、imu (served as good
  /// initial values)
  trajectory_manager_->PredictTrajectory(
      msg.lidar_timestamp, msg.lidar_max_timestamp, traj_max_time_ns_cur,
      cp_add_num_cur, non_uniform_);

  /// [3] update lidar local map
  int active_idx = trajectory_->numKnots() - 1 - cp_add_num_cur - 2;
  trajectory_->SetActiveTime(trajectory_->knts[active_idx]);
  lidar_handler_->UpdateLidarSubMap();

  /// [4] update visual local map (tracking map points for the current image
  /// frame)
  v_points_.clear();
  px_obss_.clear();

  /// [5] update UWB measurements
  // if (process_uwb) {
  //   uwb_handler_->UpdateUWBMeasurements(msg.uwb);
  // }

  /// [6] finely optimize trajectory based on prior、lidar、imu、camera、uwb
  for (int iter = 0; iter < lidar_iter_; ++iter) {
    lidar_handler_->GetLoamFeatureAssociation();

    // TODO
    trajectory_manager_->UpdateTrajectoryWithLICU(
        iter, msg.image_timestamp, msg.uwb_timestamp,
      lidar_handler_->GetPointCorrespondence(), v_points_, px_obss_, 5);
    //  else if (process_image) {
    // trajectory_manager_->UpdateTrajectoryWithLIC(
    //     iter, msg.image_timestamp, lidar_handler_->GetPointCorrespondence(),
    //     v_points_, px_obss_, 8);
    // } else {
    //   trajectory_manager_->UpdateTrajectoryWithLIC(
    //       iter, msg.image_timestamp,
    //       lidar_handler_->GetPointCorrespondence(),
    //       {}, {}, 8);
    //   trajectory_manager_->SetProcessCurImg(false);
    // }
  }
  PublishCloudAndTrajectory();

  /// [9] visualize tf in rviz
  auto pose = trajectory_->GetLidarPoseNURBS(msg.lidar_timestamp);
  odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "lidar",
                         "map");
  odom_viewer_.PublishTF(trajectory_manager_->GetGlobalFrame(),
                         Eigen::Vector3d::Zero(), "map", "global");
}

void OdometryManager::ProcessLICUOData() {
  auto &msg = msg_manager_->cur_msgs;
  msg.CheckData();

  bool process_image =
      msg.if_have_image && msg.image_timestamp > t_begin_add_cam_;
  bool process_uwb = msg.if_have_uwb;

  if (process_image && process_uwb) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; image_time: " << msg.image_timestamp * NS_TO_S
              << "; uwb_time: " << msg.uwb_timestamp * NS_TO_S;
  } else if (process_image) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; image_time: " << msg.image_timestamp * NS_TO_S;
  } else if (process_uwb) {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]"
              << "; uwb_time: " << msg.uwb_timestamp * NS_TO_S;
  } else {
    LOG(INFO) << "Process " << msg.scan_num << " scans in ["
              << msg.lidar_timestamp * NS_TO_S << ", "
              << msg.lidar_max_timestamp * NS_TO_S << "]";
  }

  /// [1] transform the format of lidar pointcloud ->
  /// feature_cur_、feature_cur_ds_
  lidar_handler_->FeatureCloudHandler(
      msg.lidar_timestamp, msg.lidar_max_timestamp, msg.lidar_corner_cloud,
      msg.lidar_surf_cloud, msg.lidar_raw_cloud);

  /// [2] coarsely optimize trajectory based on prior、imu (served as good
  /// initial values)
  trajectory_manager_->PredictTrajectory(
      msg.lidar_timestamp, msg.lidar_max_timestamp, traj_max_time_ns_cur,
      cp_add_num_cur, non_uniform_);

  /// [3] update lidar local map
  int active_idx = trajectory_->numKnots() - 1 - cp_add_num_cur - 2;
  trajectory_->SetActiveTime(trajectory_->knts[active_idx]);
  lidar_handler_->UpdateLidarSubMap();

  /// [5] update UWB measurements
  // if (process_uwb) {
  //   uwb_handler_->UpdateUWBMeasurements(msg.uwb);
  // }

  /// [6] finely optimize trajectory based on prior、lidar、imu、camera、uwb
  for (int iter = 0; iter < lidar_iter_; ++iter) {
    lidar_handler_->GetLoamFeatureAssociation();

    // TODO
    // if (process_image && process_uwb) {
    //   trajectory_manager_->UpdateTrajectoryWithLICU(
    //       iter, msg.image_timestamp, msg.uwb_timestamp,
    //       lidar_handler_->GetPointCorrespondence(), v_points_, px_obss_,
    //       uwb_handler_->GetUWBMeasurements(), 8);
    // } else if (process_image) {
    trajectory_manager_->UpdateTrajectoryWithLIC(
        iter, msg.image_timestamp, lidar_handler_->GetPointCorrespondence(),
      v_points_, px_obss_, 5);
    // } else {
    //   trajectory_manager_->UpdateTrajectoryWithLIC(
    //       iter, msg.image_timestamp,
    //       lidar_handler_->GetPointCorrespondence(),
    //       {}, {}, 8);
    //   trajectory_manager_->SetProcessCurImg(false);
    // }
  }
  PublishCloudAndTrajectory();

  /// [7] update visual global map
  PosCloud::Ptr cloud_undistort = PosCloud::Ptr(new PosCloud);
  auto latest_feature_before_active_time = lidar_handler_->GetFeatureCurrent();
  PosCloud::Ptr cloud_distort =
      latest_feature_before_active_time.surface_features;
  if (cloud_distort->size() != 0) {
    trajectory_->UndistortScanInG(*cloud_distort,
                                  latest_feature_before_active_time.timestamp,
                                  *cloud_undistort);
  }

  /// [9] visualize tf in rviz
  auto pose = trajectory_->GetLidarPoseNURBS(msg.lidar_timestamp);
  odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "lidar",
                         "map");
  odom_viewer_.PublishTF(trajectory_manager_->GetGlobalFrame(),
                         Eigen::Vector3d::Zero(), "map", "global");
}

bool OdometryManager::PrepareTwoSegMsgs(int seg_idx) {
  if (!is_initialized_)
    return false;

  int64_t data_start_time = trajectory_->GetDataStartTime();
  for (auto &data : msg_manager_->lidar_buf_) {
    // if data.raw_cloud is empty, it means that the data is invalid, continue
    if (data.raw_cloud->empty()) {
      ROS_WARN("Invalid lidar data! Jump over.");
      continue;
    }
    if (!data.is_time_wrt_traj_start) {
      data.ToRelativeMeasureTime(data_start_time); //
      msg_manager_->lidar_max_timestamps_[data.lidar_id] =
          data.max_timestamp; //
    }
  }
  for (auto &data : msg_manager_->image_buf_) {
    if (!data.is_time_wrt_traj_start) {
      data.ToRelativeMeasureTime(data_start_time);
      msg_manager_->image_max_timestamp_ = data.timestamp;
    }
  }
  for (auto &data : msg_manager_->uwb_buf_) {
    if (!data.is_time_wrt_traj_start) {
      // data.ToRelativeMeasureTime(data_start_time);
      msg_manager_->uwb_max_timestamp_ = data.timestamp;
    }
  }
  msg_manager_->RemoveBeginData(data_start_time, 0);

  //
  //
  int64_t traj_max_time_ns =
      trajectory_->maxTimeNsNURBS() + t_add_ns_ * (seg_idx + 1);

  //
  //
  bool have_msg = false;
  if (seg_idx == 0) {
    int64_t traj_last_max_time_ns = trajectory_->maxTimeNsNURBS();
    have_msg =
        msg_manager_->GetMsgs(msg_manager_->cur_msgs, traj_last_max_time_ns,
                              traj_max_time_ns, data_start_time);
  }
  if (seg_idx == 1) {
    int64_t traj_last_max_time_ns = trajectory_->maxTimeNsNURBS() + t_add_ns_;
    have_msg =
        msg_manager_->GetMsgs(msg_manager_->next_msgs, traj_last_max_time_ns,
                              traj_max_time_ns, data_start_time);
  }

  if (have_msg) {
    while (!msg_manager_->imu_buf_.empty()) {
      trajectory_manager_->AddIMUData(msg_manager_->imu_buf_.front());
      msg_manager_->imu_buf_.pop_front();
    }
    if (seg_idx == 0) {
      traj_max_time_ns_cur = traj_max_time_ns;
      LOG(INFO) << "[PrepareTwoSegMsgs] seg_idx=0, set traj_max_time_ns_cur: "
                << traj_max_time_ns_cur;
    }
    if (seg_idx == 1) {
      traj_max_time_ns_next = traj_max_time_ns;
      LOG(INFO) << "[PrepareTwoSegMsgs] seg_idx=1, set traj_max_time_ns_next: "
                << traj_max_time_ns_next;
    }
    return true;
  } else {
    return false;
  }
}

void OdometryManager::UpdateTwoSeg() {
  auto imu_datas = trajectory_manager_->GetIMUData();

  /// update the first seg
  {
    int cp_add_num = cp_add_num_coarse_;
    Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(),
                    aver_a = Eigen::Vector3d::Zero();
    double var_r = 0, var_a = 0;
    int cnt = 0;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < trajectory_->maxTimeNsNURBS() ||
          imu_datas[i].timestamp >= traj_max_time_ns_cur)
        continue;
      cnt++;
      aver_r += imu_datas[i].gyro;
      aver_a += imu_datas[i].accel;
    }
    aver_r /= cnt;
    aver_a /= cnt;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < trajectory_->maxTimeNURBS() ||
          imu_datas[i].timestamp >= traj_max_time_ns_cur)
        continue;
      var_r += (imu_datas[i].gyro - aver_r).transpose() *
               (imu_datas[i].gyro - aver_r);
      var_a += (imu_datas[i].accel - aver_a).transpose() *
               (imu_datas[i].accel - aver_a);
    }
    var_r = sqrt(var_r / (cnt - 1));
    var_a = sqrt(var_a / (cnt - 1));
    LOG(INFO) << "[aver_r_first] " << aver_r.norm() << " | [aver_a_first] "
              << aver_a.norm();
    LOG(INFO) << "[var_r_first] " << var_r << " | [var_a_first] " << var_a;

    if (non_uniform_) { // Use non-uniform b-spline
      cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
    }
    LOG(INFO) << "[cp_add_num_first] " << cp_add_num;
    cp_num_vec.push_back(cp_add_num);
    // Add control points based on the IMU data
    int64_t step =
        (traj_max_time_ns_cur - trajectory_->maxTimeNsNURBS()) / cp_add_num;
    LOG(INFO) << "[extend_step_first] " << step;
    for (int i = 0; i < cp_add_num - 1; i++) {
      int64_t time = trajectory_->maxTimeNsNURBS() + step * (i + 1);
      trajectory_->AddKntNs(time);
    }
    trajectory_->AddKntNs(traj_max_time_ns_cur);

    cp_add_num_cur = cp_add_num;
  }

  /// update the second seg
  {
    int cp_add_num = cp_add_num_coarse_;
    Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(),
                    aver_a = Eigen::Vector3d::Zero();
    double var_r = 0, var_a = 0;
    int cnt = 0;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < traj_max_time_ns_cur ||
          imu_datas[i].timestamp >= traj_max_time_ns_next)
        continue;
      cnt++;
      aver_r += imu_datas[i].gyro;
      aver_a += imu_datas[i].accel;
    }
    aver_r /= cnt;
    aver_a /= cnt;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < traj_max_time_ns_cur ||
          imu_datas[i].timestamp >= traj_max_time_ns_next)
        continue;
      var_r += (imu_datas[i].gyro - aver_r).transpose() *
               (imu_datas[i].gyro - aver_r);
      var_a += (imu_datas[i].accel - aver_a).transpose() *
               (imu_datas[i].accel - aver_a);
    }
    var_r = sqrt(var_r / (cnt - 1));
    var_a = sqrt(var_a / (cnt - 1));
    LOG(INFO) << "[aver_r_second] " << aver_r.norm() << " | [aver_a_second] "
              << aver_a.norm();
    LOG(INFO) << "[var_r_second] " << var_r << " | [var_a_second] " << var_a;

    if (non_uniform_) { // Add control points based on the IMU data
      cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
    }
    LOG(INFO) << "[cp_add_num_second] " << cp_add_num;
    cp_num_vec.push_back(cp_add_num);

    int64_t step = (traj_max_time_ns_next - traj_max_time_ns_cur) / cp_add_num;
    LOG(INFO) << "[extend_step_second] " << step;
    for (int i = 0; i < cp_add_num - 1; i++) {
      int64_t time = traj_max_time_ns_cur + step * (i + 1);
      trajectory_->AddKntNs(time);
    }
    trajectory_->AddKntNs(traj_max_time_ns_next);

    cp_add_num_next = cp_add_num;
  }
}

// to relative measure time
bool OdometryManager::PrepareMsgs() {
  if (!is_initialized_)
    return false;

  int64_t data_start_time = trajectory_->GetDataStartTime();
  for (auto &data : msg_manager_->lidar_buf_) {
    if (data.raw_cloud->empty()) {
      ROS_WARN("Invalid lidar data! Jump over.");
      continue;
    }
    if (!data.is_time_wrt_traj_start) {
      data.ToRelativeMeasureTime(data_start_time); //
      msg_manager_->lidar_max_timestamps_[data.lidar_id] =
          data.max_timestamp; //
    }
  }
  for (auto &data : msg_manager_->image_buf_) {
    if (!data.is_time_wrt_traj_start) {
      data.ToRelativeMeasureTime(data_start_time);
      msg_manager_->image_max_timestamp_ = data.timestamp;
    }
  }
  for (auto &data : msg_manager_->uwb_buf_) {
    if (!data.is_time_wrt_traj_start) {
      // data.ToRelativeMeasureTime(data_start_time);
      msg_manager_->uwb_max_timestamp_ = data.timestamp;
      // LOG(INFO) << "[PrepareMsgs] UWB data time: " << data.timestamp;
    }
  }
  msg_manager_->RemoveBeginData(data_start_time, 0);

  int64_t traj_max_time_ns = traj_max_time_ns_next + t_add_ns_;

  int64_t traj_last_max_time_ns = traj_max_time_ns_next;
  bool have_msg =
      msg_manager_->GetMsgs(msg_manager_->next_next_msgs, traj_last_max_time_ns,
                            traj_max_time_ns, data_start_time);

  if (have_msg) {
    // TODO: check this condition
    while (!msg_manager_->imu_buf_.empty()) {
      trajectory_manager_->AddIMUData(msg_manager_->imu_buf_.front());
      msg_manager_->imu_buf_.pop_front();
    }
    while (!msg_manager_->uwb_buf_.empty()) {
      trajectory_manager_->AddUWBData(msg_manager_->uwb_buf_.front());
      msg_manager_->uwb_buf_.pop_front();
    }
    traj_max_time_ns_next_next = traj_max_time_ns;
    return true;
  } else {
    return false;
  }
}

void OdometryManager::UpdateOneSeg() {
  auto imu_datas = trajectory_manager_->GetIMUData();

  /// update the first seg
  {
    int cp_add_num = cp_add_num_coarse_;
    Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(),
                    aver_a = Eigen::Vector3d::Zero();
    double var_r = 0, var_a = 0;
    int cnt = 0;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < traj_max_time_ns_next ||
          imu_datas[i].timestamp >= traj_max_time_ns_next_next)
        continue;
      cnt++;
      aver_r += imu_datas[i].gyro;
      aver_a += imu_datas[i].accel;
    }
    aver_r /= cnt;
    aver_a /= cnt;
    for (int i = 0; i < imu_datas.size(); i++) {
      if (imu_datas[i].timestamp < traj_max_time_ns_next ||
          imu_datas[i].timestamp >= traj_max_time_ns_next_next)
        continue;
      var_r += (imu_datas[i].gyro - aver_r).transpose() *
               (imu_datas[i].gyro - aver_r);
      var_a += (imu_datas[i].accel - aver_a).transpose() *
               (imu_datas[i].accel - aver_a);
    }
    var_r = sqrt(var_r / (cnt - 1));
    var_a = sqrt(var_a / (cnt - 1));
    LOG(INFO) << "[aver_r_new] " << aver_r.norm() << " | [aver_a_new] "
              << aver_a.norm();
    LOG(INFO) << "[var_r_new] " << var_r << " | [var_a_new] " << var_a;

    if (non_uniform_) { // Use non-uniform b-spline
      cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
    }
    LOG(INFO) << "[cp_add_num_new] " << cp_add_num;
    cp_num_vec.push_back(cp_add_num);

    // Add control points based on the IMU data
    int64_t step =
        (traj_max_time_ns_next_next - traj_max_time_ns_next) / cp_add_num;
    LOG(INFO) << "[extend_step_new] " << step;
    for (int i = 0; i < cp_add_num - 1; i++) {
      int64_t time = traj_max_time_ns_next + step * (i + 1);
      trajectory_->AddKntNs(time);
    }
    trajectory_->AddKntNs(traj_max_time_ns_next_next);

    cp_add_num_next_next = cp_add_num;
  }
}

void OdometryManager::SetInitialState() {
  if (is_initialized_) {
    assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
    std::cout << RED << "[Error] system state has been initialized" << RESET
              << std::endl;
    return;
  }

  is_initialized_ = true;

  if (imu_initializer_->InitialDone()) {
    SystemState sys_state = imu_initializer_->GetIMUState(); // I0toG
    trajectory_manager_->SetSystemState(sys_state, distance0_);

    trajectory_manager_->AddIMUData(imu_initializer_->GetIMUData().back());
    msg_manager_->imu_buf_.clear();
  }

  assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
}

void OdometryManager::PublishCloudAndTrajectory() {
  // Publish UWB NLOS Detection Visualization
  if (trajectory_manager_ && trajectory_manager_->GetNLOSDetector()) {
    odom_viewer_.PublishNLOSVisualization(
        trajectory_manager_->GetNLOSDetector());
  }

  // Add UWB Anchors Visualization
  if (odom_viewer_.pub_uwb_anchors_.getNumSubscribers() != 0 && uwb_handler_) {
    visualization_msgs::MarkerArray anchor_markers;

    // Get UWB anchor positions (assuming UWBHandler has GetAnchorPositions interface)
    int id = 0;
    for (const auto &[anchor_id, position] :
         msg_manager_->anchor_id_positions) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = ros::Time::now();
      marker.ns = "uwb_anchors";
      marker.id = id++;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = position.x();
      marker.pose.position.y = position.y();
      marker.pose.position.z = position.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.5;
      marker.scale.y = 0.5;
      marker.scale.z = 0.5;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.lifetime = ros::Duration(1000);
      anchor_markers.markers.push_back(marker);
    }
    odom_viewer_.pub_uwb_anchors_.publish(anchor_markers);
  }

  odom_viewer_.PublishDenseCloud(trajectory_, lidar_handler_->GetFeatureMapDs(),
                                 lidar_handler_->GetFeatureCurrent());

  odom_viewer_.PublishSplineTrajectory(trajectory_, 0.0,
                                       trajectory_->maxTimeNURBS(), 0.1);
}

double OdometryManager::SaveOdometry() {
  std::string descri;
  if (odometry_mode_ == LICO)
    descri = "LICO";
  else if (odometry_mode_ == LIO)
    descri = "LIO";
  else if (odometry_mode_ == LICUO)
    descri = "LICUO";
  else if (odometry_mode_ == LIUO)
    descri = "LIUO";

  if (msg_manager_->NumLiDAR() > 1)
    descri = descri + "2";

  int idx = -1;
  int64_t true_maxtime = trajectory_->maxTimeNsNURBS();
  for (int i = trajectory_->knts.size() - 1; i >= 0; i--) {
    if (true_maxtime == trajectory_->knts[i]) {
      idx = i;
      break;
    }
  }
  idx -= 1;
  int64_t maxtime = trajectory_->knts[idx];
  maxtime = trajectory_->maxTimeNsNURBS() - 0.1 * S_TO_NS;

  if (save_trajectory_enable_) {
    const std::string trajectory_file =
        BuildSaveFilePath(trajectory_output_dir_, descri, "", ".txt");
    trajectory_->ToTUMTxt(trajectory_file, maxtime, is_evo_viral_,
                          0.01); // 100Hz pose querying
    LOG(INFO) << "✓ Trajectory saved: " << trajectory_file;
  } else {
    LOG(INFO) << "Trajectory saving disabled by config.";
  }

  // int sum_cp = std::accumulate(cp_num_vec.begin(), cp_num_vec.end(), 0);
  // std::cout << GREEN << "ave_cp_num " << sum_cp * 1.0 / cp_num_vec.size() <<
  // RESET << std::endl;

  // Save optimized UWB anchor positions
  // const auto &anchor_positions =
  //     trajectory_manager_->GetAllUWBAnchorPositions();
  // if (!anchor_positions.empty()) {
  //   std::ofstream anchor_out(cache_path_ + "_" + descri + "_uwb_anchors.txt",
  //                            std::ios::app);
  //   std::cout << "Save optimized UWB anchor positions to "
  //             << cache_path_ + "_" + descri + "_uwb_anchors.txt" <<
  //             std::endl;
  //   if (anchor_out.is_open()) {
  //     for (const auto &kv : anchor_positions) {
  //       const Eigen::Vector3d &p = kv.second;
  //       anchor_out << maxtime << " " << std::setprecision(12) << kv.first <<
  //       " "
  //                  << p.x() << " " << p.y() << " " << p.z() << std::endl;
  //     }
  //   }
  // }

  // ========== Save Point Cloud Map (New) ==========
  if (save_pcd_enable_) {
    SavePointCloudMap(descri);
  } else {
    LOG(INFO) << "Point cloud map saving disabled by config.";
  }

  // ========== Save accumulated NLOS obstacle cloud ==========
  if (trajectory_manager_) {
    auto nlos_detector = trajectory_manager_->GetNLOSDetector();
    if (nlos_detector && nlos_detector->IsNLOSDetectionEnabled()) {
      size_t nlos_cloud_size = nlos_detector->GetAccumulatedNLOSCloudSize();
      if (nlos_cloud_size > 0) {
        std::string nlos_filename = BuildSaveFilePath(
          pcd_output_dir_, descri, "_nlos_obstacles", ".pcd");
        bool success =
            nlos_detector->SaveNLOSObstacleCloud(nlos_filename, true);
        if (success) {
          LOG(INFO) << "✓ NLOS obstacle cloud saved: " << nlos_filename << " ("
                    << nlos_cloud_size << " points)";
        } else {
          LOG(WARNING) << "✗ Failed to save NLOS obstacle cloud";
        }
      } else {
        LOG(INFO) << "No NLOS obstacles detected, skipping save";
      }

      // ========== Save NLOS time series records ==========
      size_t record_count = nlos_detector->GetTimeSeriesRecordCount();
      std::cout << "record_count: " << record_count << std::endl;
      if (record_count > 0) {
        std::string timeseries_filename = BuildSaveFilePath(
          trajectory_output_dir_, descri, "_nlos_timeseries", ".txt");
        bool success =
            nlos_detector->SaveTimeSeriesRecords(timeseries_filename);
        if (success) {
          LOG(INFO) << "✓ NLOS time series saved: " << timeseries_filename
                    << " (" << record_count << " records)";
        } else {
          LOG(WARNING) << "✗ Failed to save NLOS time series";
        }
      } else {
        LOG(INFO) << "No NLOS time series, skipping save";
      }

      // ========== Save NLOS timing records ==========
      size_t timing_count = nlos_detector->GetTimingRecordCount();
      std::cout << "timing_count: " << timing_count << std::endl;
      if (timing_count > 0) {
        std::string timing_filename = BuildSaveFilePath(
          trajectory_output_dir_, descri, "_nlos_timing", ".txt");
        bool success = nlos_detector->SaveTimingRecords(timing_filename);
        if (success) {
          LOG(INFO) << "✓ NLOS timing saved: " << timing_filename << " ("
                    << timing_count << " records)";
        } else {
          LOG(WARNING) << "✗ Failed to save NLOS timing records";
        }
      } else {
        LOG(INFO) << "No NLOS timing records, skipping save";
      }
    }
  }

  return trajectory_->maxTimeNURBS();
}

void OdometryManager::SavePointCloudMap(const std::string &descri) {
  LOG(INFO) << "\n========== Saving Point Cloud Map ==========";

  // Get keyframe map container
  const auto &keyframes = lidar_handler_->GetLocalFeatureContainer();
  const auto &keyframes_ds =
      lidar_handler_->GetLocalFeatureContainerDownsampled();

  if (keyframes.empty()) {
    LOG(WARNING) << "Map is empty, nothing to save!";
    return;
  }

  LOG(INFO) << "Keyframes: " << keyframes.size();

  // Option 1: Recommended - Keyframe downsampled map (memory efficient)
  {
    PosCloud::Ptr global_map(new PosCloud);
    size_t total_points = 0;

    for (const auto &kf_pair : keyframes_ds) {
      int64_t timestamp = kf_pair.first;
      const LiDARFeature &feature = kf_pair.second;

      // Use surface features (most common)
      if (!feature.surface_features->empty()) {
        // Transform to global frame
        SE3d pose = trajectory_->GetLidarPoseNURBS(timestamp);
        PosCloud::Ptr global_cloud(new PosCloud);
        pcl::transformPointCloud(*feature.surface_features, *global_cloud,
                                 pose.matrix());
        *global_map += *global_cloud;
        total_points += global_cloud->size();
      }
    }

    // Final downsample to remove overlap
    PosCloud::Ptr final_map(new PosCloud);
    VoxelFilter<PosPoint> voxel_filter;
    voxel_filter.SetResolution(0.001); // 10cm voxel
    voxel_filter.SetInputCloud(global_map);
    voxel_filter.Filter(final_map);

    // Save as binary compressed PCD
    std::string filename =
      BuildSaveFilePath(pcd_output_dir_, descri, "_map", ".pcd");
    int ret = pcl::io::savePCDFileBinaryCompressed(filename, *final_map);

    if (ret == 0) {
      LOG(INFO) << "✓ Map saved: " << filename;
      LOG(INFO) << "  Keyframes: " << keyframes_ds.size();
      LOG(INFO) << "  Original points: " << total_points;
      LOG(INFO) << "  Saved points: " << final_map->size();
      LOG(INFO) << "  Compression: " << std::fixed << std::setprecision(2)
                << (double)total_points / final_map->size() << "x";

      // Estimate file size
      size_t file_size_mb = (final_map->size() * 16) / (1024 * 1024);
      LOG(INFO) << "  Est. size: ~" << file_size_mb << " MB";
    } else {
      LOG(ERROR) << "✗ Failed to save map!";
    }
  }

  // Option 2: Optional - Save corner and surface separately (sparse feature map)
  // Uncomment below to enable
  /*
  {
    PosCloud::Ptr corner_map(new PosCloud);
    PosCloud::Ptr surface_map(new PosCloud);

    for (const auto& kf_pair : keyframes_ds) {
      int64_t timestamp = kf_pair.first;
      const LiDARFeature& feature = kf_pair.second;
      SE3d pose = trajectory_->GetLidarPoseNURBS(timestamp);

      if (!feature.corner_features->empty()) {
        PosCloud::Ptr global_corners(new PosCloud);
        pcl::transformPointCloud(*feature.corner_features, *global_corners,
  pose.matrix()); *corner_map += *global_corners;
      }

      if (!feature.surface_features->empty()) {
        PosCloud::Ptr global_surfaces(new PosCloud);
        pcl::transformPointCloud(*feature.surface_features, *global_surfaces,
  pose.matrix()); *surface_map += *global_surfaces;
      }
    }

    if (!corner_map->empty()) {
      std::string corner_file = cache_path_ + "_" + descri + "_corners.pcd";
      pcl::io::savePCDFileBinaryCompressed(corner_file, *corner_map);
      LOG(INFO) << "✓ Corner map: " << corner_file << " (" << corner_map->size()
  << " points)";
    }

    if (!surface_map->empty()) {
      std::string surface_file = cache_path_ + "_" + descri + "_surfaces.pcd";
      pcl::io::savePCDFileBinaryCompressed(surface_file, *surface_map);
      LOG(INFO) << "✓ Surface map: " << surface_file << " (" << surface_map->size()
  << " points)";
    }
  }
  */

  LOG(INFO) << "========== Map Save Completed ==========\n";
}

} // namespace liuo
