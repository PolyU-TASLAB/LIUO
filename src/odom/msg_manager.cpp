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

#include <odom/msg_manager.h>
#include <boost/filesystem.hpp>
#include <pcl-1.13/pcl/common/transforms.h>
#include <utils/parameter_struct.h>

namespace liuo {

MsgManager::MsgManager(const YAML::Node &node, ros::NodeHandle &nh,
                       const std::string &config_file_path)
    : has_valid_msg_(true), t_offset_imu_(0), t_offset_camera_(0),
      cur_imu_timestamp_(-1), cur_pose_timestamp_(-1),
      lidar_timestamp_end_(false), remove_wrong_time_imu_(false),
      if_normalized_(false), image_topic_(""), uwb_topic_(""),
      anchor_id_positions(), is_realtime_mode_(false) {
  
  // Resolve config path relative to the config file itself
  boost::filesystem::path config_file_p(config_file_path);
  std::string config_path = config_file_p.parent_path().string();
  
  if (!config_path.empty() && config_path.back() != '/') {
      config_path += "/";
  }

  // Check run mode
  std::string run_mode = "bag"; // Default to bag mode
  if (node["run_mode"]) {
    run_mode = node["run_mode"].as<std::string>();
  }
  is_realtime_mode_ = (run_mode == "realtime");

  OdometryMode odom_mode = OdometryMode(node["odometry_mode"].as<int>());

  nh.param<std::string>("bag_path", bag_path_, "");
  if (bag_path_ == "") {
    bag_path_ = node["bag_path"].as<std::string>();
  }
  
  // Smart resolve bag path
  if (!boost::filesystem::exists(bag_path_)) {
       // Try relative to config path
       std::string rel_bag_path = config_path + bag_path_;
       if (boost::filesystem::exists(rel_bag_path)) {
           bag_path_ = rel_bag_path;
           std::cout << "Resolved bag path to: " << bag_path_ << std::endl;
       } else {
           // Try checking if it was just missing a slash or relative to parent of config
           // config_path is .../config/
           // Maybe bag is in .../data/ ?
           // Let's try to find it relative to project root (parent of config)
           // Use config_file_path (file) -> parent is config dir -> parent is project root
           boost::filesystem::path cfg_file_p(config_file_path);
           std::string project_root = cfg_file_p.parent_path().parent_path().string();
           std::string proj_bag_path = project_root + bag_path_; 
           if (boost::filesystem::exists(proj_bag_path)) {
               bag_path_ = proj_bag_path;
               std::cout << "Resolved bag path: " << bag_path_ << std::endl;
           } else {
               // User request: also look in package/data/ folder directly using just the filename
               boost::filesystem::path original_bag_p(bag_path_);
               std::string filename = original_bag_p.filename().string();
               std::string data_dir_bag_path = project_root + "/data/" + filename;
               
               if (boost::filesystem::exists(data_dir_bag_path)) {
                   bag_path_ = data_dir_bag_path;
                   std::cout << "Resolved bag path: " << bag_path_ << std::endl;
               } else {
                   std::cerr << "Error: bag file not found!" << std::endl;
                   std::cerr << "Tried: " << bag_path_ << std::endl;
                   std::cerr << "Tried: " << rel_bag_path << std::endl;
                   std::cerr << "Tried: " << proj_bag_path << std::endl;
                   std::cerr << "Tried: " << data_dir_bag_path << std::endl;
               }
           }
       }
  }

  /// imu topic
  std::string imu_yaml = node["imu_yaml"].as<std::string>();
  std::string imu_yaml_path = config_path + imu_yaml;
  // std::cout << "Loading IMU config: " << imu_yaml_path << std::endl;
  YAML::Node imu_node = YAML::LoadFile(imu_yaml_path);
  imu_topic_ = imu_node["imu_topic"].as<std::string>();
  // pose_topic_ = imu_node["pose_topic"].as<std::string>();
  // remove_wrong_time_imu_ = imu_node["remove_wrong_time_imu"].as<bool>();
  if_normalized_ = imu_node["if_normalized"].as<bool>();

  // double imu_frequency = node["imu_frequency"].as<double>();
  // double imu_period_s = 1. / imu_frequency;

  // add_extra_timeoffset_s_ =
  //     yaml::GetValue<double>(node, "add_extra_timeoffset_s", 0);
  // LOG(INFO) << "add_extra_timeoffset_s: " << add_extra_timeoffset_s_;
  // std::cout << "add_extra_timeoffset_s: " << add_extra_timeoffset_s_ << "\n";

  /// uwb topic
  std::string uwb_yaml = node["uwb_yaml"].as<std::string>();
  std::string uwb_yaml_path = config_path + uwb_yaml;
  // std::cout << "Loading UWB config: " << uwb_yaml_path << std::endl;
  YAML::Node uwb_node = YAML::LoadFile(uwb_yaml_path);
  uwb_topic_ = uwb_node["uwb_range_topic"].as<std::string>();

  // Read and save correspondence between anchor ID and position
  std::vector<int> anchor_ids = uwb_node["AnchorId"].as<std::vector<int>>();
  std::vector<double> anchor_positions =
      uwb_node["AnchorPos"].as<std::vector<double>>();

  // Store ID and position correspondence in map
  for (size_t i = 0; i < anchor_ids.size() - 1;
       i++) { // -1 because the last one is the mobile module
    Eigen::Vector3d pos(anchor_positions[i * 3], anchor_positions[i * 3 + 1],
                        anchor_positions[i * 3 + 2]);
    anchor_id_positions[anchor_ids[i]] = pos;
  }

  /// lidar topic
  std::string lidar_yaml = node["lidar_yaml"].as<std::string>();
  std::string lidar_yaml_path = config_path + lidar_yaml;
  // std::cout << "Loading Lidar config: " << lidar_yaml_path << std::endl;
  YAML::Node lidar_node = YAML::LoadFile(lidar_yaml_path);
  num_lidars_ = lidar_node["num_lidars"].as<int>();
  lidar_timestamp_end_ = lidar_node["lidar_timestamp_end"].as<bool>();

  bool use_livox = false;
  bool use_vlp = false;
  for (int i = 0; i < num_lidars_; ++i) {
    std::string lidar_str = "lidar" + std::to_string(i);
    const auto &lidar_i = lidar_node[lidar_str];
    bool is_livox = lidar_i["is_livox"].as<bool>();
    if (is_livox) {
      lidar_types.push_back(LIVOX);
      use_livox = true;
    } else {
      lidar_types.push_back(VLP);
      use_vlp = true;
    }
    lidar_topics_.push_back(lidar_i["topic"].as<std::string>());
    EP_LktoI_.emplace_back();
    EP_LktoI_.back().Init(lidar_i["Extrinsics"]);
  }

  for (int k = 0; k < num_lidars_; ++k) {
    lidar_max_timestamps_.push_back(0);
    Eigen::Matrix4d T_Lk_to_L0 = Eigen::Matrix4d::Identity();
    if (k > 0) {
      T_Lk_to_L0.block<3, 3>(0, 0) =
          (EP_LktoI_[0].q.inverse() * EP_LktoI_[k].q).toRotationMatrix();
      T_Lk_to_L0.block<3, 1>(0, 3) =
          EP_LktoI_[0].q.inverse() * (EP_LktoI_[k].p - EP_LktoI_[0].p);

      // std::cout << "lidar " << k << "\n"
      //           << T_Lk_to_L0 << std::endl;
    }
    T_LktoL0_vec_.push_back(T_Lk_to_L0);
  }

  if (use_livox)
    livox_feature_extraction_ =
        std::make_shared<LivoxFeatureExtraction>(lidar_node);
  if (use_vlp)
    velodyne_feature_extraction_ =
        std::make_shared<VelodyneFeatureExtraction>(lidar_node);

  // Only load bag file in bag mode
  if (!is_realtime_mode_) {
    LoadBag(node);
  }
}

void MsgManager::LoadBag(const YAML::Node &node) {
  double bag_start = node["bag_start"].as<double>();
  double bag_durr = node["bag_durr"].as<double>();

  std::vector<std::string> topics;
  topics.push_back(imu_topic_); // imu
  for (auto &v : lidar_topics_) // lidar
    topics.push_back(v);
  // topics.push_back(pose_topic_);
  topics.push_back(uwb_topic_);

  bag_.open(bag_path_, rosbag::bagmode::Read);

  rosbag::View view_full;
  view_full.addQuery(bag_);
  ros::Time time_start = view_full.getBeginTime();
  time_start += ros::Duration(bag_start);
  ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime()
                                         : time_start + ros::Duration(bag_durr);
  view_.addQuery(bag_, rosbag::TopicQuery(topics), time_start, time_finish);
  if (view_.size() == 0) {
    ROS_ERROR("No messages to play on specified topics.  Exiting.");
    ros::shutdown();
    return;
  }

  std::cout << "\n🍺 LoadBag " << bag_path_ << " start at " << bag_start
            << " with duration " << (time_finish - time_start).toSec() << ".\n";
  LOG(INFO) << "LoadBag " << bag_path_ << " start at " << bag_start
            << " with duration " << (time_finish - time_start).toSec();
}

void MsgManager::SpinBagOnce() {
  static rosbag::View::iterator view_iterator = view_.begin();
  if (view_iterator == view_.end()) {
    has_valid_msg_ = false;
    LOG(INFO) << "End of bag";
    return;
  }

  const rosbag::MessageInstance &m = *view_iterator;
  std::string msg_topic = m.getTopic();
  auto msg_time = m.getTime();

  if (msg_topic == imu_topic_) // imu
  {
    sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
    IMUMsgHandle(imu_msg);
  } else if (std::find(lidar_topics_.begin(), lidar_topics_.end(), msg_topic) !=
             lidar_topics_.end()) // lidar
  {
    auto it = std::find(lidar_topics_.begin(), lidar_topics_.end(), msg_topic);
    auto idx = std::distance(lidar_topics_.begin(), it);
    if (lidar_types[idx] == VLP) //[rotating lidar: Velodyne、Ouster、Hesai]
    {
      if (!m.isType<sensor_msgs::PointCloud2>())
        std::cout << "Wrong type\n";

      auto lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
      CheckLidarMsgTimestamp(msg_time.toSec(), lidar_msg->header.stamp.toSec());
      VelodyneMsgHandle(lidar_msg, idx);
      // VelodyneMsgHandleNoFeature(lidar_msg, idx);
    } else if (lidar_types[idx] == LIVOX) //[solid-state lidar: Livox]
    {
      if (!m.isType<livox_ros_driver2::CustomMsg>())
        std::cout << "Wrong type\n";

      auto lidar_msg = m.instantiate<livox_ros_driver2::CustomMsg>();
      CheckLidarMsgTimestamp(msg_time.toSec(), lidar_msg->header.stamp.toSec());
      LivoxMsgHandle(lidar_msg, idx); // Extract features & put in LiDAR buffer
    }
  } else if (msg_topic == image_topic_) // camera
  {
    if (if_compressed_) {
      sensor_msgs::CompressedImageConstPtr image_msg =
          m.instantiate<sensor_msgs::CompressedImage>();
      ImageMsgHandle(image_msg);
    } else {
      sensor_msgs::ImageConstPtr image_msg =
          m.instantiate<sensor_msgs::Image>();
      ImageMsgHandle(image_msg);
    }
  } else if (msg_topic == uwb_topic_) {
    std::string uwb_topic_1 = "/nlink_linktrack_tagframe0";
    std::string uwb_topic_2 = "/nlink_linktrack_nodeframe3";
    if (uwb_topic_ == uwb_topic_1) {
      auto uwb_msg = m.instantiate<liuo::LinktrackTagframe0>();
      UwbMsgHandle(uwb_msg);
      if (uwb_buf_.empty()) {
        LOG(ERROR) << "UWB buf empty ";
      }
    } else if (uwb_topic_ == uwb_topic_2) {
      auto uwb_msg = m.instantiate<liuo::LinktrackNodeframe3>();
      UwbMsgHandle(uwb_msg);
      if (uwb_buf_.empty()) {
        LOG(ERROR) << "UWB buf empty ";
      }
    }
  }

  view_iterator++;
}

void MsgManager::LogInfo() const {
  int m_size[3] = {0, 0, 0};
  m_size[0] = imu_buf_.size();
  m_size[1] = lidar_buf_.size();
  // if (use_image_) m_size[2] = feature_tracker_node_->NumImageMsg();
  LOG(INFO) << "imu/lidar/image msg left: " << m_size[0] << "/" << m_size[1]
            << "/" << m_size[2];
}

void MsgManager::RemoveBeginData(int64_t start_time,            // not used
                                 int64_t relative_start_time) { // 0
  for (auto iter = lidar_buf_.begin(); iter != lidar_buf_.end();) {
    if (iter->timestamp < relative_start_time) {
      if (iter->max_timestamp <= relative_start_time) { // [1]
        iter = lidar_buf_.erase(iter);
        continue;
      } else { // [2]
        // int64_t t_aft = relative_start_time + 1e-3;  //1e-3
        int64_t t_aft = relative_start_time;
        LiDARCloudData scan_bef, scan_aft;
        scan_aft.timestamp = t_aft;
        scan_aft.max_timestamp = iter->max_timestamp;
        pcl::FilterCloudByTimestamp(iter->raw_cloud, t_aft, scan_bef.raw_cloud,
                                    scan_aft.raw_cloud);
        pcl::FilterCloudByTimestamp(iter->surf_cloud, t_aft,
                                    scan_bef.surf_cloud, scan_aft.surf_cloud);
        pcl::FilterCloudByTimestamp(iter->corner_cloud, t_aft,
                                    scan_bef.corner_cloud,
                                    scan_aft.corner_cloud);

        iter->timestamp = t_aft;
        *iter->raw_cloud = *scan_aft.raw_cloud;
        *iter->surf_cloud = *scan_aft.surf_cloud;
        *iter->corner_cloud = *scan_aft.corner_cloud;
      }
    }

    iter++;
  }

  for (auto iter = uwb_buf_.begin(); iter != uwb_buf_.end();) {
    if (iter->timestamp < relative_start_time) {
      iter = uwb_buf_.erase(iter); //
      LOG(INFO) << "Remove begin uwb data";
      continue;
    }
    iter++;
  }
}

bool MsgManager::HasEnvMsg() const {
  int env_msg = lidar_buf_.size();
  if (cur_imu_timestamp_ < 0 && env_msg > 100)
    LOG(WARNING) << "No IMU data. CHECK imu topic" << imu_topic_;

  return env_msg > 0;
}

bool MsgManager::CheckMsgIsReady(double traj_max, double start_time,
                                 double knot_dt, bool in_scan_unit) const {
  double t_imu_wrt_start = cur_imu_timestamp_ - start_time;

  //
  if (t_imu_wrt_start < traj_max) {
    return false;
  }

  //
  int64_t t_front_lidar = -1;
  // Count how many unique lidar streams
  std::vector<int> unique_lidar_ids;
  for (const auto &data : lidar_buf_) {
    if (std::find(unique_lidar_ids.begin(), unique_lidar_ids.end(),
                  data.lidar_id) != unique_lidar_ids.end())
      continue;
    unique_lidar_ids.push_back(data.lidar_id);

    //
    t_front_lidar = std::max(t_front_lidar, data.max_timestamp);
  }

  //
  if ((int)unique_lidar_ids.size() != num_lidars_)
    return false;

  //
  int64_t t_back_lidar = lidar_max_timestamps_[0];
  for (auto t : lidar_max_timestamps_) {
    t_back_lidar = std::min(t_back_lidar, t);
  }

  //
  if (in_scan_unit) {
    //
    if (t_front_lidar > t_imu_wrt_start)
      return false;
  } else {
    //
    if (t_back_lidar < traj_max)
      return false;
  }

  return true;
}

bool MsgManager::AddImageToMsg(NextMsgs &msgs, const ImageData &image,
                               int64_t traj_max) {
  if (image.timestamp >= traj_max)
    return false;
  msgs.if_have_image = true; // important!
  msgs.image_timestamp = image.timestamp;
  msgs.image = image.image;
  // msgs.image = image.image.clone();
  return true;
}

bool MsgManager::AddToMsg(NextMsgs &msgs,
                          std::deque<LiDARCloudData>::iterator scan,
                          int64_t traj_max) {
  bool add_entire_scan = false;
  // if (scan->timestamp > traj_max) return add_entire_scan;

  if (scan->max_timestamp < traj_max) { //
    *msgs.lidar_raw_cloud += (*scan->raw_cloud);
    *msgs.lidar_surf_cloud += (*scan->surf_cloud);
    *msgs.lidar_corner_cloud += (*scan->corner_cloud);

    //
    if (msgs.scan_num == 0) {
      // first scan
      msgs.lidar_timestamp = scan->timestamp;
      msgs.lidar_max_timestamp = scan->max_timestamp;
    } else {
      msgs.lidar_timestamp = std::min(msgs.lidar_timestamp, scan->timestamp);
      msgs.lidar_max_timestamp =
          std::max(msgs.lidar_max_timestamp, scan->max_timestamp);
    }

    add_entire_scan = true;
  } else { //
    LiDARCloudData scan_bef, scan_aft;
    pcl::FilterCloudByTimestamp(scan->raw_cloud, traj_max, scan_bef.raw_cloud,
                                scan_aft.raw_cloud);
    pcl::FilterCloudByTimestamp(scan->surf_cloud, traj_max, scan_bef.surf_cloud,
                                scan_aft.surf_cloud);
    pcl::FilterCloudByTimestamp(scan->corner_cloud, traj_max,
                                scan_bef.corner_cloud, scan_aft.corner_cloud);
    //
    scan_bef.timestamp = scan->timestamp;
    scan_bef.max_timestamp = traj_max - 1e-9 * S_TO_NS;
    scan_aft.timestamp = traj_max;
    scan_aft.max_timestamp = scan->max_timestamp;

    //
    scan->timestamp = traj_max;
    // *scan.max_timestamp = ； //
    *scan->raw_cloud = *scan_aft.raw_cloud;
    *scan->surf_cloud = *scan_aft.surf_cloud;
    *scan->corner_cloud = *scan_aft.corner_cloud;

    *msgs.lidar_raw_cloud += (*scan_bef.raw_cloud);
    *msgs.lidar_surf_cloud += (*scan_bef.surf_cloud);
    *msgs.lidar_corner_cloud += (*scan_bef.corner_cloud);

    //
    if (msgs.scan_num == 0) {
      // first scan
      msgs.lidar_timestamp = scan_bef.timestamp;
      msgs.lidar_max_timestamp = scan_bef.max_timestamp;
    } else {
      msgs.lidar_timestamp = std::min(msgs.lidar_timestamp, scan_bef.timestamp);
      msgs.lidar_max_timestamp =
          std::max(msgs.lidar_max_timestamp, scan_bef.max_timestamp);
    }

    add_entire_scan = false;
  }

  //
  msgs.scan_num++;

  return add_entire_scan;
}

///
bool MsgManager::GetMsgs(NextMsgs &msgs, int64_t traj_last_max,
                         int64_t traj_max, int64_t start_time) {
  msgs.Clear();

  if (imu_buf_.empty() || lidar_buf_.empty()) {
    return false;
  }
  if (cur_imu_timestamp_ - start_time < traj_max) {
    return false;
  }

  /// 1
  //
  std::vector<int> unique_lidar_ids;
  for (const auto &data : lidar_buf_) {
    if (std::find(unique_lidar_ids.begin(), unique_lidar_ids.end(),
                  data.lidar_id) != unique_lidar_ids.end())
      continue;
    unique_lidar_ids.push_back(data.lidar_id);
  }
  if (unique_lidar_ids.size() != num_lidars_) {
    return false;
  }
  //
  for (auto t : lidar_max_timestamps_) {
    if (t < traj_max) {
      return false;
    }
  }

  /// 2
  for (auto it = lidar_buf_.begin(); it != lidar_buf_.end();) {
    if (it->timestamp >= traj_max) {
      ++it;
      continue;
    }
    bool add_entire_scan = AddToMsg(msgs, it, traj_max);
    if (add_entire_scan) {
      it = lidar_buf_.erase(it); //
    } else {
      ++it; //
    }
  }
  LOG(INFO) << "[msgs_scan_num] " << msgs.scan_num;

  /// TODO 4
  if (uwb_buf_.empty()) {
    msgs.if_have_uwb = false;
    LOG(INFO) << "uwb_buf_ is empty" << std::endl;
  } else {
    if (uwb_buf_.front().timestamp < traj_max) {
      LOG(INFO) << "timestamp >= traj_max" << std::endl;
      msgs.if_have_uwb = true;
      msgs.uwb_msg = uwb_buf_.front();
      uwb_buf_.pop_front();
      LOG(INFO) << "uwb_buf_ pop_front: " << uwb_buf_.size();
    } else {
      LOG(INFO) << "timestamp < traj_max" << std::endl;
      msgs.if_have_uwb = false;
    }
  }
  return true;
}

void MsgManager::IMUMsgHandle(const sensor_msgs::Imu::ConstPtr &imu_msg) {
  int64_t t_last = cur_imu_timestamp_;
  // cur_imu_timestamp_ = imu_msg->header.stamp.toSec() -
  // add_extra_timeoffset_s_;
  cur_imu_timestamp_ = imu_msg->header.stamp.toSec() * S_TO_NS;

  IMUData data;
  IMUMsgToIMUData(imu_msg, data);

  /// problem
  // data.timestamp -= add_extra_timeoffset_s_;

  // for trajectory_manager
  imu_buf_.emplace_back(data);
}

void MsgManager::VelodyneMsgHandle(
    const sensor_msgs::PointCloud2::ConstPtr &vlp16_msg, int lidar_id) {
  RTPointCloud::Ptr vlp_raw_cloud(new RTPointCloud);
  velodyne_feature_extraction_->ParsePointCloud(vlp16_msg, vlp_raw_cloud); //

  // transform the input cloud to Lidar0 frame
  bool is_ipnl = false; // TODO
  if (is_ipnl) {
    Eigen::Matrix4d T_IPNL;
    T_IPNL << 0, 1, 0, 0, //
        -1, 0, 0, 0,      //
        0, 0, 1, 0.28,    //
        0, 0, 0, 1;
    pcl::transformPointCloud(*vlp_raw_cloud, *vlp_raw_cloud, T_IPNL);
  }

  if (lidar_id != 0) {
    pcl::transformPointCloud(*vlp_raw_cloud, *vlp_raw_cloud,
                             T_LktoL0_vec_[lidar_id]);
  }
  //
  // velodyne_feature_extraction_->LidarHandler(vlp_raw_cloud);
  velodyne_feature_extraction_->velodyne_handler(vlp_raw_cloud);

  lidar_buf_.emplace_back();
  lidar_buf_.back().lidar_id = lidar_id;
  if (lidar_timestamp_end_) {
    lidar_buf_.back().timestamp =
        (vlp16_msg->header.stamp.toSec() - 0.1003) * S_TO_NS; // kaist、viral
  } else {
    lidar_buf_.back().timestamp =
        vlp16_msg->header.stamp.toSec() * S_TO_NS; // lvi、lio
  }
  lidar_buf_.back().raw_cloud = vlp_raw_cloud;
  lidar_buf_.back().surf_cloud =
      velodyne_feature_extraction_->GetSurfaceFeature();
  lidar_buf_.back().corner_cloud =
      velodyne_feature_extraction_->GetCornerFeature();
}

void MsgManager::VelodyneMsgHandleNoFeature(
    const sensor_msgs::PointCloud2::ConstPtr &vlp16_msg, int lidar_id) {
  RTPointCloud::Ptr vlp_raw_cloud(new RTPointCloud);
  velodyne_feature_extraction_->ParsePointCloudNoFeature(vlp16_msg,
                                                         vlp_raw_cloud); //

  // // transform the input cloud to Lidar0 frame
  // if (lidar_id != 0)
  //   pcl::transformPointCloud(*vlp_raw_cloud, *vlp_raw_cloud,
  //                            T_LktoL0_vec_[lidar_id]);

  // //
  // velodyne_feature_extraction_->LidarHandler(vlp_raw_cloud);

  lidar_buf_.emplace_back();
  lidar_buf_.back().lidar_id = lidar_id;
  if (lidar_timestamp_end_) {
    lidar_buf_.back().timestamp =
        (vlp16_msg->header.stamp.toSec() - 0.1003) * S_TO_NS; // kaist、viral
  } else {
    lidar_buf_.back().timestamp =
        vlp16_msg->header.stamp.toSec() * S_TO_NS; // lvi、lio
  }
  lidar_buf_.back().raw_cloud = vlp_raw_cloud;
  lidar_buf_.back().surf_cloud =
      velodyne_feature_extraction_->GetSurfaceFeature();
  lidar_buf_.back().corner_cloud =
      velodyne_feature_extraction_->GetCornerFeature();
}

void MsgManager::LivoxMsgHandle(
    const livox_ros_driver2::CustomMsg::ConstPtr &livox_msg, int lidar_id) {
  RTPointCloud::Ptr livox_raw_cloud(new RTPointCloud);
  //
  // livox_feature_extraction_->ParsePointCloud(livox_msg, livox_raw_cloud);
  // livox_feature_extraction_->ParsePointCloudNoFeature(livox_msg,
  // livox_raw_cloud);
  livox_feature_extraction_->ParsePointCloudR3LIVE(livox_msg, livox_raw_cloud);

  LiDARCloudData data;
  data.lidar_id = lidar_id;
  data.timestamp = livox_msg->header.stamp.toSec() * S_TO_NS;
  data.raw_cloud = livox_raw_cloud;
  data.surf_cloud = livox_feature_extraction_->GetSurfaceFeature();
  data.corner_cloud = livox_feature_extraction_->GetCornerFeature();
  // if (!data.raw_cloud->empty() && !data.surf_cloud->empty() &&
  //     !data.corner_cloud->empty()) {
  //   lidar_buf_.push_back(data);
  // } else {
  //   ROS_WARN("Livox cloud is empty");
  // }

  if (data.raw_cloud->size() > 3 && data.surf_cloud->size() > 3 &&
      data.corner_cloud->size() > 0) {
    lidar_buf_.push_back(data);
  } else {
    ROS_WARN("Livox cloud is empty: raw_cloud size: %d, surf_cloud size: %d, "
             "corner_cloud size: %d",
             data.raw_cloud->size(), data.surf_cloud->size(),
             data.corner_cloud->size());
  }

  if (lidar_id != 0) {
    pcl::transformPointCloud(*data.raw_cloud, *data.raw_cloud,
                             T_LktoL0_vec_[lidar_id]);
    pcl::transformPointCloud(*data.surf_cloud, *data.surf_cloud,
                             T_LktoL0_vec_[lidar_id]);
    pcl::transformPointCloud(*data.corner_cloud, *data.corner_cloud,
                             T_LktoL0_vec_[lidar_id]);
  }
}

void MsgManager::ImageMsgHandle(const sensor_msgs::ImageConstPtr &msg) {
  if (pub_img_.getNumSubscribers() != 0) {
    pub_img_.publish(msg);
  }

  cv_bridge::CvImagePtr cvImgPtr;
  cvImgPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  if (cvImgPtr->image.empty()) {
    std::cout << RED << "[ImageMsgHandle get an empty img]" << RESET
              << std::endl;
    return;
  }

  image_buf_.emplace_back();
  image_buf_.back().timestamp = msg->header.stamp.toSec() * S_TO_NS;
  image_buf_.back().image = cvImgPtr->image;
  nerf_time_.push_back(image_buf_.back().timestamp);

  if (image_buf_.back().image.cols == 640 ||
      image_buf_.back().image.cols == 1280) {
    cv::resize(image_buf_.back().image, image_buf_.back().image,
               cv::Size(640, 512), 0, 0, cv::INTER_LINEAR);
  }

  // // for tiers
  // if (image_buf_.back().image.cols == 1920)
  // {
  //   cv::resize(image_buf_.back().image, image_buf_.back().image,
  //   cv::Size(960, 540), 0, 0, cv::INTER_LINEAR);
  // }
}

void MsgManager::ImageMsgHandle(
    const sensor_msgs::CompressedImageConstPtr &msg) {
  if (pub_img_.getNumSubscribers() != 0) {
    cv_bridge::CvImagePtr cvImgPtr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    sensor_msgs::Image imgMsg = *(cvImgPtr->toImageMsg());
    imgMsg.header = msg->header; //
    pub_img_.publish(msg);
  }

  cv_bridge::CvImagePtr cvImgPtr;
  cvImgPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  if (cvImgPtr->image.empty()) {
    std::cout << RED << "[ImageMsgHandle get an empty img]" << RESET
              << std::endl;
    return;
  }

  image_buf_.emplace_back();
  image_buf_.back().timestamp = msg->header.stamp.toSec() * S_TO_NS;
  image_buf_.back().image = cvImgPtr->image;
  nerf_time_.push_back(image_buf_.back().timestamp);

  // std::cout << image_buf_.back().image.rows << " " <<
  // image_buf_.back().image.cols << std::endl;

  if (image_buf_.back().image.cols == 640 ||
      image_buf_.back().image.cols == 1280) {
    cv::resize(image_buf_.back().image, image_buf_.back().image,
               cv::Size(640, 512), 0, 0, cv::INTER_LINEAR);
  }

  // // for mars
  // if (image_buf_.back().image.cols == 2448)
  // {
  //   // cv::resize(image_buf_.back().image, image_buf_.back().image,
  //   cv::Size(1224, 1024), 0, 0, cv::INTER_LINEAR);
  //   cv::resize(image_buf_.back().image, image_buf_.back().image,
  //   cv::Size(612, 512), 0, 0, cv::INTER_LINEAR);
  // }
}

void MsgManager::GetUWBPosInit(UwbData &measurements) {
  Eigen::Matrix<float, 5, 1> matB0;
  matB0.fill(-1);
  LOG(INFO) << "matB0: " << matB0(0, 0) << " " << matB0(1, 0) << " "
            << matB0(2, 0);

  Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t1;
  R1 << 0.406165301800, -0.913799643517, 0.00000000000, 0.913799643517,
      0.406165301800, 0.000000000000, 0.000000000000, 0.000000000000,
      1.000000000000;
  t1 << -3.024960279465, 9.532732009888, -0.889378666879;
  Eigen::Isometry3d extrinsic_matrix = Eigen::Isometry3d::Identity();
  extrinsic_matrix.rotate(R1);
  extrinsic_matrix.pretranslate(t1);

  // reduce the init_position to all anchor positions
  for (auto &[anchor_id, anchor_pos] : anchor_id_positions) {
    LOG(INFO) << "old anchor_pos : " << anchor_pos.transpose();

    Eigen::Vector4d anchor_pos_4d =
        Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(), anchor_pos.z(), 1);
    Eigen::Vector4d anchor_pos_4d_transformed =
        extrinsic_matrix * anchor_pos_4d;
    anchor_pos = Eigen::Vector3d(anchor_pos_4d_transformed.x(),
                                 anchor_pos_4d_transformed.y(),
                                 anchor_pos_4d_transformed.z());
    LOG(INFO) << "anchor_id: " << anchor_id
              << " | new anchor_pos: " << anchor_pos.transpose();
    anchor_id_positions[anchor_id] = anchor_pos;
  }
}

void MsgManager::GetUWBPos(UwbData &measurements) {
  if (measurements.anchor_distances.size() >= 3) { // At least 3 anchors required
    Eigen::Vector3d uwb_position = Eigen::Vector3d::Zero();

    const size_t n = anchor_id_positions.size();
    Eigen::MatrixXd A(n - 1, 3);
    Eigen::VectorXd b(n - 1);

    // Get first anchor as reference
    auto it = anchor_id_positions.begin();
    const Eigen::Vector3d &anchor0 = it->second;
    const double distance0 = measurements.anchor_distances.at(it->first);
    ++it;

    // Build linear equation system
    for (size_t i = 0; i < n - 1; ++i, ++it) {
      const Eigen::Vector3d &anchor = it->second;
      const double distance = measurements.anchor_distances.at(it->first);

      A(i, 0) = 2 * (anchor.x() - anchor0.x());
      A(i, 1) = 2 * (anchor.y() - anchor0.y());
      A(i, 2) = 2 * (anchor.z() - anchor0.z());

      const double d0_sq = distance0 * distance0;
      const double di_sq = distance * distance;
      const double anchor0_sq = anchor0.squaredNorm();
      const double anchor_i_sq = anchor.squaredNorm();

      b(i) = d0_sq - di_sq - anchor0_sq + anchor_i_sq;
    }

    // Solve least squares: x = (A^T A)^-1 A^T b
    uwb_position = A.householderQr().solve(b);

    // save the uwb_position in txt
    std::ofstream outfile(
        "/home/mint/ws_uav_setup/src/LIUO-UWB/data/uwb_position.txt",
        std::ios::app);
    if (outfile.is_open()) {
      outfile << std::setprecision(19) << measurements.timestamp * 1e-9 << " "
              << uwb_position[0] << " " << uwb_position[1] << " " << 0 << " "
              << 0 << " " << 0 << " " << 0 << " " << 0 << std::endl;
      outfile.close();
    } else {
      std::cerr << "Unable to open file uwb_position.txt" << std::endl;
    }
  } else {
    LOG(INFO) << YELLOW
              << "\n⚠️  Insufficient UWB measurements for initialization: "
              << measurements.anchor_distances.size() << "/3" << RESET;
  }
}

void MsgManager::CalcExtrinsic() {
  Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t1;
  R1 << 0.406165301800, -0.913799643517, 0.00000000000, 0.913799643517,
      0.406165301800, 0.000000000000, 0.000000000000, 0.000000000000,
      1.000000000000;
  t1 << -3.024960279465, 9.532732009888, -0.889378666879;
  Eigen::Isometry3d extrinsic_matrix = Eigen::Isometry3d::Identity();
  extrinsic_matrix.rotate(R1);
  extrinsic_matrix.pretranslate(t1);
}

void MsgManager::UwbMsgHandle(
    const liuo::LinktrackTagframe0::ConstPtr &uwb_msg) {
  UwbData temp_uwb_data;
  temp_uwb_data.timestamp = uwb_msg->local_time * 1e3;
  // LOG(INFO) << "UWB timestamp: " << temp_uwb_data.timestamp;

  int anchor_num = 0;
  for (size_t i = 0; i < uwb_msg->dis_arr.size(); i++) {
    if (uwb_msg->dis_arr[i] > 1e-3) {
      temp_uwb_data.anchor_distances[i] = uwb_msg->dis_arr[i];
      anchor_num++;
    } else {
      LOG(INFO) << "Jump UWB anchor " << i
                << " distance: " << uwb_msg->dis_arr[i];
      continue;
    }
  }
  LOG(INFO) << "UWB anchor_num: " << anchor_num;
  temp_uwb_data.anchor_num = anchor_num;
  temp_uwb_data.tag_position = Eigen::Vector3d(
      uwb_msg->pos_3d[0], uwb_msg->pos_3d[1], uwb_msg->pos_3d[2]);

  // GetUWBPos(temp_uwb_data);
  if (!is_uwb_init_) {
    GetUWBPosInit(temp_uwb_data);
    is_uwb_init_ = true;
  }
  temp_uwb_data.anchor_positions = anchor_id_positions;

  uwb_buf_.emplace_back(temp_uwb_data);
}

void MsgManager::UwbMsgHandle(
    const liuo::LinktrackNodeframe3::ConstPtr &uwb_msg) {
  UwbData temp_uwb_data;
  int64_t time_offset = 0;
  temp_uwb_data.timestamp = uwb_msg->header.stamp.toNSec() + time_offset;

  int anchor_num = 0;
  for (auto node_ : uwb_msg->nodes) {
    if (node_.rx_rssi - node_.fp_rssi > 6) {
      continue;
    }
    temp_uwb_data.anchor_distances.emplace(node_.id, node_.dis);
    temp_uwb_data.fp_rssi = node_.fp_rssi;
    temp_uwb_data.rx_rssi = node_.rx_rssi;
    anchor_num++;
  }
  temp_uwb_data.anchor_num = anchor_num;
  temp_uwb_data.anchor_positions = anchor_id_positions;

  uwb_buf_.emplace_back(temp_uwb_data);
}

void MsgManager::InitSubscribers(ros::NodeHandle &nh) {
  if (!is_realtime_mode_) {
    LOG(WARNING) << "InitSubscribers called in bag mode, ignoring.";
    return;
  }

  LOG(INFO) << "Initializing ROS subscribers for realtime mode...";

  // Init IMU subscriber
  sub_imu_ = nh.subscribe(imu_topic_, 1000, &MsgManager::IMUMsgHandle, this);
  LOG(INFO) << "Subscribed to IMU topic: " << imu_topic_;

  // Init LiDAR subscriber
  subs_vlp16_.clear();
  subs_livox_.clear();
  for (int i = 0; i < num_lidars_; ++i) {
    if (lidar_types[i] == VLP) {
      // Velodyne subscriber
      auto sub = nh.subscribe<sensor_msgs::PointCloud2>(
          lidar_topics_[i], 10,
          boost::bind(&MsgManager::VelodyneMsgHandle, this, _1, i));
      subs_vlp16_.push_back(sub);
      LOG(INFO) << "Subscribed to Velodyne LiDAR topic: " << lidar_topics_[i];
    } else if (lidar_types[i] == LIVOX) {
      // Livox subscriber
      auto sub = nh.subscribe<livox_ros_driver2::CustomMsg>(
          lidar_topics_[i], 10,
          boost::bind(&MsgManager::LivoxMsgHandle, this, _1, i));
      subs_livox_.push_back(sub);
      LOG(INFO) << "Subscribed to Livox LiDAR topic: " << lidar_topics_[i];
    }
  }

  // Init UWB subscriber
  std::string uwb_topic_1 = "/nlink_linktrack_tagframe0";
  std::string uwb_topic_2 = "/nlink_linktrack_nodeframe3";
  if (uwb_topic_ == uwb_topic_1) {
    sub_uwb_ =
        nh.subscribe(uwb_topic_, 100,
                     static_cast<void (MsgManager::*)(
                         const liuo::LinktrackTagframe0::ConstPtr &)>(
                         &MsgManager::UwbMsgHandle),
                     this);
  } else if (uwb_topic_ == uwb_topic_2) {
    sub_uwb_ =
        nh.subscribe(uwb_topic_, 100,
                     static_cast<void (MsgManager::*)(
                         const liuo::LinktrackNodeframe3::ConstPtr &)>(
                         &MsgManager::UwbMsgHandle),
                     this);
  }
  LOG(INFO) << "Subscribed to UWB topic: " << uwb_topic_;

  // Init publisher
  pub_img_ = nh.advertise<sensor_msgs::Image>("/liuo/img", 10);

  LOG(INFO) << "All ROS subscribers initialized successfully.";
}

} // namespace liuo