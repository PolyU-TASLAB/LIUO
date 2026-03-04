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

#include <odom/odometry_manager.h>
#include <boost/filesystem.hpp>
#include <ros/package.h>
#include <ros/ros.h>

using namespace liuo;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "liuo");
  ros::NodeHandle nh("~");

  std::string config_path;
  nh.param<std::string>("config_path", config_path, "ct_odometry.yaml");
  ROS_INFO("Odometry load %s.", config_path.c_str());

  YAML::Node config_node = YAML::LoadFile(config_path);
  namespace fs = boost::filesystem;
  fs::path config_file_path(config_path);
  std::string config_dir = config_file_path.parent_path().string();

  fs::path pkg_root = config_file_path.parent_path().parent_path();

  std::string log_path_str = "logs";
  if (config_node["log_path"]) {
    log_path_str = config_node["log_path"].as<std::string>();
  }

  fs::path log_path(log_path_str);
  if (log_path.is_relative()) {
    log_path = pkg_root / log_path;
  }

  if (!fs::exists(log_path)) {
    fs::create_directories(log_path);
    ROS_INFO("Created log directory: %s", log_path.string().c_str());
  }

  FLAGS_log_dir = log_path.string();
  FLAGS_colorlogtostderr = true;
  std::cout << "\n=============== Start LIUO Odometry ===============" << std::endl;

  OdometryManager odom_manager(config_node, nh, config_path);

  // Select run mode based on config
  std::string run_mode = "bag"; // Default to bag mode
  if (config_node["run_mode"]) {
    run_mode = config_node["run_mode"].as<std::string>();
  }

  if (run_mode == "realtime") {
    std::cout << "\n Starting in realtime subscribe mode " << std::endl;
    odom_manager.RunInSubscribeMode();
  } else {
    std::cout << "\n Starting in bag file mode " << std::endl;
    odom_manager.RunBag();
  }

  double t_traj_max = odom_manager.SaveOdometry();
  std::cout << "\n=============== All Done. ===============\n\n";

  return 0;
}
