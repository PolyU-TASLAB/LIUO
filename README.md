# LIUO

**[Outlier-Aware LiDAR/IMU/UWB Integration with NLOS Mitigation and Anchor Self-Calibration in Complex Scenarios](https://ieeexplore.ieee.org/document/11732054)**


LIUO is a ROS 1 research system for tightly coupled continuous-time fusion of LiDAR, raw IMU, and UWB ranging measurements. It represents the platform trajectory with non-uniform B-splines, jointly optimizes UWB anchor positions when self-calibration is enabled, and uses LiDAR-map geometry to mitigate UWB non-line-of-sight (NLOS) measurements.

> **Release status.** The public repository is a research preview. The configuration interface in this README is aligned with the latest manuscript revision and reviewer response. Before claiming complete result reproducibility, the frozen per-anchor UWB calibration arrays and the remaining protected core implementations must be included in a tagged release.

## Dependencies

The current package is configured for C++17 and ROS 1. Its CMake file requests:

- ROS Noetic components including `roscpp`, `rosbag`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `visualization_msgs`, `tf`, `cv_bridge`, and message generation;
- `livox_ros_driver2`;
- Eigen3;
- Ceres Solver;
- PCL 1.13.0;
- yaml-cpp 0.5 or newer;
- Boost filesystem, iostreams, program options, system, serialization, and thread;
- OpenMP when available.

The manuscript experiments use Ubuntu 20.04 and ROS Noetic. Because the exact dependency versions and protected implementation status may differ across branches, treat the commands below as the intended catkin workflow rather than a claim of a prebuilt binary release.

## Build

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/PolyU-TASLAB/LIUO.git
cd ..
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Install and build `livox_ros_driver2` in the same workspace, and ensure that Ceres, PCL, Eigen, yaml-cpp, and Boost are discoverable by CMake.

## Run a ROS bag

UrbanV2X example:

```bash
roslaunch liuo odometry.launch \
  config_path:=/config/ct_odometry_uwb_scipark.yaml \
  bag_path:=/absolute/path/to/urbanv2x.bag \
  run_mode:=bag
```

Urban-Building example:

```bash
roslaunch liuo odometry.launch \
  config_path:=/config/ct_odometry_uwb_building.yaml \
  bag_path:=/absolute/path/to/urban_building.bag \
  run_mode:=bag
```

Indoor examples use `/config/ct_odometry_uwb_vicon_los.yaml` and `/config/ct_odometry_uwb_vicon_nlos.yaml` respectively.

## Citation

```bibtex
@ARTICLE{11732054,
  author={Xiao, Naigui and Wen, Weisong and Bai, Lu and Wang, Xiangru and Hu, Jiahao and Zhang, Qi and Jiao, Jianhao and Lyu, Zhen and Huang, Feng},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Outlier-Aware LiDAR–IMU–UWB Integration with NLOS Mitigation and Anchor Self-Calibration in Complex Scenarios}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Laser radar;Distance measurement;Trajectory;Measurement;Timing;Splines (mathematics);Optimization;Calibration;Modeling;Joining processes;Sensor fusion;state estimation;autonomous navigation;continuous-time simultaneous localization and mapping;UWB anchor self-calibration;NLOS mitigation},
  doi={10.1109/TIM.2026.3742091}}
```

