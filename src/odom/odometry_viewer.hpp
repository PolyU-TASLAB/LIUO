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

#include <liuo/feature_cloud.h>
#include <liuo/imu_array.h>
#include <liuo/pose_array.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <imu/imu_state_estimator.h>
#include <lidar/lidar_feature.h>
#include <nav_msgs/Path.h>
#include <pcl-1.13/pcl/common/transforms.h>  //pcl::transformPointCloud
#include <pcl_conversions/pcl_conversions.h> //pcl::fromROSMsg
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <spline/trajectory.h>
#include <tf/transform_broadcaster.h>
#include <utils/parameter_struct.h>
#include <uwb/uwb_nlos_detector.h> // Include full definition to support inline functions
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace liuo {

enum SplineViewerType {
  Init = 0, //
  // Opt = 1,   //
  Loop = 2, //
};

class OdometryViewer {
public:
  // PublishViconData
  ros::Publisher pub_trajectory_raw_;
  ros::Publisher pub_trajectory_est_;

  // PublishIMUData
  ros::Publisher pub_imu_raw_array_;
  ros::Publisher pub_imu_est_array_;

  // PublishLoamCorrespondence
  ros::Publisher pub_target_cloud_;
  ros::Publisher pub_source_cloud_;
  ros::Publisher pub_map_corr_temp_cloud_;

  // PublishDenseCloud
  ros::Publisher pub_target_dense_cloud_;
  ros::Publisher pub_source_dense_cloud_;

  // PublishSplineTrajectory
  ros::Publisher pub_spline_trajectory_;
  ros::Publisher pub_spline_ctrl_;
  ros::Publisher pub_spline_ctrl_cloud_;
  ros::Publisher pub_spline_active_ctrl_cloud_;

  ros::Publisher pub_spline_marg_ctrl_cloud_;

  std::map<SplineViewerType, ros::Publisher> pub_spline_ctrl_type_;
  std::map<SplineViewerType, ros::Publisher> pub_spline_ctrl_cloud_type_;

  /// IMU
  ros::Publisher pub_latest_bias_vio_;
  ros::Publisher pub_latest_bias_lio_;
  // PublishIMUPropogation
  ros::Publisher pub_imu_propogation_trajectory_;

  // PublishImageLandmarks
  ros::Publisher pub_img_landmarks_;
  ros::Publisher pub_img_marg_cloud_;
  ros::Publisher pub_vio_keyframe_;
  ros::Publisher pub_image_track_;

  ros::Publisher pub_odom_gt_;

  // UWB
  ros::Publisher pub_uwb_anchors_;

  // UWB NLOS Detection Visualization
  ros::Publisher pub_nlos_obstacle_cloud_; // NLOS obstacle cloud
  ros::Publisher pub_nlos_local_cloud_;    // Local point cloud
  ros::Publisher pub_nlos_ray_path_;       // Ray path from Tag to Anchor

  // direct visual odometry
  ros::Publisher pub_track_img_;
  ros::Publisher pub_undistort_scan_in_cur_img_;
  ros::Publisher pub_old_and_new_added_points_in_cur_img_;
  ros::Publisher pub_img_aft_track_;
  ros::Publisher pub_sub_visual_map_;
  ros::Publisher pub_pg_cloud_;

  ros::Publisher pub_keyframe_image_;
  ros::Publisher pub_keyframe_pose_;
  ros::Publisher pub_keyframe_points_;

  // Not used
  ros::Publisher pub_icp_target_cloud_;
  ros::Publisher pub_icp_source_cloud_;
  ros::Publisher pub_icp_raw_source_cloud_;
  ros::Publisher pub_pose_graph_marker_;

  VPointCloud map_cloud_;

public:
  void SetPublisher(ros::NodeHandle &nh) {
    /// Vicon data
    pub_trajectory_raw_ = nh.advertise<liuo::pose_array>("/path_raw", 10);
    pub_trajectory_est_ = nh.advertise<liuo::pose_array>("/path_est", 10);
    /// IMU fitting results
    pub_imu_raw_array_ = nh.advertise<liuo::imu_array>("/imu_raw_array", 10);
    pub_imu_est_array_ = nh.advertise<liuo::imu_array>("/imu_est_array", 10);

    /// ===== LIO ===== ///
    // lidar matching results
    pub_target_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/target_fea", 10);
    pub_source_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/source_fea", 10);
    pub_map_corr_temp_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/liomap_corr_temp", 10);

    // dense feature cloud
    pub_target_dense_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/target_dense_cloud", 10);
    pub_source_dense_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/lio/source_dense_cloud", 10);

    /// spline trajectory
    pub_spline_trajectory_ =
        nh.advertise<nav_msgs::Path>("/spline/trajectory", 10);
    pub_spline_ctrl_ = nh.advertise<nav_msgs::Path>("/spline/ctrl_path", 10);
    pub_spline_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/ctrl_cloud", 10);
    pub_spline_active_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/active_ctrl_cloud", 10);
    pub_spline_marg_ctrl_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/marg_ctrl_cloud", 10);

    pub_spline_ctrl_type_[Init] =
        nh.advertise<nav_msgs::Path>("/spline/init_ctrl_path", 10);
    pub_spline_ctrl_cloud_type_[Init] =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/init_ctrl_cloud", 10);

    pub_spline_ctrl_type_[Loop] =
        nh.advertise<nav_msgs::Path>("/spline/loop_ctrl_path", 10);
    pub_spline_ctrl_cloud_type_[Loop] =
        nh.advertise<sensor_msgs::PointCloud2>("/spline/loop_ctrl_cloud", 10);

    pub_latest_bias_vio_ =
        nh.advertise<sensor_msgs::Imu>("/imu/latest_vio_bias", 10);
    pub_latest_bias_lio_ =
        nh.advertise<sensor_msgs::Imu>("/imu/latest_lio_bias", 10);
    /// imu state estimator trajectory
    pub_imu_propogation_trajectory_ =
        nh.advertise<nav_msgs::Path>("/imu/propogation_trajectory", 10);

    /// ===== VIO ===== ///
    pub_img_landmarks_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/landmarks", 10);
    pub_img_marg_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/marg_cloud", 10);
    pub_vio_keyframe_ =
        nh.advertise<sensor_msgs::PointCloud2>("/vio/keyframe", 10);
    pub_image_track_ = nh.advertise<sensor_msgs::Image>("/vio/track_image", 10);

    pub_odom_gt_ = nh.advertise<nav_msgs::Path>("/odom_gt", 10);

    pub_track_img_ = nh.advertise<sensor_msgs::Image>("/track_img", 100);
    pub_undistort_scan_in_cur_img_ =
        nh.advertise<sensor_msgs::Image>("/undistort_scan_in_cur_img", 100);
    pub_old_and_new_added_points_in_cur_img_ = nh.advertise<sensor_msgs::Image>(
        "/old_and_new_added_points_in_cur_img", 100);
    pub_img_aft_track_ =
        nh.advertise<sensor_msgs::Image>("/img_aft_track", 100);
    pub_sub_visual_map_ =
        nh.advertise<sensor_msgs::PointCloud2>("/sub_visual_map", 100);
    pub_pg_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/pg_global_cloud", 100);

    pub_keyframe_image_ =
        nh.advertise<sensor_msgs::Image>("/keyframe_image", 1000);
    pub_keyframe_pose_ =
        nh.advertise<geometry_msgs::PoseStamped>("/keyframe_pose", 1000);
    pub_keyframe_points_ =
        nh.advertise<sensor_msgs::PointCloud2>("/keyframe_points", 1000);

    pub_uwb_anchors_ =
        nh.advertise<visualization_msgs::MarkerArray>("/uwb_anchors", 1);

    /// ===== UWB NLOS Detection Visualization ===== ///
    pub_nlos_obstacle_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/uwb/nlos_obstacle_cloud", 10);
    pub_nlos_local_cloud_ =
        nh.advertise<sensor_msgs::PointCloud2>("/uwb/nlos_local_cloud", 10);
    pub_nlos_ray_path_ =
        nh.advertise<visualization_msgs::Marker>("/uwb/nlos_ray_path", 10);

    // std::cout << "[SetPublisher] init done.\n";
  }

  void PublishTrackImg(const sensor_msgs::ImagePtr &img) {
    if (pub_track_img_.getNumSubscribers() == 0) {
      return;
    }
    pub_track_img_.publish(img);
  }

  void PublishUndistortScanInCurImg(const sensor_msgs::ImagePtr &img) {
    if (pub_undistort_scan_in_cur_img_.getNumSubscribers() == 0) {
      return;
    }
    pub_undistort_scan_in_cur_img_.publish(img);
  }

  void PublishOldAndNewAddedPointsInCurImg(const sensor_msgs::ImagePtr &img) {
    if (pub_old_and_new_added_points_in_cur_img_.getNumSubscribers() == 0) {
      return;
    }
    pub_old_and_new_added_points_in_cur_img_.publish(img);
  }

  void PublishImgAftTrack(const sensor_msgs::ImagePtr &img) {
    if (pub_img_aft_track_.getNumSubscribers() == 0) {
      return;
    }
    pub_img_aft_track_.publish(img);
  }

  void PublishSubVisualMap(const VPointCloud &sub_visual_map_cloud) {
    if (pub_sub_visual_map_.getNumSubscribers() != 0 &&
        sub_visual_map_cloud.size() > 0) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(sub_visual_map_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "map";
      pub_sub_visual_map_.publish(cloud_msg);
    }
  }

  void PublishPgCloudDebug(const PosCloud &pg_global_cloud) {
    if (pub_pg_cloud_.getNumSubscribers() != 0 && pg_global_cloud.size() > 0) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(pg_global_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "map";
      pub_pg_cloud_.publish(cloud_msg);
    }
  }

  void PublishTF(Eigen::Quaterniond quat, Eigen::Vector3d pos,
                 std::string from_frame, std::string to_frame) {
    static tf::TransformBroadcaster tbr;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pos[0], pos[1], pos[2]));
    tf::Quaternion tf_q(quat.x(), quat.y(), quat.z(), quat.w());
    transform.setRotation(tf_q);
    tbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                           to_frame, from_frame));
  }

  void PublishIMUData(Trajectory::Ptr trajectory,
                      const Eigen::aligned_vector<IMUData> &imu_data,
                      const IMUBias &bias, const Eigen::Vector3d &gravity) {
    liuo::imu_array imu_array_raw;
    liuo::imu_array imu_array_est;

    for (auto const &v : imu_data) {
      geometry_msgs::Vector3 gyro, accel;
      tf::vectorEigenToMsg(v.gyro, gyro);
      tf::vectorEigenToMsg(v.accel, accel);
      imu_array_raw.timestamps.push_back(v.timestamp);
      imu_array_raw.angular_velocities.push_back(gyro);
      imu_array_raw.linear_accelerations.push_back(accel);

      Eigen::Vector3d w_b =
          trajectory->GetRotVelBody(v.timestamp) + bias.gyro_bias;
      Eigen::Vector3d a_w = trajectory->GetTransAccelWorld(v.timestamp);
      SE3d pose = trajectory->GetIMUPose(v.timestamp);
      Eigen::Vector3d a_b =
          pose.so3().inverse() * (a_w + gravity) + bias.accel_bias;

      /// some special cases for calibrating between Arpose and IMU
      if (v.gyro.norm() < 1e-10)
        w_b = Eigen::Vector3d::Zero();
      if (v.accel.norm() < 1e-10)
        a_b = Eigen::Vector3d::Zero();

      geometry_msgs::Vector3 gyro2, accel2;
      tf::vectorEigenToMsg(w_b, gyro2);
      tf::vectorEigenToMsg(a_b, accel2);
      imu_array_est.timestamps.push_back(v.timestamp);
      imu_array_est.angular_velocities.push_back(gyro2);
      imu_array_est.linear_accelerations.push_back(accel2);
    }
    imu_array_raw.header.stamp = ros::Time::now();
    imu_array_raw.header.frame_id = "/imu";

    imu_array_est.header = imu_array_raw.header;

    pub_imu_raw_array_.publish(imu_array_raw);
    pub_imu_est_array_.publish(imu_array_est);
  }

  void PublishViconData(Trajectory::Ptr trajectory,
                        Eigen::aligned_vector<PoseData> &vicon_data) {
    liuo::pose_array vicon_path_raw;
    liuo::pose_array vicon_path_est;

    for (auto const &v : vicon_data) {
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      // raw data
      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      vicon_path_raw.timestamps.push_back(v.timestamp);
      vicon_path_raw.positions.push_back(position);
      vicon_path_raw.orientations.push_back(orientation);

      // estiamted pose
      SE3d pose = trajectory->GetLidarPose(v.timestamp);
      tf::vectorEigenToMsg(pose.translation(), position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(), orientation);
      vicon_path_est.timestamps.push_back(v.timestamp);
      vicon_path_est.positions.push_back(position);
      vicon_path_est.orientations.push_back(orientation);
    }

    vicon_path_raw.header.frame_id = "map";
    vicon_path_raw.header.stamp = ros::Time::now();
    vicon_path_est.header = vicon_path_raw.header;

    pub_trajectory_raw_.publish(vicon_path_raw);
    pub_trajectory_est_.publish(vicon_path_est);
  }

  void PublishLoamMapCorrTemp(Trajectory::Ptr trajectory,
                              const VPointCloud &map_corr_temp_cloud) {
    if (pub_map_corr_temp_cloud_.getNumSubscribers() == 0)
      return;

    sensor_msgs::PointCloud2 target_msg;
    pcl::toROSMsg(map_corr_temp_cloud, target_msg);
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "map";

    pub_map_corr_temp_cloud_.publish(target_msg);
  }

  void PublishLoamCorrespondence(
      Trajectory::Ptr trajectory,
      const Eigen::aligned_vector<PointCorrespondence> &point_measurement) {
    if (pub_source_cloud_.getNumSubscribers() == 0 &&
        pub_target_cloud_.getNumSubscribers() == 0)
      return;

    VPointCloud target_cloud, source_cloud;
    for (const PointCorrespondence &cor : point_measurement) {
      if (cor.t_map < 0.0 || cor.t_point < 0.0) {
        LOG(INFO) << "[PublishLoamCorrespondence] skip " << cor.t_map << "; "
                  << cor.t_point;
        continue;
      }

      SE3d T_LktoG = trajectory->GetLidarPoseNURBS(cor.t_point);
      // SE3d T_MtoG = trajectory->GetLidarPose(cor.t_map);
      // Eigen::Vector3d p_inM = T_MtoG.inverse() * T_LktoG * cor.point;
      Eigen::Vector3d p_inM = T_LktoG * cor.point;

      VPoint p_target, p_source;
      Eigen::Vector3d p_intersect;
      if (cor.geo_type == GeometryType::Plane) {
        double dist = p_inM.dot(cor.geo_plane.head(3)) + cor.geo_plane[3];
        p_intersect = p_inM + dist * cor.geo_plane.head(3);

        p_target.intensity = 100;
        p_source.intensity = 100;
      } else {
        double t = (p_inM - cor.geo_point).dot(cor.geo_normal);
        p_intersect = cor.geo_point - t * cor.geo_normal;

        p_target.intensity = 50;
        p_source.intensity = 50;
      }

      p_target.x = p_intersect[0];
      p_target.y = p_intersect[1];
      p_target.z = p_intersect[2];
      target_cloud.push_back(p_target);
      p_source.x = p_inM[0];
      p_source.y = p_inM[1];
      p_source.z = p_inM[2];
      source_cloud.push_back(p_source);
    }

    sensor_msgs::PointCloud2 target_msg, source_msg;
    pcl::toROSMsg(target_cloud, target_msg);
    pcl::toROSMsg(source_cloud, source_msg);

    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "map";
    source_msg.header = target_msg.header;

    pub_target_cloud_.publish(target_msg);
    pub_source_cloud_.publish(source_msg);
  }

  void PublishSplineType(Trajectory::Ptr trajectory,
                         SplineViewerType viewer_type) {
    double opt_min_time = 0;
    int opt_fixed_idx = -1;
    if (viewer_type == Init) {
      opt_min_time = trajectory->opt_min_init_time_tmp;
      opt_fixed_idx = trajectory->opt_init_fixed_idx_tmp;
    } else if (viewer_type == Loop) {
      opt_min_time = trajectory->opt_min_loop_time_tmp;
      opt_fixed_idx = trajectory->opt_loop_fixed_idx_tmp;
    }
    size_t start_idx = trajectory->GetCtrlIndex(opt_min_time * S_TO_NS);
    if (opt_fixed_idx < 0)
      opt_fixed_idx = 0;

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    if (pub_spline_ctrl_type_[viewer_type].getNumSubscribers() != 0) {
      double min_time = trajectory->minTime(LiDARSensor);
      if (min_time < 0)
        min_time = 0;

      std::vector<geometry_msgs::PoseStamped> poses_ctrl;
      for (size_t i = start_idx; i < trajectory->numKnots(); ++i) {
        double t = min_time + i * trajectory->getDt();
        geometry_msgs::PoseStamped geo_ctrl;
        geo_ctrl.header.stamp = t_temp.fromSec(t);
        geo_ctrl.header.frame_id = "map";
        tf::pointEigenToMsg(trajectory->getKnotPos(i), geo_ctrl.pose.position);
        tf::quaternionEigenToMsg(trajectory->getKnotSO3(i).unit_quaternion(),
                                 geo_ctrl.pose.orientation);
        poses_ctrl.push_back(geo_ctrl);
      }

      nav_msgs::Path traj_ctrl;
      traj_ctrl.header.stamp = time_now;
      traj_ctrl.header.frame_id = "map";
      traj_ctrl.poses = poses_ctrl;
      pub_spline_ctrl_type_[viewer_type].publish(traj_ctrl);
    }

    if (pub_spline_ctrl_cloud_type_[viewer_type].getNumSubscribers() != 0) {
      VPointCloud ctrl_cloud;
      for (size_t i = start_idx; i < trajectory->numKnots(); ++i) {
        const Eigen::Vector3d &p = trajectory->getKnotPos(i);
        VPoint ctrl_p;
        ctrl_p.x = p[0];
        ctrl_p.y = p[1];
        ctrl_p.z = p[2];
        if ((int)i <= opt_fixed_idx)
          ctrl_p.intensity = 100;
        else
          ctrl_p.intensity = 200;

        ctrl_cloud.push_back(ctrl_p);
      }

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "map";
      pub_spline_ctrl_cloud_type_[viewer_type].publish(cloud_msg);
    }
  }

  void PublishMargCtrlCloud(const VPointCloud &marg_ctrl_cloud) {
    if (pub_spline_marg_ctrl_cloud_.getNumSubscribers() != 0 &&
        marg_ctrl_cloud.size() > 0) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(marg_ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "map";
      pub_spline_marg_ctrl_cloud_.publish(cloud_msg);
    }
  }

  void PublishSplineTrajectory(Trajectory::Ptr trajectory, double min_time,
                               double max_time, double dt) {
    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;

    ///
    if (pub_spline_trajectory_.getNumSubscribers() != 0) {
      std::vector<geometry_msgs::PoseStamped> poses_geo;
      // for (double t = min_time; t < max_time; t += dt)
      for (double t = min_time; t < max_time - 0.08; t += dt) {
        SE3d pose = trajectory->GetIMUPoseNURBS(t);
        geometry_msgs::PoseStamped poseIinG;
        poseIinG.header.stamp = t_temp.fromSec(t);
        poseIinG.header.frame_id = "map";
        tf::pointEigenToMsg(pose.translation(), poseIinG.pose.position);
        tf::quaternionEigenToMsg(pose.unit_quaternion(),
                                 poseIinG.pose.orientation);
        poses_geo.push_back(poseIinG);
      }

      nav_msgs::Path traj_path;
      traj_path.header.stamp = time_now;
      traj_path.header.frame_id = "map";
      traj_path.poses = poses_geo;
      pub_spline_trajectory_.publish(traj_path);
    }

    ///
    if (pub_spline_ctrl_.getNumSubscribers() != 0) {
      std::vector<geometry_msgs::PoseStamped> poses_ctrl;
      for (size_t i = 0; i < trajectory->numKnots(); ++i) {
        // double t = min_time + i * trajectory->getDt();
        double t = 0.0;
        geometry_msgs::PoseStamped geo_ctrl;
        geo_ctrl.header.stamp = t_temp.fromSec(t);
        geo_ctrl.header.frame_id = "map";
        tf::pointEigenToMsg(trajectory->getKnotPos(i), geo_ctrl.pose.position);
        tf::quaternionEigenToMsg(trajectory->getKnotSO3(i).unit_quaternion(),
                                 geo_ctrl.pose.orientation);
        poses_ctrl.push_back(geo_ctrl);
      }

      nav_msgs::Path traj_ctrl;
      traj_ctrl.header.stamp = time_now;
      traj_ctrl.header.frame_id = "map";
      traj_ctrl.poses = poses_ctrl;
      pub_spline_ctrl_.publish(traj_ctrl);
    }

    ///
    if (pub_spline_ctrl_cloud_.getNumSubscribers() != 0) {
      VPointCloud ctrl_cloud;
      for (size_t i = 0; i < trajectory->numKnots(); ++i) {
        const Eigen::Vector3d &p = trajectory->getKnotPos(i);
        VPoint ctrl_p;
        ctrl_p.x = p[0];
        ctrl_p.y = p[1];
        ctrl_p.z = p[2];
        // ctrl_p.intensity = 100;
        ctrl_p.intensity = trajectory->intensity_map[i];

        ctrl_cloud.push_back(ctrl_p);
      }

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(ctrl_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "map";
      pub_spline_ctrl_cloud_.publish(cloud_msg);
    }
  }

  void PublishLatestBias(const Eigen::Vector3d &gyro_bias,
                         const Eigen::Vector3d &accel_bias,
                         bool is_LIO = true) {
    if (pub_latest_bias_vio_.getNumSubscribers() != 0 ||
        pub_latest_bias_lio_.getNumSubscribers() != 0) {
      sensor_msgs::Imu imu_msg;
      imu_msg.header.stamp = ros::Time::now();
      imu_msg.angular_velocity.x = gyro_bias[0];
      imu_msg.angular_velocity.y = gyro_bias[1];
      imu_msg.angular_velocity.z = gyro_bias[2];
      imu_msg.linear_acceleration.x = accel_bias[0];
      imu_msg.linear_acceleration.y = accel_bias[1];
      imu_msg.linear_acceleration.z = accel_bias[2];
      if (is_LIO)
        pub_latest_bias_lio_.publish(imu_msg);
      else
        pub_latest_bias_vio_.publish(imu_msg);
    }
  }

  void PublishOdomGt(const std::vector<PoseData> &poses_gt, double max_time) {
    if (pub_odom_gt_.getNumSubscribers() == 0 || poses_gt.empty())
      return;

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    std::vector<geometry_msgs::PoseStamped> poses_geo;
    for (auto const &v : poses_gt) {
      if (v.timestamp > max_time)
        break;

      geometry_msgs::PoseStamped pose_gt;
      pose_gt.header.stamp = t_temp.fromSec(v.timestamp);
      pose_gt.header.frame_id = "map";
      tf::pointEigenToMsg(v.position, pose_gt.pose.position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(),
                               pose_gt.pose.orientation);
      poses_geo.push_back(pose_gt);
    }

    nav_msgs::Path traj_path;
    traj_path.header.stamp = time_now;
    traj_path.header.frame_id = "map";
    traj_path.poses = poses_geo;

    pub_odom_gt_.publish(traj_path);
  }

  geometry_msgs::PoseStamped fromIMUState(const IMUState &imu_state) const {
    ros::Time t_temp;
    geometry_msgs::PoseStamped poseIinG;
    poseIinG.header.stamp = t_temp.fromSec(imu_state.timestamp);
    poseIinG.header.frame_id = "map";
    tf::pointEigenToMsg(imu_state.p, poseIinG.pose.position);
    tf::quaternionEigenToMsg(imu_state.q, poseIinG.pose.orientation);

    return poseIinG;
  }

  void PublishIMUEstimator(const ImuStateEstimator::Ptr &imu_estimator) {
    /// Propogation
    std::vector<geometry_msgs::PoseStamped> poses_propogation;
    poses_propogation.push_back(
        fromIMUState(imu_estimator->GetPropagateStartState()));
    poses_propogation.push_back(fromIMUState(imu_estimator->GetIMUState()));

    nav_msgs::Path traj_propogation;
    traj_propogation.header.stamp = ros::Time::now();
    traj_propogation.header.frame_id = "map";
    traj_propogation.poses = poses_propogation;

    pub_imu_propogation_trajectory_.publish(traj_propogation);
  }

  void
  PublishDenseCloud(Trajectory::Ptr trajectory,
                    const LiDARFeature &target_feature,   // feature_map_ds_
                    const LiDARFeature &source_feature) { // feature_cur_ds_
    static int skip = 0;
    if (pub_target_dense_cloud_.getNumSubscribers() != 0 &&
        (++skip % 20 == 0)) {
      PosCloud target_cloud;
      Eigen::Matrix4d tranform_matrix = Eigen::Matrix4d::Identity();
      pcl::transformPointCloud(*target_feature.surface_features, target_cloud,
                               tranform_matrix);

      sensor_msgs::PointCloud2 target_msg;
      pcl::toROSMsg(target_cloud, target_msg);
      target_msg.header.stamp = ros::Time::now();
      target_msg.header.frame_id = "map";
      pub_target_dense_cloud_.publish(target_msg);
    }

    if (pub_source_dense_cloud_.getNumSubscribers() == 0) {
      return;
    }

    int start_idx = INT_MAX;
    {
      bool flag = false;
      int64_t time_ns = source_feature.timestamp;
      for (int i = 0; i < trajectory->knts.size() - 1; i++) {
        if (time_ns >= trajectory->knts[i] &&
            time_ns < trajectory->knts[i + 1]) {
          start_idx = i;
          flag = true;
          break;
        }
      }
      if (!flag)
        std::cout << "[PublishDenseCloud wrong]\n";
    }
    start_idx -= 2;
    if (start_idx < 0)
      start_idx = 0;

    if (pub_source_dense_cloud_.getNumSubscribers() != 0) {
      VPointCloud source_cloud;
      for (size_t i = 0; i < source_feature.surface_features->size(); i++) {
        int64_t point_timestamp =
            source_feature.surface_features->points[i].timestamp;
        if (point_timestamp >= trajectory->maxTimeNsNURBS())
          continue;
        SE3d point_pos =
            trajectory->GetLidarPoseNURBS(point_timestamp, start_idx);
        Eigen::Vector3d point_local(
            source_feature.surface_features->points[i].x,
            source_feature.surface_features->points[i].y,
            source_feature.surface_features->points[i].z);
        Eigen::Vector3d point_out =
            point_pos.so3() * point_local + point_pos.translation();
        VPoint p;
        p.x = point_out(0);
        p.y = point_out(1);
        p.z = point_out(2);
        p.intensity = source_feature.surface_features->points[i].intensity;

        // if (p.x <= 0) continue;
        // if (p.x <= 0.3 && p.z >= 0.01) continue;
        // if (p.x <= 1.0 && p.z >= 0.01) continue;  //good
        source_cloud.push_back(p);
      }

      map_cloud_ += source_cloud;
      // if (!map_cloud_.empty()) {
      //   pcl::io::savePCDFileBinary(
      //       "/home/mint/ws_uav_setup/src/LIUO-UWB/data/map_cloud.pcd",
      //       map_cloud_);
      // }

      sensor_msgs::PointCloud2 source_msg;
      pcl::toROSMsg(source_cloud, source_msg);
      source_msg.header.stamp = ros::Time::now();
      source_msg.header.frame_id = "map";
      pub_source_dense_cloud_.publish(source_msg);
    }
  }

  void ErrorStatistics(
      Trajectory::Ptr trajectory,
      const Eigen::aligned_vector<PointCorrespondence> &point_measurement,
      const Eigen::aligned_vector<IMUData> &imu_measurement,
      std::string note = " ") {
    double corner_error = 0;
    double surface_error = 0;
    int corner_cnt = 0;
    int surface_cnt = 0;
    for (const PointCorrespondence &cor : point_measurement) {
      if (cor.t_map < trajectory->minTime(LiDARSensor) ||
          cor.t_point < trajectory->minTime(LiDARSensor)) {
        LOG(INFO) << "[PublishLoamCorrespondence] skip " << cor.t_map << "; "
                  << cor.t_point;
        continue;
      }

      SE3d T_MtoG = trajectory->GetLidarPose(cor.t_map);
      SE3d T_LktoG = trajectory->GetLidarPose(cor.t_point);
      Eigen::Vector3d p_inM = T_MtoG.inverse() * T_LktoG * cor.point;

      if (cor.geo_type == GeometryType::Plane) {
        double dist = p_inM.dot(cor.geo_plane.head(3)) + cor.geo_plane[3];
        surface_error += fabs(dist);
        surface_cnt++;
      } else {
        double dist = (p_inM - cor.geo_point).cross(cor.geo_normal).norm();
        corner_error += dist;
        corner_cnt++;
      }
    }

    LOG(INFO) << "[Corner] Error : " << corner_error / corner_cnt;
    LOG(INFO) << "[Surface] Error : " << surface_error / surface_cnt;
  }

  // UWB anchor visualization functions
  void PublishUWBAnchorPositions(
      const std::map<int, Eigen::Vector3d> &anchor_positions,
      const std::map<int, Eigen::Vector3d> &initial_positions = {},
      const std::set<int> &visible_anchors = {}) {
    if (pub_uwb_anchors_.getNumSubscribers() == 0 || anchor_positions.empty()) {
      return;
    }

    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;

    // Create markers for current anchor positions
    for (const auto &anchor_pair : anchor_positions) {
      int anchor_id = anchor_pair.first;
      const Eigen::Vector3d &position = anchor_pair.second;

      // Create sphere marker for anchor position
      visualization_msgs::Marker anchor_marker;
      anchor_marker.header.frame_id = "map";
      anchor_marker.header.stamp = ros::Time::now();
      anchor_marker.ns = "uwb_anchors";
      anchor_marker.id = marker_id++;
      anchor_marker.type = visualization_msgs::Marker::SPHERE;
      anchor_marker.action = visualization_msgs::Marker::ADD;

      anchor_marker.pose.position.x = position.x();
      anchor_marker.pose.position.y = position.y();
      anchor_marker.pose.position.z = position.z();
      anchor_marker.pose.orientation.w = 1.0;

      anchor_marker.scale.x = 0.5;
      anchor_marker.scale.y = 0.5;
      anchor_marker.scale.z = 0.5;

      // Color coding: green for visible anchors, blue for others
      if (visible_anchors.find(anchor_id) != visible_anchors.end()) {
        anchor_marker.color.r = 0.0;
        anchor_marker.color.g = 1.0; // Green for visible anchors
        anchor_marker.color.b = 0.0;
      } else {
        anchor_marker.color.r = 0.0;
        anchor_marker.color.g = 0.0;
        anchor_marker.color.b = 1.0; // Blue for non-visible anchors
      }
      anchor_marker.color.a = 0.8;

      marker_array.markers.push_back(anchor_marker);

      // Create text marker for anchor ID
      visualization_msgs::Marker text_marker;
      text_marker.header.frame_id = "map";
      text_marker.header.stamp = ros::Time::now();
      text_marker.ns = "uwb_anchor_labels";
      text_marker.id = marker_id++;
      text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::Marker::ADD;

      text_marker.pose.position.x = position.x();
      text_marker.pose.position.y = position.y();
      text_marker.pose.position.z = position.z() + 0.6; // Offset above anchor
      text_marker.pose.orientation.w = 1.0;

      text_marker.scale.z = 0.3; // Text size
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      text_marker.color.a = 1.0;

      text_marker.text = "A" + std::to_string(anchor_id);
      marker_array.markers.push_back(text_marker);
    }

    // If initial positions are provided, show them as small red spheres
    if (!initial_positions.empty()) {
      for (const auto &initial_pair : initial_positions) {
        int anchor_id = initial_pair.first;
        const Eigen::Vector3d &init_position = initial_pair.second;

        // Skip if current position is the same as initial (no optimization
        // happened)
        auto current_it = anchor_positions.find(anchor_id);
        if (current_it != anchor_positions.end() &&
            (current_it->second - init_position).norm() < 0.01) {
          continue;
        }

        visualization_msgs::Marker initial_marker;
        initial_marker.header.frame_id = "map";
        initial_marker.header.stamp = ros::Time::now();
        initial_marker.ns = "uwb_anchors_initial";
        initial_marker.id = marker_id++;
        initial_marker.type = visualization_msgs::Marker::SPHERE;
        initial_marker.action = visualization_msgs::Marker::ADD;

        initial_marker.pose.position.x = init_position.x();
        initial_marker.pose.position.y = init_position.y();
        initial_marker.pose.position.z = init_position.z();
        initial_marker.pose.orientation.w = 1.0;

        initial_marker.scale.x = 0.2;
        initial_marker.scale.y = 0.2;
        initial_marker.scale.z = 0.2;

        initial_marker.color.r = 1.0; // Red for initial positions
        initial_marker.color.g = 0.0;
        initial_marker.color.b = 0.0;
        initial_marker.color.a = 0.6;

        marker_array.markers.push_back(initial_marker);

        // Draw line from initial to current position
        if (current_it != anchor_positions.end()) {
          visualization_msgs::Marker line_marker;
          line_marker.header.frame_id = "map";
          line_marker.header.stamp = ros::Time::now();
          line_marker.ns = "uwb_anchor_corrections";
          line_marker.id = marker_id++;
          line_marker.type = visualization_msgs::Marker::LINE_STRIP;
          line_marker.action = visualization_msgs::Marker::ADD;

          geometry_msgs::Point p1, p2;
          p1.x = init_position.x();
          p1.y = init_position.y();
          p1.z = init_position.z();

          p2.x = current_it->second.x();
          p2.y = current_it->second.y();
          p2.z = current_it->second.z();

          line_marker.points.push_back(p1);
          line_marker.points.push_back(p2);

          line_marker.scale.x = 0.05; // Line width
          line_marker.color.r = 1.0;
          line_marker.color.g = 0.5;
          line_marker.color.b = 0.0; // Orange line
          line_marker.color.a = 0.8;

          marker_array.markers.push_back(line_marker);
        }
      }
    }

    pub_uwb_anchors_.publish(marker_array);
  }

  void
  PublishUWBAnchorStats(const std::map<int, Eigen::Vector3d> &anchor_positions,
                        const std::map<int, Eigen::Vector3d> &initial_positions,
                        const std::set<int> &visible_anchors) {
    if (anchor_positions.empty())
      return;

    LOG(INFO) << "=== UWB Anchor Visualization Stats ===";
    LOG(INFO) << "Total anchors: " << anchor_positions.size();
    LOG(INFO) << "Visible anchors: " << visible_anchors.size();

    double total_correction = 0.0;
    int corrected_anchors = 0;

    for (const auto &anchor_pair : anchor_positions) {
      int anchor_id = anchor_pair.first;
      const Eigen::Vector3d &current_pos = anchor_pair.second;

      auto initial_it = initial_positions.find(anchor_id);
      if (initial_it != initial_positions.end()) {
        double correction = (current_pos - initial_it->second).norm();
        if (correction > 0.01) { // Only count significant corrections
          total_correction += correction;
          corrected_anchors++;
          LOG(INFO) << "Anchor " << anchor_id << " corrected by " << correction
                    << "m";
        }
      }

      std::string visibility =
          visible_anchors.find(anchor_id) != visible_anchors.end()
              ? "VISIBLE"
              : "NOT_VISIBLE";
      LOG(INFO) << "Anchor " << anchor_id << ": " << current_pos.transpose()
                << " [" << visibility << "]";
    }

    if (corrected_anchors > 0) {
      LOG(INFO) << "Average correction: "
                << total_correction / corrected_anchors << "m";
    }
    LOG(INFO) << "=====================================";
  }

  /**
   * @brief Publish UWB NLOS detection visualization data
   * @param nlos_detector NLOS detector pointer
   */
  void PublishNLOSVisualization(
      const std::shared_ptr<UWBNLOSDetector> &nlos_detector) {
    if (!nlos_detector) {
      ROS_WARN("[PublishNLOSVisualization] NLOS detector is not initialized");
      return;
    }

    // Publish NLOS obstacle cloud (red)
    if (pub_nlos_obstacle_cloud_.getNumSubscribers() > 0) {
      auto nlos_cloud = nlos_detector->GetNLOSObstacleCloud();
      if (nlos_cloud && !nlos_cloud->empty()) {
        LOG(INFO) << "🎯 PublishNLOSVisualization nlos_cloud->size(): "
                  << nlos_cloud->size();
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*nlos_cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = ros::Time::now();
        pub_nlos_obstacle_cloud_.publish(cloud_msg);
      }
    }

    // Publish local point cloud (yellow)
    if (pub_nlos_local_cloud_.getNumSubscribers() > 0) {
      auto local_cloud = nlos_detector->GetLocalCloud();
      if (local_cloud && !local_cloud->empty()) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*local_cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = ros::Time::now();
        pub_nlos_local_cloud_.publish(cloud_msg);
      }
    }

    // Publish ray path (green line)
    if (pub_nlos_ray_path_.getNumSubscribers() > 0) {
      auto ray_path = nlos_detector->GetLastRayPath();
      visualization_msgs::Marker line_marker;
      line_marker.header.frame_id = "map";
      line_marker.header.stamp = ros::Time::now();
      line_marker.ns = "nlos_ray";
      line_marker.id = 0;
      line_marker.type = visualization_msgs::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::Marker::ADD;
      line_marker.scale.x = 0.02; // Line width
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 1.0;

      geometry_msgs::Point p1, p2;
      p1.x = ray_path.first.x();
      p1.y = ray_path.first.y();
      p1.z = ray_path.first.z();
      p2.x = ray_path.second.x();
      p2.y = ray_path.second.y();
      p2.z = ray_path.second.z();

      line_marker.points.push_back(p1);
      line_marker.points.push_back(p2);

      pub_nlos_ray_path_.publish(line_marker);
    }
  }
};
} // namespace liuo
