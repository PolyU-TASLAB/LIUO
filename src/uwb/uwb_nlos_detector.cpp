/*
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

#include "uwb_nlos_detector.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include <pcl-1.13/pcl/io/pcd_io.h>
#include <pcl-1.13/pcl/io/ply_io.h>

#include <pcl-1.13/pcl/kdtree/impl/kdtree_flann.hpp>

// Explicitly instantiate KdTreeFLANN template
template class pcl::KdTreeFLANN<my_pcl::PointXYZT>;

namespace liuo {


UWBNLOSDetector::UWBNLOSDetector(const YAML::Node &node)
    : global_cloud_(new PosCloud), kdtree_(new pcl::KdTreeFLANN<PosPoint>()),
      kdtree_initialized_(false), enable_nlos_detection_(true),
      nlos_obstacle_cloud_(new PosCloud), local_cloud_cache_(new PosCloud),
      last_tag_pos_(Eigen::Vector3d::Zero()),
      last_anchor_pos_(Eigen::Vector3d::Zero()),
      accumulated_nlos_cloud_(new PosCloud),
      enable_timeseries_recording_(false), enable_timing_recording_(false) {

  voxel_resolution_ = node["nlos_voxel_resolution"].as<double>(0.2); // Default 20cm
  query_radius_ = node["nlos_query_radius"].as<double>(1.5);         // Default 1.5m
  delta_expansion_ = node["nlos_bbox_expansion"].as<double>(1.0);    // Default 1.0m
  sigma_los_squared_ =
      node["nlos_sigma_los_squared"].as<double>(0.01);          // Default 0.01
  alpha_penalty_ = node["nlos_alpha_penalty"].as<double>(0.05); // Default 0.05
  enable_nlos_detection_ = node["enable_nlos_detection"].as<bool>(true);
  enable_timeseries_recording_ =
      node["enable_timeseries_recording"].as<bool>(false);
  enable_timing_recording_ = node["enable_timing_recording"].as<bool>(false);

  LOG(INFO) << "[UWBNLOSDetector] Initialized with parameters:";
  LOG(INFO) << "  - Voxel resolution: " << voxel_resolution_ << " m";
  LOG(INFO) << "  - Query radius: " << query_radius_ << " m";
  LOG(INFO) << "  - AABB expansion: " << delta_expansion_ << " m";
  LOG(INFO) << "  - LOS variance (σ²_los): " << sigma_los_squared_;
  LOG(INFO) << "  - Penetration penalty (α): " << alpha_penalty_;
  LOG(INFO) << "  - NLOS detection enabled: " << enable_nlos_detection_;
  LOG(INFO) << "  - Time series recording enabled: "
            << enable_timeseries_recording_;
  LOG(INFO) << "  - Timing recording enabled: " << enable_timing_recording_;
}

UWBNLOSDetector::UWBNLOSDetector()
    : global_cloud_(new PosCloud), kdtree_(new pcl::KdTreeFLANN<PosPoint>()),
      kdtree_initialized_(false), voxel_resolution_(0.2), query_radius_(1.5),
      delta_expansion_(1.0), sigma_los_squared_(0.01), alpha_penalty_(0.05),
      enable_nlos_detection_(true), nlos_obstacle_cloud_(new PosCloud),
      local_cloud_cache_(new PosCloud), last_tag_pos_(Eigen::Vector3d::Zero()),
      last_anchor_pos_(Eigen::Vector3d::Zero()),
      accumulated_nlos_cloud_(new PosCloud),
      enable_timeseries_recording_(false), enable_timing_recording_(false) {
  LOG(INFO) << "[UWBNLOSDetector] Initialized with default parameters.";
}


void UWBNLOSDetector::UpdatePointCloudMap(const PosCloud::Ptr &cloud) {
  if (!cloud || cloud->empty()) {
    LOG(WARNING) << "[UWBNLOSDetector] Received empty point cloud map.";
    return;
  }

  global_cloud_ = cloud;
  kdtree_->setInputCloud(global_cloud_);
  kdtree_initialized_ = true;

  LOG(INFO) << "[UWBNLOSDetector] Updated point cloud map with "
            << cloud->size() << " points.";
}

NLOSDetectionResult
UWBNLOSDetector::DetectNLOS(const Eigen::Vector3d &tag_pos,
                            const Eigen::Vector3d &anchor_pos, int anchor_id,
                            int64_t timestamp) {
  LOG(INFO) << "[UWBNLOSDetector] DetectNLOS: tag_pos: " << tag_pos.transpose()
            << ", anchor_pos: " << anchor_pos.transpose()
            << ", anchor_id: " << anchor_id;


  auto t_total_start = std::chrono::high_resolution_clock::now();
  TimingRecord timing_record;
  if (enable_timing_recording_) {
    timing_record.timestamp = timestamp * 1e-9; // ns to s
    timing_record.anchor_id = anchor_id;
  }

  NLOSDetectionResult result;

  // TODO: Core implementation available upon acceptance.

  return result;
}

std::unordered_map<int, NLOSDetectionResult> UWBNLOSDetector::DetectNLOSBatch(
    const Eigen::Vector3d &tag_pos,
    const std::unordered_map<int, Eigen::Vector3d> &anchor_positions) {

  std::unordered_map<int, NLOSDetectionResult> results;

  for (const auto &[anchor_id, anchor_pos] : anchor_positions) {
    results[anchor_id] = DetectNLOS(tag_pos, anchor_pos, anchor_id);
  }

  return results;
}

double UWBNLOSDetector::ComputeDynamicWeight(int gpd) const {
  double variance = ComputeMeasurementVariance(gpd);
  // Weight = 1 / Variance (Definition of Information Matrix)
  return 1.0 / variance;
}

void UWBNLOSDetector::ResetStatistics() { statistics_ = Statistics(); }


PosCloud::Ptr
UWBNLOSDetector::ExtractLocalPointCloud(const Eigen::Vector3d &tag_pos,
                                        const Eigen::Vector3d &anchor_pos) {

  PosCloud::Ptr local_cloud(new PosCloud);

  // TODO: Core implementation available upon acceptance.

  return local_cloud;
}

Eigen::Vector3i
UWBNLOSDetector::PointToVoxelIndex(const Eigen::Vector3d &point) const {
  return Eigen::Vector3i(
      static_cast<int>(std::floor(point.x() / voxel_resolution_)),
      static_cast<int>(std::floor(point.y() / voxel_resolution_)),
      static_cast<int>(std::floor(point.z() / voxel_resolution_)));
}

std::unordered_set<Eigen::Vector3i, VoxelIndexHash, VoxelIndexEqual>
UWBNLOSDetector::BuildVoxelOccupancyMap(const PosCloud::Ptr &local_cloud) {
  std::unordered_set<Eigen::Vector3i, VoxelIndexHash, VoxelIndexEqual>
      occupied_voxels;

  for (const auto &point : local_cloud->points) {
    Eigen::Vector3d p(point.x, point.y, point.z);
    Eigen::Vector3i voxel_idx = PointToVoxelIndex(p);
    occupied_voxels.insert(voxel_idx);
  }

  return occupied_voxels;
}

std::vector<Eigen::Vector3i>
UWBNLOSDetector::Bresenham3D(const Eigen::Vector3i &start_voxel,
                             const Eigen::Vector3i &end_voxel) {

  std::vector<Eigen::Vector3i> path;

  // TODO: Core implementation available upon acceptance.

  return path;
}

int UWBNLOSDetector::ComputeGPD(
    const std::vector<Eigen::Vector3i> &path_voxels,
    const std::unordered_set<Eigen::Vector3i, VoxelIndexHash, VoxelIndexEqual>
        &occupied_voxels) const {

  int gpd = 0;

  // TODO: Core implementation available upon acceptance.

  return gpd;
}

double UWBNLOSDetector::ComputeMeasurementVariance(int gpd) const {
  // Linear penalty model: σ²_k = σ²_los + α·n_k
  // LOG(INFO) << "sigma_los_squared_: " << sigma_los_squared_
  //           << ", alpha_penalty_: " << alpha_penalty_ << ", gpd: " << gpd
  //           << ", measurement_variance: "
  //           << sigma_los_squared_ + alpha_penalty_ * gpd;
  return sigma_los_squared_ + alpha_penalty_ * gpd;
}

bool UWBNLOSDetector::SaveNLOSObstacleCloud(const std::string &filename,
                                            bool save_all_nlos) {
  PosCloud::Ptr cloud_to_save;

  if (save_all_nlos) {
    // Save all accumulated NLOS clouds
    cloud_to_save = accumulated_nlos_cloud_;
    if (!cloud_to_save || cloud_to_save->empty()) {
      LOG(WARNING)
          << "[UWBNLOSDetector] No accumulated NLOS obstacle points to "
             "save.";
      return false;
    }
    LOG(INFO)
        << "[UWBNLOSDetector] Saving accumulated NLOS obstacle cloud with "
        << cloud_to_save->size() << " points to: " << filename;
  } else {
    // Save only current accumulated NLOS cloud
    cloud_to_save = nlos_obstacle_cloud_;
    if (!cloud_to_save || cloud_to_save->empty()) {
      LOG(WARNING)
          << "[UWBNLOSDetector] No current NLOS obstacle points to save.";
      return false;
    }
    LOG(INFO) << "[UWBNLOSDetector] Saving current NLOS obstacle cloud with "
              << cloud_to_save->size() << " points to: " << filename;
  }

  try {
    // Determine file format and save
    std::string extension = filename.substr(filename.find_last_of(".") + 1);

    if (extension == "pcd") {
      // Save as PCD (Binary)
      if (pcl::io::savePCDFileBinary(filename, *cloud_to_save) == -1) {
        LOG(ERROR) << "[UWBNLOSDetector] Failed to save PCD file: " << filename;
        return false;
      }
    } else if (extension == "ply") {
      // Save as PLY
      if (pcl::io::savePLYFileBinary(filename, *cloud_to_save) == -1) {
        LOG(ERROR) << "[UWBNLOSDetector] Failed to save PLY file: " << filename;
        return false;
      }
    } else {
      // Default to PCD
      std::string pcd_filename = filename + ".pcd";
      if (pcl::io::savePCDFileBinary(pcd_filename, *cloud_to_save) == -1) {
        LOG(ERROR) << "[UWBNLOSDetector] Failed to save PCD file: "
                   << pcd_filename;
        return false;
      }
      LOG(INFO) << "[UWBNLOSDetector] Saved as PCD: " << pcd_filename;
    }

    LOG(INFO) << "[UWBNLOSDetector] Successfully saved NLOS obstacle cloud to: "
              << filename;
    return true;

  } catch (const std::exception &e) {
    LOG(ERROR) << "[UWBNLOSDetector] Exception while saving point cloud: "
               << e.what();
    return false;
  }
}

void UWBNLOSDetector::ClearAccumulatedNLOSCloud() {
  if (accumulated_nlos_cloud_) {
    size_t num_points = accumulated_nlos_cloud_->size();
    accumulated_nlos_cloud_->clear();
    LOG(INFO) << "[UWBNLOSDetector] Cleared accumulated NLOS cloud with "
              << num_points << " points.";
  }
}

bool UWBNLOSDetector::SaveTimeSeriesRecords(const std::string &filename) {
  std::lock_guard<std::mutex> lock(timeseries_mutex_);

  if (timeseries_records_.empty()) {
    LOG(WARNING) << "[UWBNLOSDetector] No time series records to save.";
    return false;
  }

  try {
    std::ofstream file(filename);
    if (!file.is_open()) {
      LOG(ERROR) << "[UWBNLOSDetector] Failed to open file: " << filename;
      return false;
    }

    // Write file header
    file << "# NLOS Detection Time Series Records\n";
    file << "# Format: Timestamp(s) AnchorID IsNLOS RobotPos(x,y,z) "
         << "AnchorPos(x,y,z) Distance(m) OcclusionPoint(x,y,z) GPD\n";
    file << "#" << std::string(120, '=') << "\n";

    // Statistics
    int los_count = 0, nlos_count = 0;
    std::map<int, std::pair<int, int>>
        anchor_stats; // <anchor_id, <total, nlos>>

    for (const auto &record : timeseries_records_) {
      if (record.is_nlos) {
        nlos_count++;
      } else {
        los_count++;
      }

      if (anchor_stats.find(record.anchor_id) == anchor_stats.end()) {
        anchor_stats[record.anchor_id] = {0, 0};
      }
      anchor_stats[record.anchor_id].first++;
      if (record.is_nlos) {
        anchor_stats[record.anchor_id].second++;
      }
    }

    file << "# Total measurements: " << timeseries_records_.size()
         << ", LOS: " << los_count << ", NLOS: " << nlos_count << "\n";
    file << "# Per-anchor NLOS statistics:\n";
    for (const auto &[anchor_id, stats] : anchor_stats) {
      double nlos_ratio =
          stats.first > 0 ? (100.0 * stats.second / stats.first) : 0.0;
      file << "#   Anchor " << anchor_id << ": " << stats.second << "/"
           << stats.first << " NLOS (" << std::fixed << std::setprecision(1)
           << nlos_ratio << "%)\n";
    }
    file << "#" << std::string(120, '=') << "\n\n";

    // Write data
    for (const auto &record : timeseries_records_) {
      file << std::fixed << std::setprecision(6) << record.timestamp << " "
           << record.anchor_id << " " << (record.is_nlos ? "NLOS" : "LOS ")
           << " "
           << "(" << std::setw(9) << record.robot_position.x() << ","
           << std::setw(9) << record.robot_position.y() << "," << std::setw(9)
           << record.robot_position.z() << ") "
           << "(" << std::setw(9) << record.anchor_position.x() << ","
           << std::setw(9) << record.anchor_position.y() << "," << std::setw(9)
           << record.anchor_position.z() << ") " << std::setw(9)
           << record.distance << " ";

      if (record.is_nlos && record.occlusion_point.norm() > 1e-6) {
        file << "(" << std::setw(9) << record.occlusion_point.x() << ","
             << std::setw(9) << record.occlusion_point.y() << ","
             << std::setw(9) << record.occlusion_point.z() << ")";
      } else {
        file << "(    None ,    None ,    None )";
      }

      file << " GPD=" << record.gpd << "\n";
    }

    file.close();

    LOG(INFO) << "[UWBNLOSDetector] Saved " << timeseries_records_.size()
              << " time series records to: " << filename;
    LOG(INFO) << "  - LOS: " << los_count << " ("
              << (100.0 * los_count / timeseries_records_.size()) << "%)";
    LOG(INFO) << "  - NLOS: " << nlos_count << " ("
              << (100.0 * nlos_count / timeseries_records_.size()) << "%)";

    return true;

  } catch (const std::exception &e) {
    LOG(ERROR) << "[UWBNLOSDetector] Exception while saving time series: "
               << e.what();
    return false;
  }
}

void UWBNLOSDetector::ClearTimeSeriesRecords() {
  std::lock_guard<std::mutex> lock(timeseries_mutex_);
  size_t num_records = timeseries_records_.size();
  timeseries_records_.clear();
  LOG(INFO) << "[UWBNLOSDetector] Cleared " << num_records
            << " time series records.";
}

bool UWBNLOSDetector::SaveTimingRecords(const std::string &filename) {
  std::lock_guard<std::mutex> lock(timing_mutex_);

  if (timing_records_.empty()) {
    LOG(WARNING) << "[UWBNLOSDetector] No timing records to save.";
    return false;
  }

  try {
    std::ofstream file(filename);
    if (!file.is_open()) {
      LOG(ERROR) << "[UWBNLOSDetector] Failed to open file: " << filename;
      return false;
    }

    // Write file header
    file << "# NLOS Detection Timing Statistics\n";
    file << "# Format: Timestamp(s) AnchorID IsNLOS TotalTime(ms) "
         << "ExtractLocal(ms) BuildVoxel(ms) RayTracing(ms) ComputeGPD(ms) "
         << "ExtractNLOS(ms) LocalPts VoxelNum PathVoxels GPD\n";
    file << "#" << std::string(140, '=') << "\n";

    // Statistics
    int los_count = 0, nlos_count = 0;
    double total_time_los = 0.0, total_time_nlos = 0.0;
    double avg_extract_local = 0.0, avg_build_voxel = 0.0;
    double avg_ray_tracing = 0.0, avg_compute_gpd = 0.0;
    double avg_extract_nlos = 0.0;
    std::map<int, std::pair<int, double>>
        anchor_stats; // <anchor_id, <count, total_time>>

    for (const auto &record : timing_records_) {
      if (record.is_nlos) {
        nlos_count++;
        total_time_nlos += record.total_time_ms;
      } else {
        los_count++;
        total_time_los += record.total_time_ms;
      }

      avg_extract_local += record.extract_local_cloud_ms;
      avg_build_voxel += record.build_voxel_map_ms;
      avg_ray_tracing += record.ray_tracing_ms;
      avg_compute_gpd += record.compute_gpd_ms;
      avg_extract_nlos += record.extract_nlos_cloud_ms;

      if (anchor_stats.find(record.anchor_id) == anchor_stats.end()) {
        anchor_stats[record.anchor_id] = {0, 0.0};
      }
      anchor_stats[record.anchor_id].first++;
      anchor_stats[record.anchor_id].second += record.total_time_ms;
    }

    int total = timing_records_.size();
    avg_extract_local /= total;
    avg_build_voxel /= total;
    avg_ray_tracing /= total;
    avg_compute_gpd /= total;
    avg_extract_nlos /= total;

    // Write statistics
    file << "# Total detections: " << total << ", LOS: " << los_count
         << ", NLOS: " << nlos_count << "\n";
    file << "# Average timing breakdown (ms):\n";
    file << "#   ExtractLocal: " << std::fixed << std::setprecision(3)
         << avg_extract_local << " | BuildVoxel: " << avg_build_voxel
         << " | RayTracing: " << avg_ray_tracing
         << " | ComputeGPD: " << avg_compute_gpd
         << " | ExtractNLOS: " << avg_extract_nlos << "\n";
    file << "# Average total time: LOS="
         << (los_count > 0 ? total_time_los / los_count : 0.0)
         << "ms, NLOS=" << (nlos_count > 0 ? total_time_nlos / nlos_count : 0.0)
         << "ms\n";
    file << "# Per-anchor timing statistics:\n";
    for (const auto &[anchor_id, stats] : anchor_stats) {
      double avg_time = stats.first > 0 ? stats.second / stats.first : 0.0;
      file << "#   Anchor " << anchor_id << ": " << stats.first
           << " detections, avg=" << std::fixed << std::setprecision(3)
           << avg_time << "ms\n";
    }
    file << "#" << std::string(140, '=') << "\n\n";

    // Write data
    for (const auto &record : timing_records_) {
      file << std::fixed << std::setprecision(6) << record.timestamp << " "
           << record.anchor_id << " " << (record.is_nlos ? "NLOS" : "LOS ")
           << " " << std::setprecision(4) << std::setw(8)
           << record.total_time_ms << " " << std::setw(8)
           << record.extract_local_cloud_ms << " " << std::setw(8)
           << record.build_voxel_map_ms << " " << std::setw(8)
           << record.ray_tracing_ms << " " << std::setw(8)
           << record.compute_gpd_ms << " " << std::setw(8)
           << record.extract_nlos_cloud_ms << " " << std::setw(7)
           << record.num_local_points << " " << std::setw(7)
           << record.num_voxels << " " << std::setw(7) << record.num_path_voxels
           << " " << std::setw(4) << record.gpd << "\n";
    }

    file.close();

    LOG(INFO) << "[UWBNLOSDetector] Saved " << timing_records_.size()
              << " timing records to: " << filename;
    LOG(INFO) << "  - LOS: " << los_count
              << " (avg: " << (los_count > 0 ? total_time_los / los_count : 0.0)
              << " ms)";
    LOG(INFO) << "  - NLOS: " << nlos_count << " (avg: "
              << (nlos_count > 0 ? total_time_nlos / nlos_count : 0.0)
              << " ms)";

    return true;

  } catch (const std::exception &e) {
    LOG(ERROR) << "[UWBNLOSDetector] Exception while saving timing records: "
               << e.what();
    return false;
  }
}

void UWBNLOSDetector::ClearTimingRecords() {
  std::lock_guard<std::mutex> lock(timing_mutex_);
  size_t num_records = timing_records_.size();
  timing_records_.clear();
  LOG(INFO) << "[UWBNLOSDetector] Cleared " << num_records
            << " timing records.";
}

} // namespace liuo
