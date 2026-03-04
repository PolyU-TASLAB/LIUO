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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>

#include <pcl-1.13/pcl/kdtree/kdtree_flann.h>
#include <pcl-1.13/pcl/point_cloud.h>
#include <pcl-1.13/pcl/point_types.h>

#include "../utils/mypcl_cloud_type.h"

namespace liuo {

/**
 * @brief Hash function for voxel index (for unordered_set)
 */
struct VoxelIndexHash {
  std::size_t operator()(const Eigen::Vector3i &v) const {
    // Create hash using bit shifting and XOR
    std::size_t h1 = std::hash<int>()(v.x());
    std::size_t h2 = std::hash<int>()(v.y());
    std::size_t h3 = std::hash<int>()(v.z());
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

/**
 * @brief Equality comparison function for voxel index
 */
struct VoxelIndexEqual {
  bool operator()(const Eigen::Vector3i &v1, const Eigen::Vector3i &v2) const {
    return v1.x() == v2.x() && v1.y() == v2.y() && v1.z() == v2.z();
  }
};

/**
 * @brief NLOS detection result
 */
struct NLOSDetectionResult {
  bool is_nlos;                    // Is NLOS?
  int geometric_penetration_depth; // Geometric Penetration Depth (GPD)
  double dynamic_weight;           // Dynamically calculated weight
  double measurement_variance;     // Measurement variance
  int num_occupied_voxels;         // Number of occupied voxels on path
  int total_path_voxels;           // Total voxels on path
  Eigen::Vector3d occlusion_point; // NLOS obstacle point position (centroid)

  NLOSDetectionResult()
      : is_nlos(false), geometric_penetration_depth(0), dynamic_weight(1.0),
        measurement_variance(0.0), num_occupied_voxels(0), total_path_voxels(0),
        occlusion_point(Eigen::Vector3d::Zero()) {}
};

/**
 * @brief NLOS detection time series record
 */
struct NLOSTimeSeriesRecord {
  double timestamp;                // Timestamp (seconds)
  int anchor_id;                   // Anchor ID
  bool is_nlos;                    // Is NLOS?
  Eigen::Vector3d robot_position;  // Robot position
  Eigen::Vector3d anchor_position; // Anchor position
  double distance;                 // Distance
  Eigen::Vector3d occlusion_point; // NLOS obstacle point position (valid only when NLOS)
  int gpd;                         // Geometric penetration depth

  NLOSTimeSeriesRecord()
      : timestamp(0.0), anchor_id(-1), is_nlos(false),
        robot_position(Eigen::Vector3d::Zero()),
        anchor_position(Eigen::Vector3d::Zero()), distance(0.0),
        occlusion_point(Eigen::Vector3d::Zero()), gpd(0) {}
};

/**
 * @brief NLOS detection timing statistics record
 */
struct TimingRecord {
  double timestamp;              // Timestamp (seconds)
  int anchor_id;                 // Anchor ID
  bool is_nlos;                  // Is NLOS?
  double total_time_ms;          // Total time (ms)
  double extract_local_cloud_ms; // Time to extract local cloud
  double build_voxel_map_ms;     // Time to build voxel occupancy map
  double ray_tracing_ms;         // Time for ray tracing
  double compute_gpd_ms;         // Time to compute GPD
  double extract_nlos_cloud_ms;  // Time to extract NLOS obstacle cloud
  int num_local_points;          // Number of local points
  int num_voxels;                // Number of voxels
  int num_path_voxels;           // Number of path voxels
  int gpd;                       // GPD value

  TimingRecord()
      : timestamp(0.0), anchor_id(-1), is_nlos(false), total_time_ms(0.0),
        extract_local_cloud_ms(0.0), build_voxel_map_ms(0.0),
        ray_tracing_ms(0.0), compute_gpd_ms(0.0), extract_nlos_cloud_ms(0.0),
        num_local_points(0), num_voxels(0), num_path_voxels(0), gpd(0) {}
};

/**
 * @brief UWB NLOS Detector
 *
 * NLOS identification and dynamic weight adjustment based on LiDAR geometric occlusion model
 *
 * Core Features:
 * 1. Efficient local point cloud extraction (KD-Tree radius search)
 * 2. Sparse voxel occupancy map construction (Hash Set)
 * 3. Bresenham 3D ray tracing
 * 4. Geometric Penetration Depth (GPD) computation
 * 5. GPD-based dynamic UWB weight adjustment
 *
 * Theoretical Basis:
 * - When UWB signals penetrate obstacles, measurement uncertainty is positively correlated with penetration path length (thickness)
 * - Linear penalty model: σ²_k = σ²_los + α·n_k
 * - Dynamic weight: w_U = (σ²_k)^(-1)
 */
class UWBNLOSDetector {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<UWBNLOSDetector> Ptr;

  /**
   * @brief Constructor
   * @param node YAML config node
   */
  explicit UWBNLOSDetector(const YAML::Node &node);

  /**
   * @brief Default constructor (use default parameters)
   */
  UWBNLOSDetector();

  /**
   * @brief Destructor
   */
  ~UWBNLOSDetector() = default;

  /**
   * @brief Update global point cloud map
   * @param cloud Global point cloud map (World frame)
   */
  void UpdatePointCloudMap(const PosCloud::Ptr &cloud);

  /**
   * @brief Detect if UWB measurement is NLOS and compute dynamic weight
   * @param tag_pos UWB Tag position (World frame)
   * @param anchor_pos UWB Anchor position (World frame)
   * @param anchor_id Anchor ID (for logging)
   * @param timestamp Timestamp (ns, for time series recording)
   * @return NLOS detection result
   */
  NLOSDetectionResult DetectNLOS(const Eigen::Vector3d &tag_pos,
                                 const Eigen::Vector3d &anchor_pos,
                                 int anchor_id = -1, int64_t timestamp = -1);

  /**
   * @brief Batch detect NLOS status for multiple anchors
   * @param tag_pos UWB Tag position
   * @param anchor_positions Anchor position map <anchor_id, position>
   * @return Detection results for each anchor <anchor_id, result>
   */
  std::unordered_map<int, NLOSDetectionResult> DetectNLOSBatch(
      const Eigen::Vector3d &tag_pos,
      const std::unordered_map<int, Eigen::Vector3d> &anchor_positions);

  /**
   * @brief Compute dynamic weight based on GPD
   * @param gpd Geometric Penetration Depth
   * @return Dynamic weight
   */
  double ComputeDynamicWeight(int gpd) const;

  /**
   * @brief Set voxel resolution
   */
  void SetVoxelResolution(double resolution) { voxel_resolution_ = resolution; }

  /**
   * @brief Set LOS reference variance
   */
  void SetLOSVariance(double variance) { sigma_los_squared_ = variance; }

  /**
   * @brief Set penetration penalty parameter
   */
  void SetPenetrationPenalty(double alpha) { alpha_penalty_ = alpha; }

  /**
   * @brief Set local query radius
   */
  void SetQueryRadius(double radius) { query_radius_ = radius; }

  /**
   * @brief Set bounding box expansion distance
   */
  void SetBoundingBoxExpansion(double delta) { delta_expansion_ = delta; }

  /**
   * @brief Enable/Disable NLOS detection
   */
  void EnableNLOSDetection(bool enable) { enable_nlos_detection_ = enable; }

  /**
   * @brief Get if NLOS detection is enabled
   */
  bool IsNLOSDetectionEnabled() const { return enable_nlos_detection_; }

  /**
   * @brief Get statistics
   */
  struct Statistics {
    int total_detections = 0;
    int nlos_count = 0;
    int los_count = 0;
    double avg_gpd = 0.0;
    double avg_weight_nlos = 0.0;
    double avg_weight_los = 0.0;
  };

  Statistics GetStatistics() const { return statistics_; }
  void ResetStatistics();

  /**
   * @brief Get lately detected NLOS obstacle cloud (for visualization)
   * @return NLOS obstacle cloud
   */
  PosCloud::Ptr GetNLOSObstacleCloud() const { return nlos_obstacle_cloud_; }

  /**
   * @brief Get latest ray path (Tag to Anchor line, for visualization)
   */
  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetLastRayPath() const {
    return std::make_pair(last_tag_pos_, last_anchor_pos_);
  }

  /**
   * @brief Get local cloud (for visualization)
   */
  PosCloud::Ptr GetLocalCloud() const { return local_cloud_cache_; }

  /**
   * @brief Save accumulated NLOS obstacle cloud to file
   * @param filename Output filename (supports .pcd or .ply)
   * @param save_all_nlos
   * If true, save all accumulated NLOS clouds; if false, save only current detection
   * @return True if successful
   */
  bool SaveNLOSObstacleCloud(const std::string &filename,
                             bool save_all_nlos = true);

  /**
   * @brief Clear accumulated NLOS obstacle cloud
   */
  void ClearAccumulatedNLOSCloud();

  /**
   * @brief Get accumulated NLOS obstacle cloud (for visualization)
   * @return Accumulated NLOS obstacle cloud
   */
  PosCloud::Ptr GetAccumulatedNLOSCloud() const {
    return accumulated_nlos_cloud_;
  }

  /**
   * @brief Get size of accumulated NLOS cloud
   */
  size_t GetAccumulatedNLOSCloudSize() const {
    return accumulated_nlos_cloud_ ? accumulated_nlos_cloud_->size() : 0;
  }

  /**
   * @brief Enable/Disable time series recording
   * @param enable Enable recording
   */
  void EnableTimeSeriesRecording(bool enable) {
    enable_timeseries_recording_ = enable;
  }

  /**
   * @brief Save time series records to file
   * @param filename Output filename (TXT format)
   * @return True if successful
   */
  bool SaveTimeSeriesRecords(const std::string &filename);

  /**
   * @brief Clear time series records
   */
  void ClearTimeSeriesRecords();

  /**
   * @brief Get number of time series records
   */
  size_t GetTimeSeriesRecordCount() const { return timeseries_records_.size(); }

  /**
   * @brief Enable/Disable timing statistics recording
   * @param enable Enable recording
   */
  void EnableTimingRecording(bool enable) { enable_timing_recording_ = enable; }

  /**
   * @brief Save timing records to file
   * @param filename Output filename (TXT format)
   * @return True if successful
   */
  bool SaveTimingRecords(const std::string &filename);

  /**
   * @brief Clear timing records
   */
  void ClearTimingRecords();

  /**
   * @brief Get number of timing records
   */
  size_t GetTimingRecordCount() const { return timing_records_.size(); }

private:
  /**
   * @brief Extract local point cloud around path
   * @param tag_pos Tag position
   * @param anchor_pos Anchor position
   * @return Local point cloud
   */
  PosCloud::Ptr ExtractLocalPointCloud(const Eigen::Vector3d &tag_pos,
                                       const Eigen::Vector3d &anchor_pos);

  /**
   * @brief Convert 3D point to voxel index
   * @param point 3D point
   * @return Voxel index
   */
  Eigen::Vector3i PointToVoxelIndex(const Eigen::Vector3d &point) const;

  /**
   * @brief Build local voxel occupancy map
   * @param local_cloud Local point cloud
   * @return Voxel occupancy set (Hash Set)
   */
  std::unordered_set<Eigen::Vector3i, VoxelIndexHash, VoxelIndexEqual>
  BuildVoxelOccupancyMap(const PosCloud::Ptr &local_cloud);

  /**
   * @brief Bresenham 3D ray tracing algorithm
   * @param start_voxel Start voxel index
   * @param end_voxel End voxel index
   * @return Voxel indices on the path
   */
  std::vector<Eigen::Vector3i> Bresenham3D(const Eigen::Vector3i &start_voxel,
                                           const Eigen::Vector3i &end_voxel);

  /**
   * @brief Compute Geometric Penetration Depth (GPD)
   * @param path_voxels Voxel indices on path
   * @param occupied_voxels Occupied voxel set
   * @return GPD value (number of occupied voxels)
   */
  int ComputeGPD(
      const std::vector<Eigen::Vector3i> &path_voxels,
      const std::unordered_set<Eigen::Vector3i, VoxelIndexHash, VoxelIndexEqual>
          &occupied_voxels) const;

  /**
   * @brief Compute measurement variance based on GPD
   * @param gpd Geometric Penetration Depth
   * @return Measurement variance σ²_k
   */
  double ComputeMeasurementVariance(int gpd) const;

  /**
   * @brief Check if map is initialized
   */
  bool IsMapInitialized() const {
    // std::cout << "🎯 IsMapInitialized global_cloud_->size(): "
    //           << global_cloud_->size() << std::endl;
    // std::cout << "🎯 IsMapInitialized kdtree_initialized_: "
    //           << kdtree_initialized_ << std::endl;
    return global_cloud_ && !global_cloud_->empty() && kdtree_initialized_;
  }

private:
  // Global Point Cloud Map
  PosCloud::Ptr global_cloud_;             // Global point cloud map
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_; // KD-Tree index
  bool kdtree_initialized_;                // Is KD-Tree initialized

  // Algorithm Parameters
  double voxel_resolution_; // Voxel resolution r_v (m)
  double query_radius_;     // Search radius R_q (m)
  double delta_expansion_;  // AABB expansion distance δ_R (m)

  // Dynamic Weight Parameters
  double sigma_los_squared_; // LOS reference variance σ²_los
  double alpha_penalty_;     // Penetration penalty parameter α

  // Control Switches
  bool enable_nlos_detection_; // Enable NLOS detection

  // Statistics
  mutable Statistics statistics_;

  // Visualization Cache
  mutable PosCloud::Ptr nlos_obstacle_cloud_; // Recently detected NLOS obstacle cloud
  mutable PosCloud::Ptr local_cloud_cache_;   // Recently extracted local cloud
  mutable Eigen::Vector3d last_tag_pos_;      // Recent Tag position
  mutable Eigen::Vector3d last_anchor_pos_;   // Recent Anchor position

  // NLOS Cloud Accumulation
  mutable PosCloud::Ptr
      accumulated_nlos_cloud_; // Accumulated NLOS obstacle cloud (for saving)

  // Time Series Recording
  bool enable_timeseries_recording_;                     // Enable time series recording
  std::vector<NLOSTimeSeriesRecord> timeseries_records_; // Time series records
  mutable std::mutex timeseries_mutex_; // Mutex for time series

  // Timing Statistics
  bool enable_timing_recording_;             // Enable timing statistics
  std::vector<TimingRecord> timing_records_; // Timing records
  mutable std::mutex timing_mutex_;          // Mutex for timing
};

} // namespace liuo
