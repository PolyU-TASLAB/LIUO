/*
 * Map Saver - Memory-Efficient Map Saving Utilities
 * Memory-efficient map saving tool
 */

#pragma once

#include <fstream>
#include <lidar/lidar_handler.h>
#include <memory>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <utils/cloud_tool.h>

namespace liuo {

enum MapSaveMode {
  // Mode 1: Save keyframe downsampled cloud (most memory efficient)
  KEYFRAME_DOWNSAMPLED = 0,

  // Mode 2: Octree compressed map
  OCTREE_COMPRESSED = 1,

  // Mode 3: Tile-based saving + index
  TILE_BASED = 2,

  // Mode 4: Sparse feature map (corner + surface only)
  SPARSE_FEATURE = 3,

  // Mode 5: Full map (not recommended, high memory usage)
  FULL_MAP = 4
};

struct MapSaveConfig {
  MapSaveMode mode = KEYFRAME_DOWNSAMPLED;

  // Downsample parameters
  float voxel_leaf_size = 0.1; // 10cm voxel

  // Octree parameters
  float octree_resolution = 0.1; // 10cm resolution

  // Tile parameters
  float tile_size = 50.0; // 50m tile

  // Compression parameters
  bool use_binary_pcd = true;  // Use binary PCD format
  bool use_compression = true; // Use compression

  // Content selection
  bool save_corner_features = false; // Save corner features
  bool save_surface_features = true; // Save surface features
  bool save_full_cloud = false;      // Save full cloud

  // Distance filter
  bool enable_distance_filter = true;
  float max_distance = 100.0; // Max distance threshold (meters)

  MapSaveConfig() {}
};

class MapSaver {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<MapSaver> Ptr;

  MapSaver(const MapSaveConfig &config = MapSaveConfig()) : config_(config) {}

  bool SaveMap(const LidarHandler::Ptr &lidar_handler,
               const Trajectory::Ptr &trajectory, const std::string &save_path);

  /**
   * @brief Mode 1: Save keyframe downsampled cloud (Recommended, most memory efficient)
   * Memory usage: original map 5-10%
   */
  bool
  SaveKeyframeDownsampledMap(const std::map<int64_t, LiDARFeature> &keyframes,
                             const Trajectory::Ptr &trajectory,
                             const std::string &save_path);

  /**
   * @brief Mode 2: Save Octree compressed map
   * Memory usage: original map 10-20%
   */
  bool SaveOctreeCompressedMap(const std::map<int64_t, LiDARFeature> &keyframes,
                               const Trajectory::Ptr &trajectory,
                               const std::string &save_path);

  /**
   * @brief Mode 3: Tile-based saving
   * Memory usage: Controllable, suitable for large scale maps
   */
  bool SaveTileBasedMap(const std::map<int64_t, LiDARFeature> &keyframes,
                        const Trajectory::Ptr &trajectory,
                        const std::string &save_path);

  /**
   * @brief Mode 4: Save sparse feature map
   * Memory usage: original map 1-5%
   */
  bool SaveSparseFeatureMap(const std::map<int64_t, LiDARFeature> &keyframes,
                            const Trajectory::Ptr &trajectory,
                            const std::string &save_path);

  /**
   * @brief Save map metadata
   */
  void SaveMapMetadata(const std::string &save_path, size_t num_keyframes,
                       size_t total_points, const Eigen::Vector3d &map_center,
                       const Eigen::Vector3d &map_extent);

  /**
   * @brief Estimate map memory usage
   */
  size_t
  EstimateMapMemoryUsage(const std::map<int64_t, LiDARFeature> &keyframes);

  /**
   * @brief Print save statistics
   */
  void PrintSaveStatistics();

private:
  MapSaveConfig config_;

  // Statistics
  struct Statistics {
    size_t original_points = 0;
    size_t saved_points = 0;
    size_t num_keyframes = 0;
    double compression_ratio = 0.0;
    double save_time_ms = 0.0;
  } stats_;

  /**
   * @brief Downsample cloud
   */
  PosCloud::Ptr DownsampleCloud(const PosCloud::Ptr &cloud_in, float leaf_size);

  /**
   * @brief Distance filter
   */
  PosCloud::Ptr FilterByDistance(const PosCloud::Ptr &cloud_in,
                                 const Eigen::Vector3d &center, float max_dist);

  /**
   * @brief Merge multiple clouds
   */
  PosCloud::Ptr MergeClouds(const std::vector<PosCloud::Ptr> &clouds);

  /**
   * @brief Compute map bounds
   */
  void ComputeMapBounds(const PosCloud::Ptr &cloud, Eigen::Vector3d &min_bound,
                        Eigen::Vector3d &max_bound);

  /**
   * @brief Save cloud to file
   */
  bool SaveCloudToFile(const PosCloud::Ptr &cloud, const std::string &filename,
                       bool compressed = true);
};

} // namespace liuo
