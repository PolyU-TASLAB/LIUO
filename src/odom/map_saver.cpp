/*
 * Map Saver Implementation
 */

#include "map_saver.h"
#include <glog/logging.h>
#include <iomanip>
#include <pcl/common/transforms.h>
#include <utils/tic_toc.h>

namespace liuo {

bool MapSaver::SaveMap(const LidarHandler::Ptr &lidar_handler,
                       const Trajectory::Ptr &trajectory,
                       const std::string &save_path) {
  LOG(INFO) << "\n========== Start Saving Map ==========";
  LOG(INFO) << "Save Path: " << save_path;
  LOG(INFO) << "Save Mode: " << config_.mode;

  TicToc timer;
  stats_ = Statistics();


  std::map<int64_t, LiDARFeature> keyframes_container;
  auto cloud_key_pos = lidar_handler->GetCloudKeyPos3D();

  if (!cloud_key_pos || cloud_key_pos->empty()) {
    LOG(WARNING) << "Map is empty, cannot save!";
    return false;
  }

  LOG(INFO) << "Number of Keyframes: " << cloud_key_pos->size();

  bool success = false;
  switch (config_.mode) {
  case KEYFRAME_DOWNSAMPLED:
    LOG(INFO) << "Using Mode: Keyframe Downsampled (Recommended)";
    // Need to access local_feature_container_all_ds_
    LOG(WARNING) << "Need to add access interface in LidarHandler";
    break;

  case OCTREE_COMPRESSED:
    LOG(INFO) << "Using Mode: Octree Compressed";
    // success = SaveOctreeCompressedMap(keyframes_container, trajectory,
    // save_path);
    break;

  case TILE_BASED:
    LOG(INFO) << "Using Mode: Tile Based";
    // success = SaveTileBasedMap(keyframes_container, trajectory, save_path);
    break;

  case SPARSE_FEATURE:
    LOG(INFO) << "Using Mode: Sparse Feature";
    // success = SaveSparseFeatureMap(keyframes_container, trajectory,
    // save_path);
    break;

  case FULL_MAP:
    LOG(WARNING) << "Using Mode: Full Map (Not Recommended, High Memory Usage)";
    break;

  default:
    LOG(ERROR) << "Unknown Save Mode!";
    return false;
  }

  stats_.save_time_ms = timer.toc();
  PrintSaveStatistics();

  LOG(INFO) << "========== Map Saving Completed ==========\n";
  return success;
}

bool MapSaver::SaveKeyframeDownsampledMap(
    const std::map<int64_t, LiDARFeature> &keyframes,
    const Trajectory::Ptr &trajectory, const std::string &save_path) {

  LOG(INFO) << "[SaveKeyframeDownsampledMap] Start processing...";

  PosCloud::Ptr global_map(new PosCloud);
  stats_.num_keyframes = keyframes.size();

  int processed_kf = 0;
  for (const auto &kf_pair : keyframes) {
    int64_t timestamp = kf_pair.first;
    const LiDARFeature &feature = kf_pair.second;

    // Select cloud type to save
    PosCloud::Ptr local_cloud(new PosCloud);

    if (config_.save_full_cloud && !feature.full_cloud->empty()) {
      *local_cloud += *feature.full_cloud;
    }
    if (config_.save_surface_features && !feature.surface_features->empty()) {
      *local_cloud += *feature.surface_features;
    }
    if (config_.save_corner_features && !feature.corner_features->empty()) {
      *local_cloud += *feature.corner_features;
    }

    if (local_cloud->empty())
      continue;

    stats_.original_points += local_cloud->size();

    // Downsampling
    PosCloud::Ptr downsampled =
        DownsampleCloud(local_cloud, config_.voxel_leaf_size);

    // Transform to global frame
    SE3d pose = trajectory->GetLidarPoseNURBS(timestamp);
    PosCloud::Ptr global_cloud(new PosCloud);
    pcl::transformPointCloud(*downsampled, *global_cloud, pose.matrix());

    if (config_.enable_distance_filter) {
      Eigen::Vector3d map_center = pose.translation();
      global_cloud =
          FilterByDistance(global_cloud, map_center, config_.max_distance);
    }

    *global_map += *global_cloud;

    processed_kf++;
    if (processed_kf % 10 == 0) {
      LOG(INFO) << "Processed Keyframes: " << processed_kf << "/" << keyframes.size();
    }
  }

  LOG(INFO) << "Executing final downsample...";
  PosCloud::Ptr final_map =
      DownsampleCloud(global_map, config_.voxel_leaf_size);
  stats_.saved_points = final_map->size();
  stats_.compression_ratio =
      (double)stats_.original_points / stats_.saved_points;

  std::string filename = save_path + "_keyframe_ds.pcd";
  bool success = SaveCloudToFile(final_map, filename, config_.use_compression);

  if (success) {
    Eigen::Vector3d min_bound, max_bound;
    ComputeMapBounds(final_map, min_bound, max_bound);
    Eigen::Vector3d map_center = (min_bound + max_bound) / 2.0;
    Eigen::Vector3d map_extent = max_bound - min_bound;
    SaveMapMetadata(save_path + "_metadata.txt", keyframes.size(),
                    final_map->size(), map_center, map_extent);
  }

  return success;
}

bool MapSaver::SaveOctreeCompressedMap(
    const std::map<int64_t, LiDARFeature> &keyframes,
    const Trajectory::Ptr &trajectory, const std::string &save_path) {

  LOG(INFO) << "[SaveOctreeCompressedMap] Start processing...";

  // Merge all keyframes to global frame first
  PosCloud::Ptr global_map(new PosCloud);

  for (const auto &kf_pair : keyframes) {
    int64_t timestamp = kf_pair.first;
    const LiDARFeature &feature = kf_pair.second;

    PosCloud::Ptr local_cloud(new PosCloud);
    if (config_.save_surface_features && !feature.surface_features->empty()) {
      *local_cloud += *feature.surface_features;
    }
    if (local_cloud->empty())
      continue;

    SE3d pose = trajectory->GetLidarPoseNURBS(timestamp);
    PosCloud::Ptr global_cloud(new PosCloud);
    pcl::transformPointCloud(*local_cloud, *global_cloud, pose.matrix());

    *global_map += *global_cloud;
  }

  stats_.original_points = global_map->size();

  // Use voxel grid downsample (Using project custom VoxelFilter)
  PosCloud::Ptr compressed_map(new PosCloud);
  VoxelFilter<PosPoint> voxel_filter;
  voxel_filter.SetResolution(config_.octree_resolution);
  voxel_filter.SetInputCloud(global_map);
  voxel_filter.Filter(compressed_map);

  // Or use octree method (commented out alternative)

  stats_.saved_points = compressed_map->size();
  stats_.compression_ratio =
      (double)stats_.original_points / stats_.saved_points;

  std::string filename = save_path + "_octree.pcd";
  return SaveCloudToFile(compressed_map, filename, config_.use_compression);
}

bool MapSaver::SaveTileBasedMap(
    const std::map<int64_t, LiDARFeature> &keyframes,
    const Trajectory::Ptr &trajectory, const std::string &save_path) {

  LOG(INFO) << "[SaveTileBasedMap] Start tile-based saving...";

  Eigen::Vector3d min_bound(1e6, 1e6, 1e6);
  Eigen::Vector3d max_bound(-1e6, -1e6, -1e6);

  for (const auto &kf_pair : keyframes) {
    SE3d pose = trajectory->GetLidarPoseNURBS(kf_pair.first);
    Eigen::Vector3d pos = pose.translation();

    min_bound = min_bound.cwiseMin(pos);
    max_bound = max_bound.cwiseMax(pos);
  }

  LOG(INFO) << "Map Extent: [" << min_bound.transpose() << "] ~ ["
            << max_bound.transpose() << "]";

  int nx = std::ceil((max_bound.x() - min_bound.x()) / config_.tile_size) + 1;
  int ny = std::ceil((max_bound.y() - min_bound.y()) / config_.tile_size) + 1;

  LOG(INFO) << "Tile Grid: " << nx << " x " << ny << " (Total " << nx * ny
            << " tiles)";

  std::map<std::pair<int, int>, PosCloud::Ptr> tile_clouds;

  for (const auto &kf_pair : keyframes) {
    int64_t timestamp = kf_pair.first;
    const LiDARFeature &feature = kf_pair.second;
    SE3d pose = trajectory->GetLidarPoseNURBS(timestamp);

    PosCloud::Ptr local_cloud(new PosCloud);
    if (config_.save_surface_features && !feature.surface_features->empty()) {
      *local_cloud += *feature.surface_features;
    }
    if (local_cloud->empty())
      continue;

    // Transform to global frame
    PosCloud::Ptr global_cloud(new PosCloud);
    pcl::transformPointCloud(*local_cloud, *global_cloud, pose.matrix());

    Eigen::Vector3d pos = pose.translation();
    int tile_x = (pos.x() - min_bound.x()) / config_.tile_size;
    int tile_y = (pos.y() - min_bound.y()) / config_.tile_size;

    auto key = std::make_pair(tile_x, tile_y);
    if (tile_clouds.find(key) == tile_clouds.end()) {
      tile_clouds[key] = PosCloud::Ptr(new PosCloud);
    }

    *tile_clouds[key] += *global_cloud;
  }

  std::ofstream tile_index(save_path + "_tile_index.txt");
  tile_index << "# Tile Index File\n";
  tile_index << "# tile_x tile_y num_points filename\n";

  int tile_count = 0;
  for (auto &tile_pair : tile_clouds) {
    int tx = tile_pair.first.first;
    int ty = tile_pair.first.second;
    PosCloud::Ptr &cloud = tile_pair.second;

    // Downsampling
    PosCloud::Ptr ds_cloud = DownsampleCloud(cloud, config_.voxel_leaf_size);

    std::stringstream ss;
    ss << save_path << "_tile_" << std::setfill('0') << std::setw(4) << tx
       << "_" << std::setw(4) << ty << ".pcd";
    std::string tile_filename = ss.str();

    if (SaveCloudToFile(ds_cloud, tile_filename, config_.use_compression)) {
      tile_index << tx << " " << ty << " " << ds_cloud->size() << " "
                 << tile_filename << "\n";
      tile_count++;
      stats_.saved_points += ds_cloud->size();
    }
  }

  tile_index.close();
  LOG(INFO) << "Successfully saved " << tile_count << " tiles";

  return true;
}

bool MapSaver::SaveSparseFeatureMap(
    const std::map<int64_t, LiDARFeature> &keyframes,
    const Trajectory::Ptr &trajectory, const std::string &save_path) {

  LOG(INFO) << "[SaveSparseFeatureMap] Saving sparse feature map...";

  PosCloud::Ptr corner_map(new PosCloud);
  PosCloud::Ptr surface_map(new PosCloud);

  for (const auto &kf_pair : keyframes) {
    int64_t timestamp = kf_pair.first;
    const LiDARFeature &feature = kf_pair.second;
    SE3d pose = trajectory->GetLidarPoseNURBS(timestamp);

    if (!feature.corner_features->empty()) {
      PosCloud::Ptr global_corners(new PosCloud);
      pcl::transformPointCloud(*feature.corner_features, *global_corners,
                               pose.matrix());
      *corner_map += *global_corners;
    }

    if (!feature.surface_features->empty()) {
      PosCloud::Ptr global_surfaces(new PosCloud);
      pcl::transformPointCloud(*feature.surface_features, *global_surfaces,
                               pose.matrix());
      *surface_map += *global_surfaces;
    }
  }

  if (!corner_map->empty()) {
    PosCloud::Ptr ds_corners =
        DownsampleCloud(corner_map, config_.voxel_leaf_size * 0.5);
    SaveCloudToFile(ds_corners, save_path + "_corners.pcd", true);
    stats_.saved_points += ds_corners->size();
    LOG(INFO) << "Number of Corners: " << ds_corners->size();
  }

  if (!surface_map->empty()) {
    PosCloud::Ptr ds_surfaces =
        DownsampleCloud(surface_map, config_.voxel_leaf_size);
    SaveCloudToFile(ds_surfaces, save_path + "_surfaces.pcd", true);
    stats_.saved_points += ds_surfaces->size();
    LOG(INFO) << "Number of Surfaces: " << ds_surfaces->size();
  }

  return true;
}

void MapSaver::SaveMapMetadata(const std::string &save_path,
                               size_t num_keyframes, size_t total_points,
                               const Eigen::Vector3d &map_center,
                               const Eigen::Vector3d &map_extent) {
  std::ofstream meta_file(save_path);
  if (!meta_file.is_open()) {
    LOG(WARNING) << "Cannot create metadata file: " << save_path;
    return;
  }

  meta_file << "# Map Metadata\n";
  meta_file << "num_keyframes: " << num_keyframes << "\n";
  meta_file << "total_points: " << total_points << "\n";
  meta_file << "map_center: " << map_center.x() << " " << map_center.y() << " "
            << map_center.z() << "\n";
  meta_file << "map_extent: " << map_extent.x() << " " << map_extent.y() << " "
            << map_extent.z() << "\n";
  meta_file << "voxel_size: " << config_.voxel_leaf_size << "\n";
  meta_file << "save_mode: " << config_.mode << "\n";

  meta_file.close();
  LOG(INFO) << "Metadata saved: " << save_path;
}

size_t MapSaver::EstimateMapMemoryUsage(
    const std::map<int64_t, LiDARFeature> &keyframes) {
  size_t total_bytes = 0;

  for (const auto &kf_pair : keyframes) {
    const LiDARFeature &feature = kf_pair.second;

    // Each point approx 16 bytes (x, y, z, intensity)
    total_bytes += feature.corner_features->size() * 16;
    total_bytes += feature.surface_features->size() * 16;
    total_bytes += feature.full_cloud->size() * 16;
  }

  return total_bytes;
}

void MapSaver::PrintSaveStatistics() {
  LOG(INFO) << "\n========== Save Statistics ==========";
  LOG(INFO) << "Num Keyframes: " << stats_.num_keyframes;
  LOG(INFO) << "Original Points: " << stats_.original_points;
  LOG(INFO) << "Saved Points: " << stats_.saved_points;
  LOG(INFO) << "Compression Ratio: " << std::fixed << std::setprecision(2)
            << stats_.compression_ratio << "x";
  LOG(INFO) << "Memory Savings: " << std::fixed << std::setprecision(1)
            << (1.0 - 1.0 / stats_.compression_ratio) * 100 << "%";
  LOG(INFO) << "Time Elapsed: " << stats_.save_time_ms << " ms";

  // Estimate file size
  size_t estimated_size_mb = (stats_.saved_points * 16) / (1024 * 1024);
  LOG(INFO) << "Estimated File Size: " << estimated_size_mb << " MB";
  LOG(INFO) << "==============================\n";
}

PosCloud::Ptr MapSaver::DownsampleCloud(const PosCloud::Ptr &cloud_in,
                                        float leaf_size) {
  if (cloud_in->empty()) {
    return PosCloud::Ptr(new PosCloud);
  }

  PosCloud::Ptr cloud_out(new PosCloud);
  VoxelFilter<PosPoint> voxel_filter;
  voxel_filter.SetResolution(leaf_size);
  voxel_filter.SetInputCloud(cloud_in);
  voxel_filter.Filter(cloud_out);

  return cloud_out;
}

PosCloud::Ptr MapSaver::FilterByDistance(const PosCloud::Ptr &cloud_in,
                                         const Eigen::Vector3d &center,
                                         float max_dist) {
  PosCloud::Ptr cloud_out(new PosCloud);
  float max_dist_sq = max_dist * max_dist;

  for (const auto &point : cloud_in->points) {
    Eigen::Vector3d pt(point.x, point.y, point.z);
    if ((pt - center).squaredNorm() <= max_dist_sq) {
      cloud_out->push_back(point);
    }
  }

  return cloud_out;
}

PosCloud::Ptr MapSaver::MergeClouds(const std::vector<PosCloud::Ptr> &clouds) {
  PosCloud::Ptr merged(new PosCloud);
  for (const auto &cloud : clouds) {
    *merged += *cloud;
  }
  return merged;
}

void MapSaver::ComputeMapBounds(const PosCloud::Ptr &cloud,
                                Eigen::Vector3d &min_bound,
                                Eigen::Vector3d &max_bound) {
  if (cloud->empty())
    return;

  min_bound = Eigen::Vector3d(1e6, 1e6, 1e6);
  max_bound = Eigen::Vector3d(-1e6, -1e6, -1e6);

  for (const auto &point : cloud->points) {
    min_bound.x() = std::min(min_bound.x(), (double)point.x);
    min_bound.y() = std::min(min_bound.y(), (double)point.y);
    min_bound.z() = std::min(min_bound.z(), (double)point.z);

    max_bound.x() = std::max(max_bound.x(), (double)point.x);
    max_bound.y() = std::max(max_bound.y(), (double)point.y);
    max_bound.z() = std::max(max_bound.z(), (double)point.z);
  }
}

bool MapSaver::SaveCloudToFile(const PosCloud::Ptr &cloud,
                               const std::string &filename, bool compressed) {
  if (cloud->empty()) {
    LOG(WARNING) << "Cloud is empty, skip saving: " << filename;
    return false;
  }

  LOG(INFO) << "Saving Cloud: " << filename << " (" << cloud->size() << " points)";

  int ret = 0;
  if (compressed && config_.use_binary_pcd) {
    ret = pcl::io::savePCDFileBinaryCompressed(filename, *cloud);
  } else if (config_.use_binary_pcd) {
    ret = pcl::io::savePCDFileBinary(filename, *cloud);
  } else {
    ret = pcl::io::savePCDFileASCII(filename, *cloud);
  }

  if (ret == 0) {
    LOG(INFO) << "✓ Save Success";
    return true;
  } else {
    LOG(ERROR) << "✗ Save Failed: " << filename;
    return false;
  }
}

} // namespace liuo
