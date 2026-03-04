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

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "../odom/msg_manager.h"

namespace liuo {

class UWBHandler {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<UWBHandler> Ptr;

  UWBHandler(const YAML::Node& node);
  ~UWBHandler() = default;

  // Update UWB measurements with new data
  void UpdateUWBMeasurements(const int64_t timestamp,
                             const std::map<int, double>& distances);

  // Update UWB measurements with UwbData
  void UpdateUWBMeasurements(const UwbData& uwb_data);

  // Get current UWB measurements
  std::deque<UwbData> GetUWBMeasurements() const;

  // Set anchor positions from configuration
  void SetAnchorPositions(const std::map<int, Eigen::Vector3d>& positions);

  // Get anchor positions
  std::map<int, Eigen::Vector3d> GetAnchorPositions() const;

  // Clear old measurements before a given timestamp
  void ClearOldMeasurements(int64_t timestamp);

  // Get the latest measurement timestamp
  int64_t GetLatestTimestamp() const;

  // Check if we have valid anchor positions
  bool HasValidAnchors() const;

 public:
  std::map<int, Eigen::Vector3d> anchor_positions_;
  std::unordered_map<
      int, Eigen::Vector3d, std::hash<int>, std::equal_to<int>,
      Eigen::aligned_allocator<std::pair<const int, Eigen::Vector3d>>>
      anchor_id_positions_;

 private:
  // Configuration parameters
  double max_range_;        // Maximum valid UWB range
  size_t max_buffer_size_;  // Maximum number of measurements to store
  Eigen::Isometry3d extrinsics_ = Eigen::Isometry3d::Identity();

  // Data storage
  std::deque<UwbData> measurements_;

  // Thread safety
  mutable std::mutex mutex_;

  // Validate UWB measurement
  bool IsValidMeasurement(double distance) const;
};

}  // namespace liuo