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

#include "uwb_handler.h"

#include <glog/logging.h>

namespace liuo {

UWBHandler::UWBHandler(const YAML::Node& node) {
  // Load configuration parameters
  max_range_ = node["max_range"].as<double>();
  max_buffer_size_ = node["max_buffer_size"].as<size_t>();
  std::vector<std::string> anchor_ids =
      node["AnchorId"].as<std::vector<std::string>>();
  std::vector<double> anchor_positions =
      node["AnchorPos"].as<std::vector<double>>();
  Eigen::Vector3d trans_(node["Extrinsics"]["Trans"][0].as<double>(),
                         node["Extrinsics"]["Trans"][1].as<double>(),
                         node["Extrinsics"]["Trans"][2].as<double>());
  Eigen::Matrix3d rot_ = Eigen::Matrix3d::Identity();
  rot_ << node["Extrinsics"]["Rot"][0].as<double>(),
      node["Extrinsics"]["Rot"][1].as<double>(),
      node["Extrinsics"]["Rot"][2].as<double>(),
      node["Extrinsics"]["Rot"][3].as<double>(),
      node["Extrinsics"]["Rot"][4].as<double>(),
      node["Extrinsics"]["Rot"][5].as<double>(),
      node["Extrinsics"]["Rot"][6].as<double>(),
      node["Extrinsics"]["Rot"][7].as<double>(),
      node["Extrinsics"]["Rot"][8].as<double>();
  extrinsics_.rotate(rot_);
  extrinsics_.pretranslate(trans_);
  for (size_t i = 0; i < anchor_ids.size(); i++) {
    Eigen::Vector3d pos(anchor_positions[i * 3], anchor_positions[i * 3 + 1],
                        anchor_positions[i * 3 + 2]);
    int anchor_id = std::stoi(anchor_ids[i]);
    Eigen::Vector3d pos_body = extrinsics_ * pos;
    anchor_id_positions_[anchor_id] = pos_body;
  }
  LOG(INFO) << "UWBHandler initialized with max_range: " << max_range_
            << ", max_buffer_size: " << max_buffer_size_;
}

void UWBHandler::UpdateUWBMeasurements(const UwbData& uwb_data) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Add the measurement to our buffer
  measurements_.push_back(uwb_data);

  // Update anchor positions if they exist in the measurement
  for (const auto& [anchor_id, position] : uwb_data.anchor_positions) {
    anchor_positions_[anchor_id] = position;
  }

  // Keep buffer size within limits
  while (measurements_.size() > max_buffer_size_) {
    measurements_.pop_front();
  }
}

std::deque<UwbData> UWBHandler::GetUWBMeasurements() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return measurements_;
}

void UWBHandler::SetAnchorPositions(
    const std::map<int, Eigen::Vector3d>& positions) {
  std::lock_guard<std::mutex> lock(mutex_);
  anchor_positions_ = positions;

  // Update existing measurements with new anchor positions
  for (auto& measurement : measurements_) {
    for (const auto& dist_pair : measurement.anchor_distances) {
      auto pos_it = positions.find(dist_pair.first);
      if (pos_it != positions.end()) {
        measurement.anchor_positions[dist_pair.first] = pos_it->second;
      }
    }
  }

  LOG(INFO) << "Updated anchor positions with " << positions.size()
            << " anchors";
}

std::map<int, Eigen::Vector3d> UWBHandler::GetAnchorPositions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return anchor_positions_;
}

void UWBHandler::ClearOldMeasurements(int64_t timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);

  while (!measurements_.empty() &&
         measurements_.front().timestamp < timestamp) {
    measurements_.pop_front();
  }
}

int64_t UWBHandler::GetLatestTimestamp() const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (measurements_.empty()) {
    return -1;
  }
  return measurements_.back().timestamp;
}

bool UWBHandler::HasValidAnchors() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !anchor_positions_.empty();
}

bool UWBHandler::IsValidMeasurement(double distance) const {
  // Check if distance is within valid range
  return distance > 0.0 && distance < max_range_;
}

}  // namespace liuo