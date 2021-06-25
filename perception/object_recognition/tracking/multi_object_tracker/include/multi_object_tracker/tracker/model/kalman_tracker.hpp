/*
 * Copyright 2018 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * v1.0 Yukihiro Saito
 */

#pragma once
#include "autoware_perception_msgs/DynamicObject.h"
#include "tracker_base.hpp"
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>



class KalmanTracker : public Tracker
{
private:
  autoware_perception_msgs::DynamicObject object_;
  double filtered_posx_;
  double filtered_posy_;
  double pos_filter_gain_;
  double filtered_vx_;
  double filtered_vy_;
  double v_filter_gain_;
  double filtered_area_;
  double area_filter_gain_;
  double last_measurement_posx_;
  double last_measurement_posy_;
  ros::Time last_update_time_;
  ros::Time last_measurement_time_;

  Eigen::VectorXd _x; //t-1时刻状态量.(pose,vx,vy,dim等)
  Eigen::MatrixXd _P; //t-1时刻的协方差矩阵.说明状态量的不确定程度.
  Eigen::MatrixXd _F; //状态转移矩阵.
  Eigen::MatrixXd _Q; //t-1时刻的系统噪声矩阵.
  Eigen::MatrixXd _I; 

  Eigen::MatrixXd _P_CV;
  Eigen::MatrixXd _F_CV;
  Eigen::MatrixXd _Q_CV;
  Eigen::MatrixXd _radar_R_CV;
  Eigen::MatrixXd _lidar_R_CV;
  Eigen::MatrixXd _radar_H_CV;
	Eigen::MatrixXd _lidar_H_CV;

public:
  KalmanTracker(const ros::Time & time, sensor_type sensor,const autoware_perception_msgs::DynamicObject & object);

  bool predict(const ros::Time & time) override;
  bool measure(
    const autoware_perception_msgs::DynamicObject & object, sensor_type sensor,const ros::Time & time) override;
  bool getEstimatedDynamicObject(
    const ros::Time & time, autoware_perception_msgs::DynamicObject & object) override;
  virtual ~KalmanTracker(){};
  // double getS(Eigen::MatrixXd s) override;
  
  int _n_CV = 4;
  int _lidar_n_CV = 2;
  int _radar_n_CV = 4;
  double _acc_x_CV;
  double _acc_y_CV;
};
