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
 *
 * v1.0 Yukihiro Saito
 */

#include "multi_object_tracker/tracker/model/kalman_tracker.hpp"
#include "multi_object_tracker/utils/utils.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

KalmanTracker::KalmanTracker(
  const ros::Time & time,sensor_type sensor, const autoware_perception_msgs::DynamicObject & object)

: Tracker(time, object.semantic.type),
  filtered_posx_(object.state.pose_covariance.pose.position.x),
  filtered_posy_(object.state.pose_covariance.pose.position.y),
  
  area_filter_gain_(0.8),
  last_measurement_posx_(object.state.pose_covariance.pose.position.x),
  last_measurement_posy_(object.state.pose_covariance.pose.position.y),
  last_update_time_(time),
  last_measurement_time_(time)
{
  object_ = object;
  
  Eigen::VectorXd x_CV(4);
	
	_acc_x_CV = 4;
  _acc_y_CV = 4;

  _lidar_R_CV = Eigen::MatrixXd(_lidar_n_CV, _lidar_n_CV);
	_lidar_H_CV = Eigen::MatrixXd(_lidar_n_CV, _n_CV);
  _radar_R_CV = Eigen::MatrixXd(_radar_n_CV, _radar_n_CV);
	_radar_H_CV = Eigen::MatrixXd(_radar_n_CV, _n_CV);


   
	_P_CV = Eigen::MatrixXd(_n_CV, _n_CV);
    //[1,0,0,0]
    //[0,1,0,0]
    //[0,0,1,0]
    //[0,0,0,1]
	_F_CV = Eigen::MatrixXd::Identity(_n_CV, _n_CV);
	_Q_CV = Eigen::MatrixXd::Zero(_n_CV, _n_CV);
    
    //测量矩阵H
	_lidar_H_CV <<  1.0, 0.0, 0.0, 0.0,
				          0.0, 1.0, 0.0, 0.0;
  
  _radar_H_CV <<  1.0, 0.0, 0.0, 0.0,
	          			0.0, 1.0, 0.0, 0.0,
				          0.0, 0.0, 1.0, 0.0,
				          0.0, 0.0, 0.0, 1.0;
  
  
  //不同传感器的状态量和协方差矩阵是不同的  协方差矩阵表达了不同的状态量之间的相关性.
  if(sensor == sensor_type::lidar)
  {
    double distance_cov_CV = 1;
    double velocity_cov_CV = 2;
    x_CV(0) = last_measurement_posx_;
    x_CV(1) = last_measurement_posy_;
    x_CV(2) = 0;
    x_CV(3) = 0;

    _P_CV <<   	distance_cov_CV, 0.0, 0.0, 0.0,
                0.0, distance_cov_CV, 0.0, 0.0,
                0.0, 0.0, velocity_cov_CV, 0.0,
                0.0, 0.0, 0.0, velocity_cov_CV;
  }
  else if(sensor == sensor_type::radar)
  {
    double distance_cov_CV = 2;
    double velocity_cov_CV = 0.1;
    x_CV(0) = last_measurement_posx_;
    x_CV(1) = last_measurement_posy_;
    x_CV(2) = object.state.twist_covariance.twist.linear.x;
    x_CV(3) = object.state.twist_covariance.twist.linear.y;

    _P_CV <<   	distance_cov_CV, 0.0, 0.0, 0.0,
                0.0, distance_cov_CV, 0.0, 0.0,
                0.0, 0.0, velocity_cov_CV, 0.0,
                0.0, 0.0, 0.0, velocity_cov_CV;
  }
  _I = Eigen::MatrixXd::Identity(_n_CV, _n_CV);
  _x = x_CV;
  _P = _P_CV;
  _F = _F_CV;  //初始化控制矩阵
  _Q = _Q_CV;

}

//做两件事: 根据t-1时刻的状态量x,系统协方差P预测t时刻的状态量x',协方差P'.
//1.根据运动学模型建立的预测方程. 预测t时刻状态量. x=[x,y,vx,vy]  x'=Fx
//2.P'=FPFT + Q  更新协方差 
bool KalmanTracker::predict(const ros::Time & time)
{
  double dt = (time - last_update_time_).toSec();
  last_update_time_ = time;

  if (dt < 0.0) 
  {
    dt = 0.0;
    return false;
  }

  //更新状态量  x=[x,y,vx,vy]  x'=Fx  公式1
  _x(0) = _x(0) + _x(2)*dt;
	_x(1) = _x(1) + _x(3)*dt;
	_x(2) = _x(2);
	_x(3) = _x(3);

  //更新控制矩阵.
    //[1,0,dt,0]
    //[0,1,0,dt]
    //[0,0,1,0]
    //[0,0,0,1]
  _F(0, 2) = dt;
  _F(1, 3) = dt;

  const double dt2 = dt * dt;
  const double dt3 = dt * dt2;
  const double dt4 = dt * dt3;

  const double r11 = dt4 * _acc_x_CV / 4;
  const double r13 = dt3 * _acc_x_CV / 2;
  const double r22 = dt4 * _acc_y_CV / 4;
  const double r24 = dt3 * _acc_y_CV / 2;
  const double r31 = dt3 * _acc_x_CV / 2;
  const double r33 = dt2 * _acc_x_CV;
  const double r42 = dt3 * _acc_y_CV / 2;
  const double r44 = dt2 * _acc_y_CV;
  
  //系统的噪声矩阵怎么来的? 为何要实时更新?
  _Q_CV <<  r11, 0.0, r13, 0.0,
		        0.0, r22, 0.0, r24,
		        r31, 0.0, r33, 0.0,
		        0.0, r42, 0.0, r44;

  _Q  = _Q_CV;                           //
  _P  = _F * _P * _F.transpose() + _Q;   //P'=FPFT + Q  更新协方差 公式2

  return true;
}

//新的目标检测结果得到以后,更新卡尔曼滤波的各个参数.
bool KalmanTracker::measure(
  const autoware_perception_msgs::DynamicObject & object, sensor_type sensor,const ros::Time & time)
{
  
  // vx,vy
  double dt = (time - last_measurement_time_).toSec();
  last_measurement_time_ = time;
  last_update_time_ = time;

  /****/
  if (0.0 < dt)
  {
    if(sensor == sensor_type::lidar)
    {
      int type = object.semantic.type;
      if (type == autoware_perception_msgs::Semantic::UNKNOWN)
      {
        type = getType();
        setType(type);
      }
      else
      {
        object_ = object;
      }
      
      double R_dis_cov_CV = 0.01;
      
      //检测器的噪声矩阵R  不变.
      _lidar_R_CV << 	R_dis_cov_CV, 0.0,
                      0.0, R_dis_cov_CV;
      
      Eigen::VectorXd z(2);
      z(0) = object.state.pose_covariance.pose.position.x;
      z(1) = object.state.pose_covariance.pose.position.y;

      Eigen::MatrixXd H  = _lidar_H_CV; //测量矩阵 不变
      Eigen::VectorXd Hx = _lidar_H_CV * _x;
      Eigen::MatrixXd R  = _lidar_R_CV;

      Eigen::MatrixXd PHt = _P * H.transpose();
      Eigen::MatrixXd S = H * PHt + R; //公式4:S=HP'HT + R
      Eigen::MatrixXd K = PHt * S.inverse(); //卡尔曼增益. 用于估计误差的重要程度.  公式5:K=P'HtS.inverse()
      Eigen::VectorXd y = z - Hx;  //计算detection和tracking的误差.  公式3
      _x = _x + K * y;     //更新系统均值  公式6
      _P = (_I - K * H) * _P;  //更新系统协方差 公式7
      // std::cout<<"lidar _x : "<<_x<<std::endl;
    }
    else if(sensor == sensor_type::radar)
    {
      double R_dis_cov_CV = 1.0*1.0;
      double R_vel_cov_CV = 0.4*0.4;
      
      if(0 == object.state.radar_velocity_confidence)
      {
        R_vel_cov_CV = 1.0*1.0;
      }
      _radar_R_CV << 	R_dis_cov_CV, 0.0,0.0,0.0,
                      0.0, R_dis_cov_CV,0.0,0.0,
                      0.0, 0.0,R_vel_cov_CV,0.0,
                      0.0, 0.0,0.0,R_vel_cov_CV;
      ROS_INFO("radar measurement updata conv velocity: %lf",R_vel_cov_CV);
      Eigen::VectorXd z(4);
      z(0) = object.state.pose_covariance.pose.position.x;
      z(1) = object.state.pose_covariance.pose.position.y;
      z(2) = object.state.vel_map.x;
      z(3) = object.state.vel_map.y;
      // std::cout<<"radar _z : "<<z<<std::endl;
      
      Eigen::MatrixXd H  = _radar_H_CV;
      Eigen::VectorXd Hx = _radar_H_CV * _x;
      Eigen::MatrixXd R  = _radar_R_CV;

      Eigen::MatrixXd PHt = _P * H.transpose();
      // std::cout<<"p : "<< _P <<std::endl;
      // std::cout<<"pht : "<< PHt <<std::endl;

      Eigen::MatrixXd S = H * PHt + R;
      Eigen::MatrixXd K = PHt * S.inverse();
      Eigen::VectorXd y = z - Hx;
      // std::cout<<"radar bef _x : "<<_x<<std::endl;

      _x = _x + K * y;
      _P = (_I - K * H) * _P;
      // std::cout<<"radar aft _x : "<<_x<<std::endl;
    }

    
  }

  
  return true;
}

// double KalmanTracker::getS(Eigen::MatrixXd s)
// {
//   double ss = 0;
//   return ss;
// }

bool KalmanTracker::getEstimatedDynamicObject(
  const ros::Time & time, autoware_perception_msgs::DynamicObject & object)
{
  object = object_;
  object.id = unique_id::toMsg(getUUID());
  object.semantic.type = getType();

  double dt = (time - last_update_time_).toSec();
  if (dt < 0.0) dt = 0.0;
  
  filtered_posx_  = _x(0);
  filtered_posy_  = _x(1);
  filtered_vx_    = _x(2);
  filtered_vy_    = _x(3);
  // std::cout<<"estimation _x :"<<_x<<std::endl;
  //ROS_INFO("obj x:%lf y:%lf vx:%lf vy:%lf",object.state.pose_covariance.pose.position.x,object.state.pose_covariance.pose.position.y,object.state.twist_covariance.twist.linear.x,object.state.twist_covariance.twist.linear.y);
  //计算yaw
  
  
   object.state.orientation_reliable = false;

  if(std::sqrt(filtered_vx_*filtered_vx_ + filtered_vy_*filtered_vy_)> 1.5)
  {
    double yaw_cv = std::atan2(_x(3), _x(2));
    tf2::Quaternion quat;
    quat.setRPY(0,0,yaw_cv);
    object.state.pose_covariance.pose.orientation.x = quat.x();
    object.state.pose_covariance.pose.orientation.y = quat.y();
    object.state.pose_covariance.pose.orientation.z = quat.z();
    object.state.pose_covariance.pose.orientation.w = quat.w();
    object.state.orientation_reliable = true;
    object.state.twist_reliable = true;
  }
  if(std::sqrt(filtered_vx_*filtered_vx_ + filtered_vy_*filtered_vy_)<= 1.5)
  {
    filtered_vx_ = 0;
    filtered_vy_ = 0;
  }

  object.state.vel_map.x = filtered_vx_;
  object.state.vel_map.y = filtered_vy_;

  object.state.pose_covariance.pose.position.x = filtered_posx_;
  object.state.pose_covariance.pose.position.y = filtered_posy_;
  object.state.pose_covariance.pose.position.x += filtered_vx_ * dt;
  object.state.pose_covariance.pose.position.y += filtered_vy_ * dt;

  // object.state.orientation_reliable = false;

  double roll, pitch, yaw;
  tf2::Quaternion quaternion;
  tf2::fromMsg(object.state.pose_covariance.pose.orientation, quaternion);
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
  object.state.twist_covariance.twist.linear.x = filtered_vx_ * std::cos(-yaw) - filtered_vy_ * std::sin(-yaw);
  object.state.twist_covariance.twist.linear.y = filtered_vx_ * std::sin(-yaw) + filtered_vy_ * std::cos(-yaw);
  // object.state.twist_covariance.twist.linear.x = filtered_vx_;
  // object.state.twist_covariance.twist.linear.y = filtered_vy_;
  // if(isnan(object.state.pose_covariance.pose.position.x)|| isnan(object.state.pose_covariance.pose.position.y)
  //   ||isnan(object.state.twist_covariance.twist.linear.x)||isnan(object.state.twist_covariance.twist.linear.y)
  //   ||isnan(yaw))
  // {
  // std::cout<< "px: "<< object.state.pose_covariance.pose.position.x <<"py: "<<object.state.pose_covariance.pose.position.y 
  //          <<"vx: "<<  object.state.twist_covariance.twist.linear.x <<"vy: "<<object.state.twist_covariance.twist.linear.y
  //          <<"yaw: " << yaw <<std::endl;
  // return false;
  // }

  object.state.twist_reliable = true;
  
  return true;
}
