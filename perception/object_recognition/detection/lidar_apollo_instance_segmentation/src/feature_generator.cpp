/*
 * Copyright 2020 TierIV. All rights reserved.
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
 */

#include "lidar_apollo_instance_segmentation/feature_generator.h"
#include "lidar_apollo_instance_segmentation/log_table.h"

namespace
{
inline float normalizeIntensity(float intensity) { return intensity / 255.0f; }
}  // namespace

FeatureGenerator::FeatureGenerator(
  const int width, const int height, const int range, const bool use_intensity_feature,
  const bool use_constant_feature)
: min_height_(-5.0),
  max_height_(5.0),
  use_intensity_feature_(use_intensity_feature),
  use_constant_feature_(use_constant_feature)
{
  //select feature map type
  //送入模型的特征有多种生成方式.默认use_intensity_feature: true,use_constant_feature: false
  if (use_constant_feature && use_intensity_feature) 
  {
    map_ptr_ = std::make_shared<FeatureMapWithConstantAndIntensity>(width, height, range);
  } 
  else if (use_constant_feature) 
  {
    map_ptr_ = std::make_shared<FeatureMapWithConstant>(width, height, range);
  } 
  else if (use_intensity_feature) 
  {
    map_ptr_ = std::make_shared<FeatureMapWithIntensity>(width, height, range);
  } 
  else 
  {
    map_ptr_ = std::make_shared<FeatureMap>(width, height, range);
  }
  map_ptr_->initializeMap(map_ptr_->map_data);
}

std::shared_ptr<FeatureMapInterface> FeatureGenerator::generate(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr & pc_ptr)
{
  const double epsilon = 1e-6;
  //特征置为初始值
  map_ptr_->resetMap(map_ptr_->map_data);
  //特征总数目
  int size = map_ptr_->height * map_ptr_->width;

  /*
  width:640,height:640,range:60
  在width方向上,60m对应320个网格.
  －－－－－－－－－－－－－－ｘ
  |            X
  |            |
  |            |
  |　　　Ｙ－－LIDAR
  |
  |
  y

  如上图所示,lidar的X,Y方向的扫描距离60m.  x,y方向划分出640个网格.
　xy为伪图像坐标系. XY为lidar坐标系.

  */

  //计算出每一m占据几个grid
  const float inv_res_x = 0.5 * map_ptr_->width / map_ptr_->range;
  const float inv_res_y = 0.5 * map_ptr_->height / map_ptr_->range;

  for (size_t i = 0; i < pc_ptr->points.size(); ++i) 
  {
    //过滤掉过高的点和过低的点
    if (pc_ptr->points[i].z <= min_height_ || max_height_ <= pc_ptr->points[i].z) continue;

    //计算出xy坐标系下的坐标
    const int pos_x = std::floor((map_ptr_->range - pc_ptr->points[i].y) * inv_res_x);  // x on grid
    const int pos_y = std::floor((map_ptr_->range - pc_ptr->points[i].x) * inv_res_y);  // y on grid
    if (pos_x < 0 || map_ptr_->width <= pos_x || pos_y < 0 || map_ptr_->height <= pos_y) continue;

    //计算出(x,y)这个grid对应的索引idx
    const int idx = pos_y * map_ptr_->width + pos_x;

    if (map_ptr_->max_height_data[idx] < pc_ptr->points[i].z) 
    {
      //更新这个grid内的点的最大高度
      map_ptr_->max_height_data[idx] = pc_ptr->points[i].z;
      if (map_ptr_->top_intensity_data != nullptr)
        //更新这个grid内最高的点的强度. 强度/255
        map_ptr_->top_intensity_data[idx] = normalizeIntensity(pc_ptr->points[i].intensity);
    }
    //对于idx代表的每一个grid内的每个点的高度z相加
    map_ptr_->mean_height_data[idx] += static_cast<float>(pc_ptr->points[i].z);
    if (map_ptr_->mean_intensity_data != nullptr)
    　//对idx代表的每一个grid内的每个点的强度相加
      map_ptr_->mean_intensity_data[idx] += normalizeIntensity(pc_ptr->points[i].intensity);
    
    //更新idx代表的每一个grid内的点的数量
    map_ptr_->count_data[idx] += 1.0f;
  }

  for (int i = 0; i < size; ++i) 
  {
    if (map_ptr_->count_data[i] < epsilon) 
    {
      map_ptr_->max_height_data[i] = 0.0f;
    } 
    else 
    {
      //对每一个grid内的点的高度求均值
      map_ptr_->mean_height_data[i] /= map_ptr_->count_data[i];
      if (map_ptr_->mean_intensity_data != nullptr)
      　//对每一个grid内的点的强度求均值
        map_ptr_->mean_intensity_data[i] /= map_ptr_->count_data[i];
      
      //标记当前grid为非空grid,即包含有点的grid.
      map_ptr_->nonempty_data[i] = 1.0f;
    }

    //更新grid内的点的数量 x->ln(1+x)  why?
    map_ptr_->count_data[i] = calcApproximateLog(map_ptr_->count_data[i]);
  }
  return map_ptr_;
}
