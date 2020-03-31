#include <algorithm>
#include <numeric>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
#include <pcl/search/kdtree.h>

template<typename TreeT, typename PointT>
inline float nearestDistance(const TreeT& tree, const PointT& pt)
{
  const int k = 1;
  std::vector<int> indices (k);
  std::vector<float> sqr_distances (k);

  tree.nearestKSearch(pt, k, indices, sqr_distances);

  return sqr_distances[0];
}

// compare cloudB to cloudA
// use threshold for identifying outliers and not considering those for the similarity
// a good value for threshold is 5 * <cloud_resolution>, e.g. 10cm for a cloud with 2cm resolution
template<typename CloudT>
float _similarity(const CloudT& cloudA, const CloudT& cloudB, float threshold)
{
  // compare B to A
  int num_outlier = 0;
  pcl::search::KdTree<typename CloudT::PointType> tree;
  tree.setInputCloud(cloudA.makeShared());
  auto sum = std::accumulate(cloudB.begin(), cloudB.end(), 0.0f, [&](auto current_sum, const auto& pt) {
    const auto dist = nearestDistance(tree, pt);

    if(dist < threshold)
    {
      return current_sum + dist;
    }
    else
    {
      num_outlier++;
      return current_sum;
    }
  });

  return sum / (cloudB.size() - num_outlier);
}

// comparing the clouds each way, A->B, B->A and taking the average
template<typename CloudT>
inline float similarity(const CloudT* cloudA, const CloudT* cloudB, float threshold = std::numeric_limits<float>::max())
{
  // compare B to A
  const auto similarityB2A = _similarity(*cloudA, *cloudB, threshold);
  // compare A to B
  const auto similarityA2B = _similarity(*cloudB, *cloudA, threshold);

  return (similarityA2B * 0.5f) + (similarityB2A * 0.5f);
}
