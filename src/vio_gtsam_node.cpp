#define BOOST_BIND_GLOBAL_PLACEHOLDERS 0 // boost warning

#include "ros/ros.h"
#include "vio_gtsam.hpp"

#include <message_filters/sync_policies/approximate_time.h> // message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
#include <message_filters/subscriber.h> // message_filters::Subscriber<sensor_msgs::Image>
#include <sensor_msgs/Image.h> // sensor_msgs::ImageConstPtr
#include "sensor_msgs/Imu.h" // sensor_msgs::Imu::ConstPtr
#include "opencv2/imgproc/imgproc.hpp" // cv::Mat
#include <cv_bridge/cv_bridge.h> // cv_bridge::toCvCopy

#include "opencv2/video/tracking.hpp" // KLT

using namespace std;

struct FeatureSet
{
    std::vector<cv::Point2f> features;
    std::vector<int> ids;
    size_t get_size() { return features.size(); }
}

struct VIONode
{

    VIONode(ros::NodeHandle & n);

    void stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right);

    const string img_left = "/stereo/left/image_rect";
    const string img_right = "/stereo/right/image_rect";
    const int img_queue_size = 1;

    cv::Mat proj_mat_l;
    cv::Mat proj_mat_r;

    cv::Mat image_left_t0;
    cv::Mat image_right_t0;
    cv::Mat image_left_t1;
    cv::Mat image_right_t1;

private:
    VIOEstimator vio_estimator;
    ros::Subscriber imu_subscriber;
    size_t frame_id = 0;

    FeatureSet feature_set;
    size_t feature_id = 0;

    cv::Mat ros_img_to_cv_img(const sensor_msgs::ImageConstPtr img) const;
    void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);
    void stereo_update(const cv::Mat & image_left, const cv::Mat & image_right);
    void feature_matching();
    void circular_matching();
    void add_new_landmarks();

    void detect_new_features(std::vector<cv::Point2f> & points, 
        std::vector<int> & response_strengths) const;

    void bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
        const std::vector<int> & strengths);

    void triangulate_features(const std::vector<cv::Point2f> & points_left, 
        const std::vector<cv::Point2f> & points_right, 
        cv::Mat & points3D) const;
};

VIONode::VIONode(ros::NodeHandle & n)
{
    // initialize gtsam estimator
    vio_estimator.initialize_pose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); // initialize pose (attitude important)
    vio_estimator.init();
    vio_estimator.test_odometry_plus_loop_closure();

    imu_subscriber = n.subscribe("/imu/data", 1000, &VIONode::imu_callback,  this);

    // camera projection matrices
    const float fx = vio_estimator.camera_params.fx; 
    const float fy = vio_estimator.camera_params.fy;
    const float cx = vio_estimator.camera_params.cx;
    const float cy = vio_estimator.camera_params.cy;
    const float b = vio_estimator.camera_params.b;
    proj_mat_l = (cv::Mat_<double>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    proj_mat_r = (cv::Mat_<double>(3, 4) << fx, 0., cx, b, 0., fy, cy, 0., 0,  0., 1., 0.);
}

// ros images to opencv images
cv::Mat VIONode::ros_img_to_cv_img(const sensor_msgs::ImageConstPtr img) const 
{
    cv_bridge::CvImagePtr cv_ptr;
    // sandbox exception
    try {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) 
    {
        std::cerr << "exception" << std::endl;
        return cv::Mat();
    }
    return cv_ptr->image;
}

void VIONode::delete_unmatched_features(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          FeatureSet & current_features)
{
    int idx_correction = 0;
    for(int i = 0; i < status3.size(); i++)
    {  
        cv::Point2f pt0 = points0.at(i - idx_correction);
        cv::Point2f pt1 = points1.at(i - idx_correction);
        cv::Point2f pt2 = points2.at(i - idx_correction);
        cv::Point2f pt3 = points3.at(i - idx_correction);
        cv::Point2f pt0_r = points0_return.at(i - idx_correction);
        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
            (status2.at(i) == 0) || (pt2.x < 0) || (pt2.y < 0) ||
            (status1.at(i) == 0) || (pt1.x < 0) || (pt1.y < 0) ||
            (status0.at(i) == 0) || (pt0.x < 0) || (pt0.y < 0))   
        {
            if((pt0.x < 0) || (pt0.y < 0) || (pt1.x < 0) || (pt1.y < 0) || 
                (pt2.x < 0) || (pt2.y < 0) || (pt3.x < 0) || (pt3.y < 0))    
            {
                status3.at(i) = 0;
            }
            points0.erase(points0.begin() + (i - idx_correction));
            points1.erase(points1.begin() + (i - idx_correction));
            points2.erase(points2.begin() + (i - idx_correction));
            points3.erase(points3.begin() + (i - idx_correction));
            points0_return.erase(points0_return.begin() + (i - idx_correction));

            // also update the feature set 
            current_features.ages.erase (current_features.ages.begin() + (i - idx_correction));
            current_features.strengths.erase (current_features.strengths.begin() + (i - idx_correction));
            idx_correction++;
        }
    }  
}

void VIONode::circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) { 
    
    std::vector<float> err;         
    cv::Size winSize=cv::Size(20,20); // Lucas-Kanade optical flow window size                                                                                          
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;
    // sparse iterative version of the Lucas-Kanade optical flow in pyramids
    cv::calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, 
        winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, 
        winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, 
        winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, 
        winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    delete_unmatched_features(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
        status0, status1, status2, status3, current_features);
}

void VIONode::triangulate_features(const std::vector<cv::Point2f> & points_left, 
    const std::vector<cv::Point2f> & points_right, 
    cv::Mat & points3D) const
{
    cv::Mat points3D, points4D;
    cv::triangulatePoints(proj_mat_l, proj_mat_r, points_left, points_right, points4D);
    cv::convertPointsFromHomogeneous(points4D.t(), points3D);
}

void VIONode::detect_new_features(std::vector<cv::Point2f> & features, 
    std::vector<int> & strengths) const
{
    std::vector<cv::KeyPoint> keypoints;
    bool nonmax_suppression = true;
    const int fast_threshold = 20;

    // FAST feature detector
    cv::FAST(image_left_t0, keypoints, fast_threshold, nonmax_suppression);
    cv::KeyPoint::convert(keypoints, features, std::vector<int>());

    // Feature corner strengths
    strengths.reserve(features.size());
    for (const auto & keypoint : keypoints) strengths.push_back(keypoint.response); 
}

void bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
    const std::vector<int> & strengths)
{
    // sort features by strengths

    const int num_buckets_per_axis = 10;
    const int num_features_per_bucket = 2;
    const int num_features_min = 100;

    // iterate through features 
    std::vector<int> buckets;
    buckets.resize(num_buckets_per_axis*num_buckets_per_axis);
    std::vector<int> idx;
    for (size_t i = 0; i < features.size(); ++i)
    {
        // compute bucket idx
        int idx = 0;

        // check if bucket has space
        if (buckets[idx] < num_features_per_bucket)
        {
            // add bucket counter
            buckets[idx]++;
            // add index to output
            idx.push_back(i);
            // leave loop if enough features found
            if (idx.size() == num_features_min) break;
        }
    }

    // for now clear everything and start over
    feature_set.features.resize(idx.size());
    feature_set.ids.resize(idx.size());
    for (size_t i = 0; i < idx.size(); ++i)
    {
        feature_set.features.push_back(features[i]);
        feature_set.ids.push_back(feature_id);
        feature_id++;
    }
}


void VIONode::replace_all_features()
{
    // detect features in image
    std::vector<cv::Point2f> features;
    std::vector<int> strengths;
    detect_new_features(features, strengths);

    // bucket features in image
    bucket_and_update_feature_set(features, strengths);
}

void VIONode::transform_to_world(const cv::Mat & points3D_cam, cv::Mat & points3D_world)
{
    Pose3 pose = vio_estimator.get_pose();
}

void VIONode::feature_matching()
{
    const int min_num_features = 100;
    static size_t landmark_id = 0;

    bool replaced_features = false;

    // if ! enough features in current feature set
    if (feature_set.get_size() < min_num_features)
    {
        replace_all_features();
        replaced_features = true;

    }

    // tracks features between frames into next position and updates featureset positions
    circular_matching();

    if (replaced_features)
    {
        // triangulate and add em
    }
    update_feature_set();

    // iterate through features
    for (auto & feature : feature_set)
    {

    }

    // circular matching
    // triangulate inliers and add them to the set
    // update featureset with the new positions
    // detect features on image
    // bucketing, select N of the strongest features
    // track old features - circular matching
    // add new positions to stereo set

    // track new features - circular matching
    // triangulate inliers and add to featureset (increment landmark ids)
    // get current pose and put newly triangulated stereo points into world frame

    // add to stereo set

    // compute gtsam stuff

    // print pose
}

void VIONode::stereo_update(const cv::Mat & image_left, const cv::Mat & image_right)
{

    if (!frame_id)
    {
        image_left_t0 = image_left;
        image_right_t0 = image_right;
        frame_id++;
        return;
    } else 
    {
        image_left_t1 = image_left;
        image_right_t1 = image_right;
        frame_id++;
    }

    // feature matching
    feature_matching();

    // update images
    image_left_t0 = image_left_t1;
    image_right_t0 = image_right_t1;
}

void VIONode::stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
{
    cv::Mat l = ros_img_to_cv_img(image_left);
    cv::Mat r = ros_img_to_cv_img(image_right);
    stereo_update(l, r);

    // ros node just passes things along

    // some form of cleanup on the graph, merging, removing landmarks that have not enough views

    // some form of visualization
}

void VIONode::imu_callback(const sensor_msgs::Imu::ConstPtr& msg)
{
    geometry_msgs::Vector3 w = msg->angular_velocity;
    geometry_msgs::Vector3 f = msg->linear_acceleration;
    vio_estimator.integrate_imu(f.x, f.y, f.z, w.x, w.y, w.z);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vio_gtsam_node");

    ros::NodeHandle n;

    VIONode vio_node(n);

    // initialization procedure sets up the IMU orientation
    // imu camera extrinsics???

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, vio_node.img_left, vio_node.img_queue_size);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, vio_node.img_right, vio_node.img_queue_size);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(vio_node.img_queue_size), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&VIONode::stereo_callback, &vio_node, _1, _2));

    ros::spin();
    return 0;
}