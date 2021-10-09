#define BOOST_BIND_GLOBAL_PLACEHOLDERS 0 // boost warning

#include "vio_gtsam_node.hpp"

#include "message_filters/sync_policies/approximate_time.h" // message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
#include "message_filters/subscriber.h" // message_filters::Subscriber<sensor_msgs::Image>
#include <cv_bridge/cv_bridge.h> // cv_bridge::toCvCopy
#include <opencv2/video/tracking.hpp> // KLT
#include <opencv2/calib3d/calib3d.hpp> // triangulatePoints

#include <iostream> // cout

// try pure integration with imu
// compare y values

VIONode::VIONode(ros::NodeHandle & n)
{
    imu_subscriber = n.subscribe(imu_topic, 1000, &VIONode::imu_callback,  this);

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

// transform 3D points to world frame given pose: (p_w = R_wc*p_c + t_wc)
vector<cv::Point3d> VIONode::transform_to_world(const cv::Mat & points3D_cam, 
    const cv::Mat & R_wb,
    const cv::Mat & t_wb) const
{
    // transform point from camera frame to world frame via body frame
    vector<cv::Point3d> points3D_w;
    points3D_w.reserve(points3D_cam.rows);
    for (int i = 0; i < points3D_cam.rows; ++i)
    {
        cv::Point3d pt_c = {points3D_cam.at<cv::Point3f>(i).x, 
            points3D_cam.at<cv::Point3f>(i).y, 
            points3D_cam.at<cv::Point3f>(i).z};
        // convert to body frame
        cv::Mat pt_b = R_bc*cv::Mat(pt_c) + t_bc;
        // convert to world frame
        cv::Mat pt_w = R_wb*cv::Mat(pt_b) + t_wb;
        points3D_w.push_back(cv::Point3d(pt_w));
    }    
    return points3D_w;
}

void VIONode::triangulate_features(const std::vector<cv::Point2f> & features_left, 
    const std::vector<cv::Point2f> & features_right, 
    cv::Mat & features_3D) const
{
    assert(features_left.size() == features_right.size());
    if (features_left.size() > 0)
    {
        cv::Mat points4D;
        cv::triangulatePoints(proj_mat_l, proj_mat_r, features_left, features_right, points4D);
        cv::convertPointsFromHomogeneous(points4D.t(), features_3D);
    }
}

void VIONode::detect_new_features(std::vector<cv::Point2f> & features, 
    std::vector<int> & strengths) const
{
    std::vector<cv::KeyPoint> keypoints;
    bool nonmax_suppression = true;
    const int fast_threshold = 20;

    // FAST feature detector
    cv::FAST(image_left_t0, keypoints, fast_threshold, nonmax_suppression);
    cv::KeyPoint::convert(keypoints, features);

    // Feature corner strengths
    strengths.reserve(features.size());
    for (const auto & keypoint : keypoints) strengths.push_back(keypoint.response); 
}

void VIONode::bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
    const std::vector<int> & strengths)
{
    // sort features by strengths
    const int num_buckets_per_axis = 10;
    const int num_features_per_bucket = 3;
    const int num_features_min = 200;
    const int dim_bucket_x = img_width/num_buckets_per_axis;
    const int dim_bucket_y = img_height/num_buckets_per_axis;


    // iterate through features 
    std::vector<int> buckets;
    buckets.resize(num_buckets_per_axis*num_buckets_per_axis);
    std::vector<int> idx;
    for (size_t i = 0; i < features.size(); ++i)
    {
        // compute bucket idx
        int x = features[i].x;
        int y = features[i].y;

        int bucket_idx = x/dim_bucket_x + (y*num_buckets_per_axis)/dim_bucket_y;

        // check if bucket has space
        if (buckets[bucket_idx] < num_features_per_bucket)
        {
            // add bucket counter
            buckets[bucket_idx]++;
            // add index to output
            idx.push_back(i);
            // leave loop if enough features found
            if (idx.size() == num_features_min) break;
        }
    }

    // clear everything and start over
    reset_feature_set();
    feature_set.features_l.resize(idx.size());
    feature_set.ids.resize(idx.size());
    for (size_t i = 0; i < idx.size(); ++i)
    {
        feature_set.features_l.push_back(features[i]);
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

void VIONode::circular_matching(std::vector<cv::Point2f> & points_l_0, 
    std::vector<cv::Point2f> & points_r_0, 
    std::vector<cv::Point2f> & points_l_1, 
    std::vector<cv::Point2f> & points_r_1,
    std::vector<int> & ids) const
{
    std::vector<cv::Point2f> points_l_0_return;   
    std::vector<float> err;         
    
    cv::Size win_size = cv::Size(20,20); // Lucas-Kanade optical flow window size                                                                                          
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    // sparse iterative version of the Lucas-Kanade optical flow in pyramids
    // To do: reuse pyramids
    cv::calcOpticalFlowPyrLK(image_left_t0, image_right_t0, points_l_0, points_r_0, status0, err, 
        win_size, 3, term_crit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(image_right_t0, image_right_t1, points_r_0, points_r_1, status1, err, 
        win_size, 3, term_crit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(image_right_t1, image_left_t1, points_r_1, points_l_1, status2, err, 
        win_size, 3, term_crit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    cv::calcOpticalFlowPyrLK(image_left_t1, image_left_t0, points_l_1, points_l_0_return, status3, err, 
        win_size, 3, term_crit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);

    // remove unmatched features after circular matching
    // To do: add distance check (vertical direction)
    int idx_correction = 0;
    for (size_t i = 0; i < status3.size(); ++i)
    {  
        const cv::Point2f & pt0 = points_l_0.at(i - idx_correction);
        const cv::Point2f & pt1 = points_r_0.at(i - idx_correction);
        const cv::Point2f & pt2 = points_r_1.at(i - idx_correction);
        const cv::Point2f & pt3 = points_l_1.at(i - idx_correction);
        const cv::Point2f & pt0_r = points_l_0_return.at(i - idx_correction);
        const float max_match_y_distance = 2.0f; // pixels
        const float min_match_disparity = 2.0f; // pixels
        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
            (status2.at(i) == 0) || (pt2.x < 0) || (pt2.y < 0) ||
            (status1.at(i) == 0) || (pt1.x < 0) || (pt1.y < 0) ||
            (status0.at(i) == 0) || (pt0_r.x < 0) || (pt0_r.y < 0) ||
            (std::abs(pt0.y - pt1.y) > max_match_y_distance) || // rectified image - y values same
            (std::abs(pt2.y - pt3.y) > max_match_y_distance) ||
            (cv::norm(pt0 - pt3)) < min_match_disparity) // keep only high disparity points
        {
            points_l_0.erase(points_l_0.begin() + (i - idx_correction));
            points_r_0.erase(points_r_0.begin() + (i - idx_correction));
            points_r_1.erase(points_r_1.begin() + (i - idx_correction));
            points_l_1.erase(points_l_1.begin() + (i - idx_correction));
            points_l_0_return.erase(points_l_0_return.begin() + (i - idx_correction));
            ids.erase(ids.begin() + (i - idx_correction));
            idx_correction++;
        }
    }
}

void VIONode::run_gtsam(const std::vector<cv::Point2f> & features_l1, 
    const std::vector<cv::Point2f> & features_r1,
    const std::vector<int> & ids)
{
    // add latest stereo factor and run gtsam
    std::vector<StereoFeature> stereo_features;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        StereoFeature stereo_feature(features_l1[i].x, features_r1[i].x, features_l1[i].y, ids[i]);
        stereo_features.push_back(stereo_feature);
    }
    vio_estimator.stereo_update(stereo_features);
}

// GTSAM pose to opencv matrices
void VIONode::gtsam_to_open_cv_pose(const Pose3 & gtsam_pose, cv::Mat& R_wb, cv::Mat& t_wb) const 
{
    Rot3 R = gtsam_pose.rotation();
    Point3 t = gtsam_pose.translation();
    R_wb = (cv::Mat_<double>(3,3) << R.r1().x(), R.r1().y(), R.r1().z(), 
        R.r2().x(), R.r2().y(), R.r2().z(), 
        R.r3().x(), R.r3().y(), R.r3().z());
    t_wb = (cv::Mat_<double>(3,1) << t.x(), t.y(), t.z());
}

void VIONode::reset_feature_set()
{
    feature_set.features_l.clear();
    feature_set.features_r.clear();
    feature_set.ids.clear();
}

void VIONode::add_new_landmarks(const std::vector<cv::Point2f> & features_l0, 
    const std::vector<cv::Point2f> & features_r0, 
    const std::vector<cv::Point3d> & points3D_w0, 
    const std::vector<int> & ids)
{
    // add new landmarks and their stereo factors
    std::vector<StereoFeature> stereo_features;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        Point3 pt(points3D_w0[i].x, points3D_w0[i].y, points3D_w0[i].z);
        vio_estimator.add_landmark_estimate(pt, ids[i]);
        StereoFeature stereo_feature(features_l0[i].x, features_r0[i].x, features_l0[i].y, ids[i]);
        stereo_features.push_back(stereo_feature);
    }
    vio_estimator.add_stereo_factors(stereo_features);
}


void VIONode::feature_tracking(const cv::Mat & image_left, const cv::Mat & image_right)
{
    // check if this is the first image
    if (!frame_id)
    {
        image_left_t0 = image_left;
        image_right_t0 = image_right;
        frame_id++;
        return;
    } 

    // store new image
    image_left_t1 = image_left;
    image_right_t1 = image_right;
    frame_id++;

    // if ! enough features in current feature set, replace all of them
    bool replaced_features = false;
    const int min_num_features = 100;
    if (feature_set.get_size() < min_num_features)
    {
        replace_all_features();
        replaced_features = true;
        std::cout << "[vio_gtsam_node]: replaced features." << endl;
    }

    // track features between frames into next position and updates feature_set positions
    std::vector<cv::Point2f> & features_l0 = feature_set.features_l;
    std::vector<cv::Point2f> & features_r0 = feature_set.features_r;
    std::vector<cv::Point2f> features_l1;
    std::vector<cv::Point2f> features_r1;
    circular_matching(features_l0, features_r0, features_l1, features_r1, feature_set.ids);

    std::cout << "[vio_gtsam_node]: tracking " << features_l0.size() << " features." << endl;

    // if detected new features, add them as landmarks to gtsam graph
    if (replaced_features)
    {
        cv::Mat features_3D_l0;
        triangulate_features(feature_set.features_l, feature_set.features_r, features_3D_l0);
        Pose3 gtsam_pose = vio_estimator.get_pose(); // a lot of this is bad, should use IMU prediction instead
        // see the large example of stereovo
        cv::Mat R_wb;
        cv::Mat t_wb;
        gtsam_to_open_cv_pose(gtsam_pose, R_wb, t_wb);
        vector<cv::Point3d> features_3D_w0 = transform_to_world(features_3D_l0, R_wb, t_wb);

        // add new landmarks (stereo features at previous image) 
        add_new_landmarks(features_l0, features_r0, features_3D_w0, feature_set.ids);
    }

    // update feature_set
    feature_set.features_l = features_l1;
    feature_set.features_r = features_r1;

    // set previous image as new image
    image_left_t0 = image_left_t1;
    image_right_t0 = image_right_t1;
}

void VIONode::stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
{
    if (state == STATE::RUNNING_VIO)
    {
        cv::Mat l = ros_img_to_cv_img(image_left);
        cv::Mat r = ros_img_to_cv_img(image_right);
        feature_tracking(l, r);
        if (frame_id > 1)
        {
            run_gtsam(feature_set.features_l, feature_set.features_r, feature_set.ids);
            cout << "[vio_gtsam_node]: running gtsam itr #" << frame_id - 1 << endl;
        }
    } 
}

void VIONode::initialize_estimator(double qw, double qx, double qy, double qz)
{
    // initialize gtsam estimator
    vio_estimator.initialize_pose(qw, qx, qy, qz, 0.0, 0.0, 0.0); // initialize pose (attitude important)
    vio_estimator.init();

    // obtain body-camera extrinsics from estimator
    Point3 r1 = vio_estimator.camera_params.R_bc.r1();
    Point3 r2 = vio_estimator.camera_params.R_bc.r2();
    Point3 r3 = vio_estimator.camera_params.R_bc.r3();
    R_bc = (cv::Mat_<double>(3,3) << r1.x(), r1.y(), r1.z(),
        r2.x(), r2.y(), r2.z(),
        r3.x(), r3.y(), r3.z());
    t_bc = (cv::Mat_<double>(3,1) << vio_estimator.camera_params.t_bc.x(), 
        vio_estimator.camera_params.t_bc.y(), 
        vio_estimator.camera_params.t_bc.z());
}

void VIONode::imu_callback(const sensor_msgs::Imu::ConstPtr& msg)
{
    geometry_msgs::Vector3 w = msg->angular_velocity;
    geometry_msgs::Vector3 f = msg->linear_acceleration;
    if (state == STATE::IMU_INITIALIZATION)
    {
        attitude_initializer.add_data(f.x, f.y, f.z, w.x, w.y, w.z);
        num_imu_init_measurements++;
        // check if initialization completed
        if (num_imu_init_measurements == num_imu_init_measurements_required)
        {
            state = STATE::RUNNING_VIO;
            double qw, qx, qy, qz;
            attitude_initializer.compute_orientation(num_imu_init_measurements_required, qw, qx, qy, qz);
            initialize_estimator(qw, qx, qy, qz);
            cout << "[vio_gtsam_node]: initial quaternion: " << qw << " " << qx << " " << qy << " " << qz << endl;
        }
    } else if (state == STATE::RUNNING_VIO)
    {
        vio_estimator.integrate_imu(f.x, f.y, f.z, w.x, w.y, w.z);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vio_gtsam_node");

    ros::NodeHandle n;

    VIONode vio_node(n);

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, vio_node.img_left, vio_node.img_queue_size);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, vio_node.img_right, vio_node.img_queue_size);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(vio_node.img_queue_size), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&VIONode::stereo_callback, &vio_node, _1, _2));

    ros::spin();
    return 0;
}