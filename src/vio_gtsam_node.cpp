#define BOOST_BIND_GLOBAL_PLACEHOLDERS 0 // boost warning

#include "vio_gtsam_node.hpp"

#include "message_filters/sync_policies/approximate_time.h" // message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
#include "message_filters/subscriber.h" // message_filters::Subscriber<sensor_msgs::Image>
#include <cv_bridge/cv_bridge.h> // cv_bridge::toCvCopy

#include <iostream> // cout

// going into circular matching not = output

VIONode::VIONode(ros::NodeHandle & n)
{
    imu_subscriber = n.subscribe(imu_topic, 1000, &VIONode::imu_callback,  this);

    // camera projection matrices
    const float fx = vio_estimator.camera_params.fx; 
    const float fy = vio_estimator.camera_params.fy;
    const float cx = vio_estimator.camera_params.cx;
    const float cy = vio_estimator.camera_params.cy;
    const float b = vio_estimator.camera_params.b;

    features.init(fx, fy, cx, cy, b, img_width, img_height);
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


void VIONode::stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right)
{
    if (state == STATE::RUNNING_VIO)
    {
        cv::Mat l = ros_img_to_cv_img(image_left);
        cv::Mat r = ros_img_to_cv_img(image_right);

        bool replaced_features = false;
        features.feature_tracking(l, r, replaced_features);

        // if detected new features, add them as landmarks to gtsam graph
        if (replaced_features)
        {
            cv::Mat features_3D_l0;
            features.triangulate_features(features.feature_set.features_l_prev, features.feature_set.features_r_prev, features_3D_l0);
            Pose3 gtsam_pose = vio_estimator.get_pose();
            // see the large example of stereovo
            cv::Mat R_wb;
            cv::Mat t_wb;
            gtsam_to_open_cv_pose(gtsam_pose, R_wb, t_wb);
            vector<cv::Point3d> features_3D_w0 = transform_to_world(features_3D_l0, R_wb, t_wb);

            // add new landmarks (stereo features at previous image) 
            add_new_landmarks(features.feature_set.features_l_prev, features.feature_set.features_r_prev, features_3D_w0, features.feature_set.ids);
        }

        if (frame_id > 1)
        {
            cout << "[vio_gtsam_node]: running gtsam itr #" << frame_id - 1 << endl;
            run_gtsam(features.feature_set.features_l, features.feature_set.features_r, features.feature_set.ids);
            cout << "\n[vio_gtsam_node]: *************************\n" << endl;
        }
        frame_id++;
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