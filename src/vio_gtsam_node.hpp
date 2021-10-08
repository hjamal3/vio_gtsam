#ifndef VIO_NODE_H
#define VIO_NODE_H
#endif

#include "vio_gtsam.hpp"
#include "attitude_initializer.hpp"

#include "ros/ros.h"
#include "sensor_msgs/Image.h" // sensor_msgs::ImageConstPtr
#include "sensor_msgs/Imu.h" // sensor_msgs::Imu::ConstPtr
#include <opencv2/imgproc/imgproc.hpp> // cv::Mat

using namespace std;

enum STATE
{
    IMU_INITIALIZATION = 1,
    RUNNING_VIO = 2
};

// features of the latest image
struct FeatureSet
{
    std::vector<cv::Point2f> features_l;
    std::vector<cv::Point2f> features_r;
    std::vector<int> ids;
    size_t get_size() {
        return features_l.size(); 
    }
};

struct VIONode
{
public:

    VIONode(ros::NodeHandle & n);

    void stereo_callback(const sensor_msgs::ImageConstPtr& image_left, const sensor_msgs::ImageConstPtr& image_right);

    const string imu_topic = "/imu/data";
    const string img_left = "/stereo/left/image_rect";
    const string img_right = "/stereo/right/image_rect";
    const int img_queue_size = 1;

    cv::Mat proj_mat_l;
    cv::Mat proj_mat_r;

    cv::Mat image_left_t0;
    cv::Mat image_right_t0;
    cv::Mat image_left_t1;
    cv::Mat image_right_t1;
    static constexpr int img_width = 640;
    static constexpr int img_height = 480;

private:
    VIOEstimator vio_estimator;
    ros::Subscriber imu_subscriber;
    size_t frame_id = 0;

    FeatureSet feature_set;
    size_t feature_id = 0;

    STATE state = STATE::IMU_INITIALIZATION;
    int num_imu_init_measurements = 0;
    static constexpr int imu_hz = 125; 
    static constexpr int seconds_to_init_imu = 5;
    static constexpr int num_imu_init_measurements_required = imu_hz*seconds_to_init_imu;
    AttitudeInitializer attitude_initializer;

    // camera body frame transformation
    cv::Mat R_bc;
    cv::Mat t_bc;

    cv::Mat ros_img_to_cv_img(const sensor_msgs::ImageConstPtr img) const;

    vector<cv::Point3d> transform_to_world(const cv::Mat & points3D_cam, 
        const cv::Mat & R_wb,
        const cv::Mat & t_wb) const;

    void triangulate_features(const std::vector<cv::Point2f> & points_left, 
        const std::vector<cv::Point2f> & points_right, 
        cv::Mat & points3D) const;

    void detect_new_features(std::vector<cv::Point2f> & points, 
        std::vector<int> & response_strengths) const;

    void bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
        const std::vector<int> & strengths);

    void replace_all_features();

    void circular_matching(std::vector<cv::Point2f> & points_l_0, 
        std::vector<cv::Point2f> & points_r_0, 
        std::vector<cv::Point2f> & points_l_1, 
        std::vector<cv::Point2f> & points_r_1,
        std::vector<int> & ids) const;


    void run_gtsam(const std::vector<cv::Point2f> & features_l1, 
        const std::vector<cv::Point2f> & features_r1,
        const std::vector<int> & ids);

    void gtsam_to_open_cv_pose(const Pose3 & gtsam_pose, cv::Mat& R_wb,  cv::Mat& t_wb) const;

    void add_new_landmarks(const std::vector<cv::Point2f> & features_l0, 
        const std::vector<cv::Point2f> & features_r0, 
        const std::vector<cv::Point3d> & points3D_w0, 
        const std::vector<int> & ids);

    void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);

    void feature_tracking(const cv::Mat & image_left, const cv::Mat & image_right);

    void reset_feature_set();

    void initialize_estimator(double qw, double qx, double qy, double qz);

};