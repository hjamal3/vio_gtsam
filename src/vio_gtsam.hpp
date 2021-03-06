#ifndef VIO_GTSAM_H
#define VIO_GTSAM_H

#include <gtsam/geometry/Cal3_S2Stereo.h> // gtsam::Cal3_S2Stereo

#include <gtsam/geometry/Pose3.h> // gtsam::Pose3

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/StereoFactor.h> // GenericStereoFactor

#include <gtsam/nonlinear/NonlinearFactorGraph.h> // gtsam::NonlinearFactorGraph
#include <gtsam/nonlinear/GaussNewtonOptimizer.h> // gtsam::GaussNewtonParams
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

// The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// nonlinear functions around an initial linearization point, then solve the linear system
// to update the linearization point. This happens repeatedly until the solver converges
// to a consistent set of variable gtsam::Values. This requires us to specify an initial guess
// for each variable, held in a gtsam::Values container.
#include <gtsam/nonlinear/Values.h> // gtsam::Values

struct StereoFeature
{
    StereoFeature(int xl, int xr, int y, size_t landmark_id) : xl(xl), xr(xr), y(y), landmark_id(landmark_id) {}
    int xl = 0; // feature x coordinate in left image (pixel)
    int xr = 0; // feature x coordinate in right image
    int y = 0;  // feature y coordinate in both images (input image is rectified)
    size_t landmark_id = -1;
};

struct CameraParameters
{
    // Stereo camera: double fx, double fy, double s, double u0, double v0, double b (baseline)
    static constexpr double fx = 220.44908;
    static constexpr double fy = 220.44908;
    static constexpr double s = 0.0; // skew
    static constexpr double cx = 222.01352; // offset x
    static constexpr double cy = 146.41498; // offset y
    static constexpr double b = 0.04979077254; // baseline in meters

    // Stereo camera intrinsic calibration object
    gtsam::Cal3_S2Stereo::shared_ptr K;

    // Extrinsic calibration object (p_body = body_P_sensor*p_sensor)
    gtsam::Pose3 body_P_sensor;
    const gtsam::Rot3 R_bc = gtsam::Rot3::RzRyRx(M_PI_2, 0.0, -M_PI_2-0.401426);
    const gtsam::Point3 t_bc = gtsam::Point3(0.25, -0.10, 1.0);

    // noise model for stereo points
    const gtsam::noiseModel::Isotropic::shared_ptr stereo_model = gtsam::noiseModel::Isotropic::Sigma(3,1);

    void init_params();
};

struct IMUParameters
{
    // noise parameters for gyroscope and accelerometer. Derived from Allan Variance curves.
    static constexpr double accel_noise_sigma = 0.0003924;
    static constexpr double gyro_noise_sigma = 0.000205689024915;
    static constexpr double accel_bias_rw_sigma = 0.004905;
    static constexpr double gyro_bias_rw_sigma = 0.000001454441043;
    
    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> preint_params;
    const double integration_sigma = 1e-8; // magic number from Carlone
    const double bias_acc_omega_int = 1e-5; // magic number from Carlone
    
    gtsam::imuBias::ConstantBias prior_imu_bias; // assume zero initial bias

    std::shared_ptr<gtsam::PreintegrationType> imu_preintegrated_;

    // noise model for bias. ex: xk+1 = Fxk + w, w ~ N(0, I*1e-3)
    const double sigma_bias = 1e-3; 
    gtsam::noiseModel::Diagonal::shared_ptr bias_noise_model = gtsam::noiseModel::Isotropic::Sigma(6,sigma_bias);

    double dt = 0.008; // sampling time (s)

    static constexpr double g = 9.80665;

    void init_params();

};

class VIOEstimator
{
public:
    // initialization
    void init_vio();
    
    void add_landmark_estimate(gtsam::Point3 l, size_t landmark_id);

    void add_odometry_factor(gtsam::Pose3 pose, size_t pose_id_prev, size_t pose_id_next);

    void integrate_imu(double ax, double ay, double az, double wx, double wy, double wz);

    void stereo_vio_update(const std::vector<StereoFeature> & stereo_features);

    void add_stereo_factors(const std::vector<StereoFeature> & stereo_features);

    void initialize_pose(double qw, double qx, double qy, double qz, double x, double y, double z);

    void test_odometry_plus_loop_closure();

    size_t get_pose_id() const { return latest_pose_id; }

    gtsam::Pose3 get_pose() const { return prev_pose; }
    gtsam::Vector3 get_vel() const { return prev_vel; }

    // Camera parameters 
    CameraParameters camera_params;

private:
    // Factor graph
    gtsam::NonlinearFactorGraph graph;

    // Initial estimate (to be added to)
    gtsam::Values initial_estimate;

    // Gauss-Newton nonlinear optimizer
    gtsam::GaussNewtonParams gn_parameters;
    gtsam::LevenbergMarquardtParams lm_parameters;

    // Stop iterating once the change in error between steps is less than this value
    static constexpr double relative_error_tol = 1e-5;

    // Do not perform more than N iteration steps
    static constexpr int max_iterations = 100;

    // Print output 
    static constexpr bool print_output = true;

    // IMU parameters
    IMUParameters imu_params;

    // Latest pose
    gtsam::Pose3 prev_pose;
    gtsam::Vector3 prev_vel;
    gtsam::NavState prev_state;
    gtsam::imuBias::ConstantBias prev_bias;

    // initial uncertainty CHANGE THIS TO 3D
    gtsam::noiseModel::Diagonal::shared_ptr pose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.0, 0.0, 0.1, 0.3, 0.3, 0.0).finished());
    gtsam::noiseModel::Diagonal::shared_ptr velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1); // m/s

    // latest pose id, updates every time a new image is received
    size_t latest_pose_id = 1; 

    // Odometry model (pose estimate between frames using PnP or ICP etc): rad, rad, rad, m, m, m
    const double angular_var = 0.1;
    const double trans_var = 0.2;
    gtsam::noiseModel::Diagonal::shared_ptr odometry_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) 
        << angular_var, angular_var, angular_var, trans_var, trans_var, trans_var).finished());

    void optimize_graph_and_update_state_vio();

    void add_pose_estimate(gtsam::Pose3 pose, size_t pose_id);

    void add_velocity_estimate(gtsam::Vector3 vel, size_t pose_id);

    void add_bias_estimate(gtsam::imuBias::ConstantBias bias, size_t pose_id);

    void create_imu_factor();

    void add_stereo_factor(int xl, int xr, int y, size_t pose_id, size_t landmark_id);

    // Print covariances
    void print_covariances(const gtsam::NonlinearFactorGraph & graph, const gtsam::Values & result) const;
};

#endif 