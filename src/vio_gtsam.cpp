/**
 * @file vio_gtsam.cpp
 * @brief A stereo vio sample
 * @date Sept 24, 2021
 * @author Haidar Jamal
 */

// We will use simple integer Keys to refer to the robot poses.
#include "vio_gtsam.hpp"
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h> // symbol_shorthand
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Marginals.h> // Marginals


using namespace std;
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::L; // landmark (Point3) (x,y,z)

void CameraParameters::init_params()
{
    // Set up camera calibration matrix
    K = boost::make_shared<Cal3_S2Stereo>(Cal3_S2Stereo(fx, fy, s, cx, cy, b));
}

void IMUParameters::init_params()
{
    preint_params = PreintegratedCombinedMeasurements::Params::MakeSharedU(g);

    // PreintegrationBase params:
    preint_params->accelerometerCovariance = Matrix33::Identity(3,3) * pow(accel_noise_sigma,2); // acc white noise in continuous
    preint_params->integrationCovariance = Matrix33::Identity(3,3)*integration_sigma; // integration uncertainty continuous
    // should be using 2nd order integration
    // PreintegratedRotation params:
    preint_params->gyroscopeCovariance = Matrix33::Identity(3,3) * pow(gyro_noise_sigma,2); // gyro white noise in continuous
    // PreintegrationCombinedMeasurements params:
    preint_params->biasAccCovariance = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2); // acc bias in continuous
    preint_params->biasOmegaCovariance = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2); // gyro bias in continuous
    preint_params->biasAccOmegaInt = Matrix::Identity(6,6)*bias_acc_omega_int;

    imu_preintegrated_ =
        std::make_shared<PreintegratedCombinedMeasurements>(preint_params, prior_imu_bias);
}

void VIOEstimator::init()
{
    // Add initial pose and bias estimates to Values
    // RzRyRx = Rotations around Z, Y, then X: RzRyRx(double x, double y, double z);
    add_pose_estimate(prev_pose, 0);
    add_velocity_estimate(prev_vel, 0);
    add_bias_estimate(imu_params.prior_imu_bias, 0);

    // Store previous state for the imu integration and the latest predicted outcome.
    prev_state = NavState(prev_pose, prev_vel);
    prev_bias = imuBias::ConstantBias(imu_params.prior_imu_bias);

    // Add priors for pose and imu biases to graph - mean and a noise model (covariance matrix)
    graph.add(PriorFactor<Pose3>(X(0), prev_pose, pose_prior_noise));
    graph.add(PriorFactor<Vector3>(V(0), prev_vel, velocity_noise_model));
    graph.add(PriorFactor<imuBias::ConstantBias>(B(0), imu_params.prior_imu_bias, imu_params.bias_noise_model));

    if (print_output) initial_estimate.print("\nInitial Estimate:\n");

    // Set optimizer parameters 
    // Stop iterating once the change in error between steps is less than this value
    parameters.relativeErrorTol = relative_error_tol;

    // Do not perform more than N iteration steps
    parameters.maxIterations = max_iterations;

    camera_params.init_params();
    imu_params.init_params();
}

void VIOEstimator::optimize_graph_and_update_state()
{
    GaussNewtonOptimizer optimizer(graph, initial_estimate, parameters);
    Values result = optimizer.optimize();

    if (print_output) result.print("Final Result:\n");

    // this frame result is set to next frame initial estimate
    initial_estimate = result;

    if (print_output) print_covariances(graph, result);

    // Overwrite the beginning of the preintegration for the next step.
    prev_pose = result.at<Pose3>(X(latest_pose_id));
    prev_vel = result.at<Vector3>(V(latest_pose_id));
    prev_state = NavState(prev_pose, prev_vel);
    prev_bias = result.at<imuBias::ConstantBias>(B(latest_pose_id));
    imu_params.imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
}

// add stereo factor (assuming rectified images) to graph
void VIOEstimator::add_stereo_factor(int xl, int xr, int y, size_t pose_id, size_t landmark_id)
{
    graph.emplace_shared<GenericStereoFactor<Pose3,Point3>>(StereoPoint2(xl, xr, y), 
        camera_params.stereo_model, X(pose_id), L(landmark_id), camera_params.K);
}

// add odometry factor to graph. for example from PnP perhaps. unused for now.
void VIOEstimator::add_odometry_factor(Pose3 pose, size_t pose_id_prev, size_t pose_id_next)
{
    graph.emplace_shared<BetweenFactor<Pose3>>(X(pose_id_prev), X(pose_id_next), pose, odometry_model);
}

// add new 3D point
void VIOEstimator::add_landmark_estimate(Point3 l, size_t landmark_id)
{
    initial_estimate.insert(L(landmark_id), l);
}

// add new inital pose to graph
void VIOEstimator::add_pose_estimate(Pose3 pose, size_t pose_id)
{
    initial_estimate.insert(X(pose_id), pose);
}

void VIOEstimator::add_velocity_estimate(Vector3 vel, size_t pose_id)
{
    initial_estimate.insert(V(pose_id), vel);
}

void VIOEstimator::add_bias_estimate(imuBias::ConstantBias bias, size_t pose_id)
{
    initial_estimate.insert(B(pose_id), bias);
}

void VIOEstimator::integrate_imu(double ax, double ay, double az, double wx, double wy, double wz)
{
    Vector3 a(ax, ay, az);
    Vector3 w(wx, wy, wz);
    imu_params.imu_preintegrated_->integrateMeasurement(a, w, imu_params.dt);
}

void VIOEstimator::create_imu_factor()
{
    // combine all IMU measurements into a factor
    // constructor: pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, PreintegratedCombinedMeasurements
    const PreintegratedCombinedMeasurements& preint_imu_combined =
        dynamic_cast<const PreintegratedCombinedMeasurements&>(
        *imu_params.imu_preintegrated_);

    CombinedImuFactor imu_factor(X(latest_pose_id-1), V(latest_pose_id-1),
        X(latest_pose_id), V(latest_pose_id),
        B(latest_pose_id-1), B(latest_pose_id),
        preint_imu_combined);

    graph.add(imu_factor);
}

// received a new image's stereo features
void VIOEstimator::stereo_update(const std::vector<StereoFeature> & stereo_features)
{
    // update pose_id
    latest_pose_id++;

    // add all new feature points to graph
    for (const auto & stereo_feature : stereo_features)
    {
        add_stereo_factor(stereo_feature.xl, stereo_feature.xr, stereo_feature.y, latest_pose_id, stereo_feature.landmark_id);
    }

    // add IMU factor to graph
    create_imu_factor();

    // add new state estimate
    NavState prop_state = imu_params.imu_preintegrated_->predict(prev_state, prev_bias);
    add_pose_estimate(prop_state.pose(), latest_pose_id);
    add_velocity_estimate(prop_state.v(), latest_pose_id);
    add_bias_estimate(prev_bias, latest_pose_id); // carries over

    // optimize graph and update state
    optimize_graph_and_update_state();
}

void VIOEstimator::initialize_pose(double qw, double qx, double qy, double qz,
    double x, double y, double z)
{
    prev_pose = Pose3(Rot3::Quaternion(qw, qx, qy, qz), Point3(x, y, z));
}

void VIOEstimator::print_covariances(const NonlinearFactorGraph & graph, const Values & result) const
{
    cout.precision(3);
    Marginals marginals(graph, result);
    for (size_t i = 0; i < latest_pose_id; ++i)
    {
        cout << "x" << i << " covariance:\n" << marginals.marginalCovariance(X(i)) << endl;
    }
}

// todo unit test
void VIOEstimator::test_odometry_plus_loop_closure()
{
    // 2b. Add odometry factors
    // Create odometry (Between) factors between consecutive poses
    Pose3 T_1(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(2.0, 0.0, 0.0));
    Pose3 T_2(Rot3::RzRyRx(0.0, 0.0, M_PI_2), Point3(2.0, 0.0, 0.0));

    add_odometry_factor(T_1, 0, 1);
    add_odometry_factor(T_2, 1, 2);
    add_odometry_factor(T_2, 2, 3);
    add_odometry_factor(T_2, 3, 4);

    // 2c. Add the loop closure constraint
    // This factor encodes the fact that we have returned to the same pose. In real systems,
    // these constraints may be identified in many ways, such as appearance-based techniques
    // with camera images. We will use another Between Factor to enforce this constraint:
    add_odometry_factor(T_2, 4, 1);

    if (print_output) graph.print("\nFactor Graph:\n"); // print
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, -0.2), Point3(2.3, 0.1, 0.0)), 1);
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, M_PI_2), Point3(4.1, 0.1, 0.0)), 2);
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, M_PI), Point3(4.0, 2.0, 0.0)), 3);
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, -M_PI_2), Point3(2.1, 2.1, 0.0)), 4);

    if (print_output) initial_estimate.print("\nInitial Estimate:\n"); // print
    optimize_graph_and_update_state();
}