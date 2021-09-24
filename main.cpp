#define BOOST_BIND_GLOBAL_PLACEHOLDERS 0 
/**
 * @file vio_gtsam.cpp
 * @brief A stereo vio sample
 * @date Sept 24, 2021
 * @author Haidar Jamal
 */
#include <gtsam/geometry/Pose3.h> // Pose3

// We will use simple integer Keys to refer to the robot poses.
#include <gtsam/inference/Key.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/StereoFactor.h> // GenericStereoFactor
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/nonlinear/GaussNewtonOptimizer.h> // GaussNewtonParams

#include <gtsam/nonlinear/Marginals.h> // Marginals

// The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// nonlinear functions around an initial linearization point, then solve the linear system
// to update the linearization point. This happens repeatedly until the solver converges
// to a consistent set of variable values. This requires us to specify an initial guess
// for each variable, held in a Values container.
#include <gtsam/nonlinear/Values.h> // Values

#include <gtsam/geometry/Cal3_S2Stereo.h> // Cal3_S2Stereo


using namespace std;
using namespace gtsam;

struct CameraParameters
{
    // Stereo camera: double fx, double fy, double s, double u0, double v0, double b (baseline)
    static constexpr double fx = 1000;
    static constexpr double fy = 1000;
    static constexpr double s = 0.0; // skew
    static constexpr double cx = 320; // offset x
    static constexpr double cy = 240; // offset y
    static constexpr double b = 0.2; // baseline in meters

    // Stereo camera calibration object
    Cal3_S2Stereo::shared_ptr K;

    // noise model for stereo points
    const noiseModel::Isotropic::shared_ptr stereo_model = noiseModel::Isotropic::Sigma(3,1);

    void init_params();
};

void CameraParameters::init_params()
{
    // Set up camera calibration matrix
    K = boost::make_shared<Cal3_S2Stereo>(Cal3_S2Stereo(fx, fy, s, cx, cy, b));

}

struct IMUParameters
{
    // noise parameters for gyroscope and accelerometer. Derived from Allan Variance curves.
    static constexpr double accel_noise_sigma = 0.0003924;
    static constexpr double gyro_noise_sigma = 0.000205689024915;
    static constexpr double accel_bias_rw_sigma = 0.004905;
    static constexpr double gyro_bias_rw_sigma = 0.000001454441043;
    
    boost::shared_ptr<PreintegratedCombinedMeasurements::Params> preint_params;

    void init_params();

};

void IMUParameters::init_params()
{
    preint_params = PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);

    // PreintegrationBase params:
    preint_params->accelerometerCovariance = Matrix33::Identity(3,3) * pow(accel_noise_sigma,2); // acc white noise in continuous
    preint_params->integrationCovariance = Matrix33::Identity(3,3)* 1e-8; // integration uncertainty continuous
    // should be using 2nd order integration
    // PreintegratedRotation params:
    preint_params->gyroscopeCovariance = Matrix33::Identity(3,3) * pow(gyro_noise_sigma,2); // gyro white noise in continuous
    // PreintegrationCombinedMeasurements params:
    preint_params->biasAccCovariance = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2); // acc bias in continuous
    preint_params->biasOmegaCovariance = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2); // gyro bias in continuous
    preint_params->biasAccOmegaInt = Matrix::Identity(6,6)*1e-5;
}


class VIOEstimator
{
public:

    VIOEstimator();
    
    void init();

    void optimize_graph();

    void test_odometry_plus_loop_closure();

    size_t add_landmark(Point3 l);

    void add_stereo(int xl, int xr, int y, size_t pose_id, size_t landmark_id);

    void add_pose_estimate(Pose3 pose);

    size_t get_pose_id() const { return pose_id; }

    Pose3 get_current_pose() const { return current_pose; }

private:

    // latest pose
    Pose3 current_pose;

    // latest pose id
    size_t pose_id = 0;
    size_t landmark_id = 0;

    // Factor graph
    NonlinearFactorGraph graph;

    // Initial estimate (to be added to)
    Values initial_estimate;

    // Gauss-Newton nonlinear optimizer
    GaussNewtonParams parameters;

    // Stop iterating once the change in error between steps is less than this value
    static constexpr double relative_error_tol = 1e-5;

    // Do not perform more than N iteration steps
    static constexpr int max_iterations = 100;

    // Print output 
    static constexpr bool print_output = false;

    // Camera parameters 
    CameraParameters camera_params;

    // IMU parameters
    IMUParameters imu_params;

    // imu integration type
    //PreintegrationType *imu_preintegrated_ = std::make_shared<PreintegratedCombinedMeasurements>(p, prior_imu_bias);

    // Print covariances
    void print_covariances(const NonlinearFactorGraph & graph, const Values & result) const;

};

VIOEstimator::VIOEstimator()
{
    init();
}

void VIOEstimator::init()
{
    // initialize pose to origin. RzRyRx = Rotations around Z, Y, then X: RzRyRx(double x, double y, double z);
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0)));

    if (print_output) initial_estimate.print("\nInitial Estimate:\n");

    // Set optimizer parameters 
    // Stop iterating once the change in error between steps is less than this value
    parameters.relativeErrorTol = relative_error_tol;
    // Do not perform more than N iteration steps
    parameters.maxIterations = max_iterations;

    camera_params.init_params();
    imu_params.init_params();

}

void VIOEstimator::optimize_graph()
{
    GaussNewtonOptimizer optimizer(graph, initial_estimate, parameters);
    Values result = optimizer.optimize();

    if (print_output) result.print("Final Result:\n");

    // this frame result is set to next frame initial estimate
    initial_estimate = result;

    if (print_output) print_covariances(graph, result);

}

void VIOEstimator::print_covariances(const NonlinearFactorGraph & graph, const Values & result) const
{
    cout.precision(3);
    Marginals marginals(graph, result);
    for (size_t i = 0; i <= pose_id; ++i)
    {
        cout << "x" << i << " covariance:\n" << marginals.marginalCovariance(i) << endl;
    }
}

// add new 3D point
size_t VIOEstimator::add_landmark(Point3 l)
{
    initial_estimate.insert(landmark_id, l);
    landmark_id++;
    return landmark_id;
}

// add stereo point (assuming rectified images)
void VIOEstimator::add_stereo(int xl, int xr, int y, size_t pose_id, size_t landmark_id)
{
    graph.emplace_shared<GenericStereoFactor<Pose3,Point3>>(StereoPoint2(xl, xr, y), 
        camera_params.stereo_model, pose_id, landmark_id, camera_params.K);
}

// add new initial pose to graph
void VIOEstimator::add_pose_estimate(Pose3 pose)
{
    initial_estimate.insert(pose_id, pose);
    pose_id++;
}

// todo unit test
void VIOEstimator::test_odometry_plus_loop_closure()
{
    // 2a. Add a prior on the first pose, setting it to the origin
    // A prior factor consists of a mean and a noise model (covariance matrix)
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 0.0, 0.0, 0.1, 0.3, 0.3, 0.0).finished());
    graph.emplace_shared<PriorFactor<Pose3> >(0, Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)), priorNoise);

    // For simplicity, we will use the same noise model for odometry and loop closures
    noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.0, 0.0, 0.1, 0.2, 0.2, 0.0).finished());

    // 2b. Add odometry factors
    // Create odometry (Between) factors between consecutive poses
    Pose3 T_1(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(2.0, 0.0, 0.0));
    Pose3 T_2(Rot3::RzRyRx(0.0, 0.0, M_PI_2), Point3(2.0, 0.0, 0.0));

    graph.emplace_shared<BetweenFactor<Pose3> >(0, 1, T_1, model);
    graph.emplace_shared<BetweenFactor<Pose3> >(1, 2, T_2, model); 
    graph.emplace_shared<BetweenFactor<Pose3> >(2, 3, T_2, model);
    graph.emplace_shared<BetweenFactor<Pose3> >(3, 4, T_2, model);

    // 2c. Add the loop closure constraint
    // This factor encodes the fact that we have returned to the same pose. In real systems,
    // these constraints may be identified in many ways, such as appearance-based techniques
    // with camera images. We will use another Between Factor to enforce this constraint:
    graph.emplace_shared<BetweenFactor<Pose3> >(4, 1, T_2, model);

    if (print_output) graph.print("\nFactor Graph:\n"); // print
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, -0.2), Point3(2.3, 0.1, 0.0)));
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, M_PI_2), Point3(4.1, 0.1, 0.0)));
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, M_PI), Point3(4.0, 2.0, 0.0)));
    add_pose_estimate(Pose3(Rot3::RzRyRx(0.0, 0.0, -M_PI_2), Point3(2.1, 2.1, 0.0)));

    if (print_output) initial_estimate.print("\nInitial Estimate:\n"); // print
    optimize_graph();
}

int main(int argc, char** argv) {

    VIOEstimator vo;
    vo.test_odometry_plus_loop_closure();

    return 0;
}
