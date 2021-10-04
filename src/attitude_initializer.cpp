/**
 * @file attitude_initializer.cpp
 * @brief Initializes vio attitude by averaging accelerometer measurements of stationary system.
 * @date October 4, 2021
 * @author Haidar Jamal
 */

#include "attitude_initializer.hpp"
#include <eigen3/Eigen/Dense> // attitude initializer

void AttitudeInitializer::add_data(double ax, double ay, double az, double wx, double wy, double wz)
{
    // add to matrix
    sum_accel -= Eigen::Vector3d(ax, ay, az);
    sum_gyro += Eigen::Vector3d(wx, wy, wz); // unused (for biases)
}

void AttitudeInitializer::compute_orientation(int num_measurements, double & qw, double & qx, double & qy, double & qz) const
{
    // compute initial orientation
    Eigen::Vector3d g_b = sum_accel / num_measurements;

    // initial roll (phi) and pitch (theta)
    double phi = atan2(-g_b[1],-g_b[2]);
    double theta = atan2(g_b[0], sqrt(g_b[1]*g_b[1] + g_b[2]*g_b[2]));

    // set initial yaw to zero
    double psi = 0;

    // q is navigation to body transformation: R_bi
    // YPR: R_ib = R(yaw)R(pitch)R(Roll)
    Eigen::Quaternion<double> q = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ())
    * Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX());

    qw = q.w();
    qx = q.x();
    qy = q.y();
    qz = q.z();
}