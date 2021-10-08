/**
 * @file attitude_initializer.hpp
 * @brief Initializes vio attitude by averaging accelerometer measurements of stationary system.
 * @date October 4, 2021
 * @author Haidar Jamal
 */


#ifndef ATTITUDE_INITIALIZER_H
#define ATTITUDE_INITIALIZER_H
#endif

#include <eigen3/Eigen/Core> // attitude initializer

struct AttitudeInitializer
{
    // data storage elements
    Eigen::Vector3d sum_accel;
    Eigen::Vector3d sum_gyro;

    void add_data(double ax, double ay, double az, double wx, double wy, double wz);
    void compute_orientation(int num_measurements, double & qw, double & qx, double & qy, double & qz) const;
};

