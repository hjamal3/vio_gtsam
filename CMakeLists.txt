cmake_minimum_required(VERSION 3.0.2)
project(vio_gtsam CXX)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  sensor_msgs
  tf
  cv_bridge
  image_transport
  message_filters
)

find_package(Boost)
include_directories(
    ${BOOST_INCLUDE_DIR}
)

find_package(GTSAM REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

## System dependencies are found with CMake's conventionsg
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 3.3.1 REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_library(vio_gtsam SHARED src/vio_gtsam.cpp)
target_link_libraries(vio_gtsam gtsam)

add_library(attitude_initializer src/attitude_initializer.cpp)
target_link_libraries(attitude_initializer Eigen3::Eigen)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES stereo_visual_odometry
#  CATKIN_DEPENDS geometry_msgs roscpp std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/stereo_visual_odometry_node.cpp)
add_executable(vio_gtsam_node src/vio_gtsam_node.cpp)
add_dependencies(vio_gtsam_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vio_gtsam_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   gtsam
   vio_gtsam
   attitude_initializer
)
