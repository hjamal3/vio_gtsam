#define BOOST_BIND_GLOBAL_PLACEHOLDERS 0 // boost warning

#include "features.hpp"

#include <opencv2/video/tracking.hpp> // KLT
#include <opencv2/calib3d/calib3d.hpp> // triangulatePoints

#include <iostream> // cout

void Features::init(double fx, double fy, double cx, double cy, double b, int _img_width, int _img_height)
{
    proj_mat_l = (cv::Mat_<double>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    proj_mat_r = (cv::Mat_<double>(3, 4) << fx, 0., cx, b, 0., fy, cy, 0., 0,  0., 1., 0.);
    img_width = _img_width;
    img_height = _img_height;
}

void Features::triangulate_features(const std::vector<cv::Point2f> & features_left, 
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

void Features::detect_new_features(std::vector<cv::Point2f> & features, 
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

void Features::bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
    const std::vector<int> & strengths)
{
    // sort features by strengths
    const int num_buckets_per_axis = 10;
    const int num_features_per_bucket = 3;
    const int num_features_min = 200;
    const int dim_bucket_x = img_width/num_buckets_per_axis;
    const int dim_bucket_y = img_height/num_buckets_per_axis;

    cout << "[features]: dim_bucket_x & y: " << dim_bucket_x << " " << dim_bucket_y << endl;

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

    cout << "[features]: this many features post bucketing: " << idx.size() << endl;

    // clear everything and start over
    reset_feature_set();
    feature_set.features_l.resize(idx.size());
    feature_set.ids.resize(idx.size());
    for (size_t i = 0; i < idx.size(); ++i)
    {
        feature_set.features_l.push_back(features[idx[i]]);
        feature_set.ids.push_back(feature_id);
        feature_id++;
    }
}

void Features::circular_matching(std::vector<cv::Point2f> & points_l_0, 
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
    int idx_correction = 0;
    for (size_t i = 0; i < status3.size(); ++i)
    {  
        const cv::Point2f & pt0 = points_l_0.at(i - idx_correction);
        const cv::Point2f & pt1 = points_r_0.at(i - idx_correction);
        const cv::Point2f & pt2 = points_r_1.at(i - idx_correction);
        const cv::Point2f & pt3 = points_l_1.at(i - idx_correction);
        const cv::Point2f & pt0_r = points_l_0_return.at(i - idx_correction);

        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
            (status2.at(i) == 0) || (pt2.x < 0) || (pt2.y < 0) ||
            (status1.at(i) == 0) || (pt1.x < 0) || (pt1.y < 0) ||
            (status0.at(i) == 0) || (pt0_r.x < 0) || (pt0_r.y < 0) ||
            (std::abs(pt0.y - pt1.y) > max_stereo_y_distance) || // rectified image - y values same
            (cv::norm(pt0 - pt3)) < min_match_disparity || // keep only high disparity in a match
            (std::abs(pt0.x - pt1.x) < min_stereo_x_disparity) || // keep only high disparity in a stereo pair
            (cv::norm(pt0-pt0_r)) > return_distance) 
        {
            cout << pt0 << endl;
            cout << pt1 << endl;
            cout << " " << endl;
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

void Features::reset_feature_set()
{
    feature_set.features_l.clear();
    feature_set.features_r.clear();
    feature_set.features_l_prev.clear();
    feature_set.features_r_prev.clear();
    feature_set.ids.clear();
}

void Features::feature_tracking(const cv::Mat & image_left, const cv::Mat & image_right, bool & replaced_features)
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
    replaced_features = false;
    const int min_num_features = 100;
    if (feature_set.get_size() < min_num_features)
    {
        cout << "[features]: replacing all features." << endl;

        // detect features in image
        std::vector<cv::Point2f> features;
        std::vector<int> strengths;
        detect_new_features(features, strengths);

        cout << "[features]: detected " << features.size() << " features." << endl;

        // bucket features in image
        bucket_and_update_feature_set(features, strengths);

        replaced_features = true;
    }

    cout << "[features]: " << feature_set.features_l.size() << " features going into circular matching" << endl;

    // track features between frames into next position and updates feature_set positions
    std::vector<cv::Point2f> & features_l0 = feature_set.features_l;
    std::vector<cv::Point2f> & features_r0 = feature_set.features_r;
    std::vector<cv::Point2f> features_l1;
    std::vector<cv::Point2f> features_r1;
    circular_matching(features_l0, features_r0, features_l1, features_r1, feature_set.ids);

    std::cout << "[features]: post circular_matching: " << features_l0.size() << " features." << endl;

    // update feature_set
    feature_set.features_l_prev = feature_set.features_l;
    feature_set.features_r_prev = feature_set.features_r;
    feature_set.features_l = features_l1;
    feature_set.features_r = features_r1;

    // set previous image as new image
    image_left_t0 = image_left_t1;
    image_right_t0 = image_right_t1;
}