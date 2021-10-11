#include <opencv2/imgproc/imgproc.hpp> // cv::Mat

using namespace std;

// features of the latest image
struct FeatureSet
{
    std::vector<cv::Point2f> features_l;
    std::vector<cv::Point2f> features_r;
    std::vector<cv::Point2f> features_l_prev;
    std::vector<cv::Point2f> features_r_prev;
    std::vector<int> ids;
    size_t get_size() {
        return features_l.size(); 
    }
};

struct Features
{
public:
    FeatureSet feature_set;

    void init(double fx, double fy, double cx, double cy, double b, int img_width, int img_height);
    
    void triangulate_features(const std::vector<cv::Point2f> & points_left, 
        const std::vector<cv::Point2f> & points_right, 
        cv::Mat & points3D) const;

    void feature_tracking(const cv::Mat & image_left, const cv::Mat & image_right, bool & replaced_features);


private:
        // maximum vertical distance between a stereo pair (should be 0 since images rectified)
        const float max_stereo_y_distance = 2.0f; // pixels

        // minimum horizontal distance between a stereo pair
        const float min_stereo_x_disparity = 1.0f; // pixels

        // minimum motion of feature
        const float min_match_disparity = 2.0f; // pixels

        // minimum return distance 
        const float return_distance = 1.0f; // pixels

        cv::Mat image_left_t0;
        cv::Mat image_right_t0;
        cv::Mat image_left_t1;
        cv::Mat image_right_t1;

        cv::Mat proj_mat_l;
        cv::Mat proj_mat_r;

        int img_width;
        int img_height;

        size_t feature_id = 0;
        size_t frame_id = 0;

        void detect_new_features(std::vector<cv::Point2f> & points, 
            std::vector<int> & response_strengths) const;

        void bucket_and_update_feature_set(const std::vector<cv::Point2f> & features, 
            const std::vector<int> & strengths);

        void circular_matching(std::vector<cv::Point2f> & points_l_0, 
            std::vector<cv::Point2f> & points_r_0, 
            std::vector<cv::Point2f> & points_l_1, 
            std::vector<cv::Point2f> & points_r_1,
            std::vector<int> & ids) const;
        
        void reset_feature_set();

};