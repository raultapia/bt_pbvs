#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Trigger.h>

namespace {

constexpr double SQUARE_SIDE = 1.85;

std::vector<cv::Point2f> get_corners_square(cv::Mat &img) {
  /***--- Color thresholding ---***/
  cv::Mat mask;
  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(35, 100, 100), cv::Scalar(85, 255, 255), mask);

  /***--- Draw mask ---***/
  cv::Mat overlay;
  cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGR);
  overlay.setTo(cv::Scalar(255, 255, 255), mask);
  const double alpha = 0.5;
  cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

  /***--- Find contours ---***/
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  /***--- Sort by area ---***/
  std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
    return cv::contourArea(a) > cv::contourArea(b);
  });

  /***--- Get corners ---***/
  std::vector<cv::Point2f> approx;
  for(const std::vector<cv::Point> &contour : contours) {
    cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
    if(approx.size() == 4) {
      for(const cv::Point &point : approx) {
        cv::circle(img, point, 10, cv::Scalar(127, 0, 255), -1);
      }
      for(size_t i = 0; i < 4; ++i) {
        cv::line(img, approx[i], approx[(i + 1) % 4], cv::Scalar(255, 128, 0), 2);
      }
      return approx;
    }
  }
  return {};
}

cv::Point find_centroid(const std::vector<cv::Point2f> &points) {
  cv::Point center(0, 0);
  for(const auto &point : points) {
    center.x += point.x;
    center.y += point.y;
  }
  center.x /= points.size();
  center.y /= points.size();
  return center;
}

inline double calculate_angle(const cv::Point2f &point, const cv::Point2f &center) {
  return atan2(point.y - center.y, point.x - center.x);
}

void sort_points_clockwise(std::vector<cv::Point2f> &points) {
  auto topLeft = *std::min_element(points.begin(), points.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
  });
  cv::Point center = find_centroid(points);
  std::sort(points.begin(), points.end(), [&center, &topLeft](const cv::Point2f &a, const cv::Point2f &b) {
    double angleA = calculate_angle(a, center);
    double angleB = calculate_angle(b, center);
    if(a == topLeft) {
      angleA = -1;
    }
    if(b == topLeft) {
      angleB = -1;
    }
    return angleA < angleB;
  });
}

inline Eigen::Matrix4d get_transformation_matrix(const cv::Mat &rvec, const cv::Mat &tvec) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  const Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Map(R.ptr<double>());
  const Eigen::Vector3d translation(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();
  transformationMatrix.block<3, 3>(0, 0) = rotationMatrix;
  transformationMatrix.block<3, 1>(0, 3) = translation;
  return transformationMatrix;
}

inline geometry_msgs::PoseStamped matrix_to_pose_stamped(const Eigen::Matrix4d &transform) {
  geometry_msgs::PoseStamped pose;

  pose.pose.position.x = transform(0, 3);
  pose.pose.position.y = transform(1, 3);
  pose.pose.position.z = transform(2, 3);

  const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
  Eigen::Quaterniond quat(rotation);
  pose.pose.orientation.x = quat.x();
  pose.pose.orientation.y = quat.y();
  pose.pose.orientation.z = quat.z();
  pose.pose.orientation.w = quat.w();

  pose.header.frame_id = "/cam";
  pose.header.stamp = ros::Time::now();
  return pose;
}
} // namespace

class Potter {
public:
  Potter() {
    imageSub_ = nh_.subscribe("/webcam/image_raw", 1, &Potter::imageCallback, this);
    altitudeSub_ = nh_.subscribe("/mavros/global_position/rel_alt", 1, &Potter::altitudeCallback, this);
    cameraInfoSub_ = nh_.subscribe("/webcam/camera_info", 1, &Potter::cameraInfoCallback, this);
    imagePub_ = nh_.advertise<sensor_msgs::Image>("/potter/debug", 1);
    featuresPub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/potter/features", 1);
    refPosePub_ = nh_.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 1);
    runSrv_ = nh_.advertiseService("/potter/run", &Potter::run, this);
  }

  bool run(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    running_ = false;
    res.success = true;
    return true;
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg) {
    fx_ = msg->K[0];
    fy_ = msg->K[4];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
    for(int i = 0; i < 5; i++) {
      d_[i] = msg->D[i];
    }
  }

  void altitudeCallback(const std_msgs::Float64ConstPtr &msg) {
    altitude_ = msg->data;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    if(running_) {
      return;
    }

    /***--- In ---***/
    cv::Mat img = (cv_bridge::toCvCopy(msg, msg->encoding)->image);

    /***--- Checks ---***/
    if(altitude_ < 0.5) {
      ROS_WARN("Altitude lower than 0.5 meters.");
      return;
    }
    if(fx_ < 0 || fy_ < 0 || cx_ < 0 || cy_ < 0) {
      ROS_WARN("Camera not configured.");
      return;
    }

    /***--- Corner detection ---***/
    std::vector<cv::Point2f> corners = get_corners_square(img);
    if(corners.size() != 4) {
      imagePub_.publish(cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::RGB8, img).toImageMsg());
      return;
    }
    sort_points_clockwise(corners);

    /***--- PnP ---***/
    std::vector<cv::Point3f> p3D;
    p3D.emplace_back(0, 0, 0);
    p3D.emplace_back(SQUARE_SIDE, 0, 0);
    p3D.emplace_back(SQUARE_SIDE, SQUARE_SIDE, 0);
    p3D.emplace_back(0, SQUARE_SIDE, 0);
    // p3D.emplace_back(0, 0, 0);
    // p3D.emplace_back(0, -SQUARE_SIDE, 0);
    // p3D.emplace_back(0, -SQUARE_SIDE, -SQUARE_SIDE);
    // p3D.emplace_back(0, 0, -SQUARE_SIDE);
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << d_[0], d_[1], d_[2], d_[3], d_[4]);
    cv::Mat rvec, tvec;
    const bool sucess = cv::solvePnP(p3D, corners, cameraMatrix, distCoeffs, rvec, tvec);
    if(!sucess) {
      return;
    }

    /***--- PBVS ---***/
    const Eigen::Matrix4d current_pose = get_transformation_matrix(rvec, tvec);
    Eigen::Matrix4d target_pose = Eigen::Matrix4d::Identity();
    target_pose(0, 3) = 0.5 * SQUARE_SIDE;
    target_pose(1, 3) = 0.5 * SQUARE_SIDE;
    target_pose(2, 3) = 1.5;
    // target_pose(0, 3) = 1.5;
    // target_pose(1, 3) = -0.5 * SQUARE_SIDE;
    // target_pose(2, 3) = -0.5 * SQUARE_SIDE;
    // target_pose(0,0) = 0.9396926; target_pose(0,1) = -0.3420202;
    // target_pose(1,0) = 0.3420202; target_pose(1,1) = 0.9396926;

    Eigen::Matrix4d Rz, Rx;
    Rz << cos(-M_PI / 2), -sin(-M_PI / 2), 0, 0,
        sin(-M_PI / 2), cos(-M_PI / 2), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    Rx << 1, 0, 0, 0,
        0, cos(-M_PI / 2), -sin(-M_PI / 2), 0,
        0, sin(-M_PI / 2), cos(-M_PI / 2), 0,
        0, 0, 0, 1;

    const Eigen::Matrix4d cmd_pose = (Rz * Rx) * (current_pose * target_pose) * (Rz * Rx).inverse();
    refPosePub_.publish(matrix_to_pose_stamped(cmd_pose));
    ROS_INFO("PBVS: Running!");
    running_ = true;

    /***--- Features ---***/
    geometry_msgs::PolygonStamped msg_features;
    msg_features.header.stamp = ros::Time::now();
    msg_features.header.frame_id = "/cam";
    for(int i = 0; i < 4; i++) {
      msg_features.polygon.points.emplace_back();
      msg_features.polygon.points.back().x = corners[i].x;
      msg_features.polygon.points.back().y = corners[i].y;
    }
    featuresPub_.publish(msg_features);

    /***--- Out ---***/
    imagePub_.publish(cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::RGB8, img).toImageMsg());
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber imageSub_;
  ros::Subscriber altitudeSub_;
  ros::Subscriber cameraInfoSub_;
  ros::Publisher imagePub_;
  ros::Publisher featuresPub_;
  ros::Publisher refPosePub_;
  ros::ServiceServer runSrv_;
  double altitude_{};
  double fx_{}, fy_{}, cx_{}, cy_{};
  std::array<double, 5> d_{};
  bool running_ = true;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "potter");
  const Potter potter;
  ros::spin();
  return 0;
}
