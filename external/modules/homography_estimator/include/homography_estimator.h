#ifndef HOMOGRAPHY_ESTIMATOR_H
#define HOMOGRAPHY_ESTIMATOR_H

#include <lms/module.h>

#include <opencv2/core/core.hpp>
#include <lms/imaging/image.h>
#include <opencv2/features2d/features2d.hpp>

#include "homography.h"

/**
 * @brief LMS module homography_estimator
 **/
class HomographyEstimator : public lms::Module {
public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;

protected:
    lms::ReadDataChannel<lms::imaging::Image> image;

    Pattern pattern;
    cv::Size patternSize;
    cv::Size outlineSize;
    std::vector<cv::Point2f> worldPoints;
    std::vector<cv::Point2f> estimatePoints;
    std::vector<cv::Point2f> outlinePoints;
    std::vector<cv::Point2f> detectedPoints;

    cv::Mat estimate;
    cv::Mat refinement;
    cv::Mat cam2world;
    cv::Mat world2cam;
    cv::Mat topView2cam;
    cv::Size topViewSize;

    bool initParameters();
    bool setPattern();
    void computePatternPoints();

    void saveHomography();

    cv::Size getSize();

    bool findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points);
    bool detectPattern(cv::Mat& img, cv::Mat& visualization, std::vector<cv::Point2f>& points);

    void computeEstimate();
    bool computeRefinement();
    void computeTopView();

    // Blob Detector
    cv::Ptr<cv::SimpleBlobDetector> blobDetector;
    cv::SimpleBlobDetector::Params getBlobDetectorParams();

public:
    void userClickCallback(int e, int x, int y, int d);
};

#endif // HOMOGRAPHY_ESTIMATOR_H
