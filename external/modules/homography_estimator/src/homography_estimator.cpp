#include "homography_estimator.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Mouse callback handler
static void onMouse(int e, int x, int y, int d, void *ptr) {
    if (e == cv::EVENT_LBUTTONDOWN ) {
        static_cast<HomographyEstimator*>(ptr)->userClickCallback(e, x, y, d);
    }
}

bool HomographyEstimator::initialize() {
    image = readChannel<lms::imaging::Image>("IMAGE");

    if(!initParameters()) {
        return false;
    }

    cv::namedWindow("homography_estimator_input");
    cv::namedWindow("homography_estimator_keypoints", cv::WINDOW_NORMAL);
    cv::namedWindow("homography_estimator_pattern", cv::WINDOW_NORMAL);
    cv::namedWindow("homography_estimator_preview");

    if(config().get<bool>("estimatePoints_points_click", true)) {
        cv::setMouseCallback("homography_estimator_input", onMouse, this);
        logger.info("estimate") << "Please click the corners in clockwise direction, beginning with bottom right";
    } else {
        auto outlineX = config().getArray<int>("estimatePoints_points_x");
        auto outlineY = config().getArray<int>("estimatePoints_points_y");
        if(outlineX.size() != 4 || outlineY.size() != 4) {
            logger.error("outline") << "outline points must contain exactly 4 points each";
        }

        auto xIt = outlineX.begin();
        auto yIt = outlineY.begin();
        for(size_t i = 0; i < 4; i++) {
            estimatePoints.emplace_back(*xIt++, *yIt++);
        }
    }

    // Choose pattern outline by hand
    outlineSize = (patternSize + cv::Size(2, 2)) * config().get<int>("outline_scale_factor", 50);
    outlinePoints.clear();
    outlinePoints.emplace_back(0, 0);
    outlinePoints.emplace_back(outlineSize.width, 0);
    outlinePoints.emplace_back(outlineSize.width, outlineSize.height);
    outlinePoints.emplace_back(0, outlineSize.height);

    // Setup blob detector
    auto blobParams = getBlobDetectorParams();
    blobDetector = cv::SimpleBlobDetector::create(blobParams);

    return true;
}

bool HomographyEstimator::deinitialize() {
    cv::destroyWindow("homography_estimator_input");
    cv::destroyWindow("homography_estimator_pattern");
    cv::destroyWindow("homography_estimator_keypoints");
    cv::destroyWindow("homography_estimator_preview");
    return true;
}

bool HomographyEstimator::cycle() {

    cv::Mat img, imgColor;
    img = image->convertToOpenCVMat();
    cv::cvtColor(img, imgColor, CV_GRAY2BGR);

    cv::imshow("homography_estimator_input", imgColor);

    if(estimatePoints.size() != 4) {
        logger.info("invalid point size: ")<<estimatePoints.size();
        cv::waitKey(config().get<int>("wait", 10));
        return true;
    }

    // Determine Homography
    if(estimate.empty()) {
            // Compute estimate
            computeEstimate();
            logger.error("estimate") << estimate;
    }
    if(!estimate.empty() && topView2cam.empty()){
        // Warp image using estimate
        cv::Mat warped, tmp, warpedColor;
        // Warp the logo image to change its perspective
        cv::warpPerspective(img, tmp, estimate, outlineSize);
        cv::equalizeHist(tmp, warped);
        cv::cvtColor(warped, warpedColor, CV_GRAY2BGR);

        //TODO warum detektieren wir zwei mal keypoints?
        // Detect keypoints
        {
            std::vector<cv::KeyPoint> keypoints;
            blobDetector->detect(warpedColor, keypoints);
            cv::Mat detection;
            cv::drawKeypoints(warped, keypoints, detection);
            cv::imshow("homography_estimator_keypoints", detection);
        }

        // Detect pattern
        bool patternDetected = detectPattern(warpedColor, warpedColor, detectedPoints); //TODO was macht das?

        // Detect Keypoints
        cv::imshow("homography_estimator_pattern", warpedColor);
        // Compute Refinement
        if(patternDetected) {
            if(computeRefinement()){
                // Compute Final homography (cam2world)
                cam2world = refinement * estimate;

                world2cam = cam2world.inv();
                logger.info("cam2world") << cam2world;
                logger.info("world2cam") << world2cam;

                // Save to file
                saveHomography();

                // Compute top view
                computeTopView();
            }
        }else{
            logger.error("pattern not detected!");
        }

    }

    if(!topView2cam.empty()) {
        // Show preview
        cv::Mat topView;
        cv::warpPerspective(img, topView, topView2cam.inv(), topViewSize);
        cv::imshow("homography_estimator_preview", topView);
    }

    cv::waitKey(config().get<int>("wait", 10));

    return true;
}

void HomographyEstimator::userClickCallback(int e, int x, int y, int d){
    (void)e;
    (void)d;
    if(estimatePoints.size() == 4) {
        logger.error("userClickCallback")<<"already got 4 points";
    }
    cv::Point2f p(x, y);
    estimatePoints.push_back(p);
    logger.info("click") << p;
}

bool HomographyEstimator::detectPattern(cv::Mat& img, cv::Mat& visualization, std::vector<cv::Point2f>& points){
    bool found = findPoints(img, points);
    drawChessboardCorners(visualization, patternSize, cv::Mat(points), found);
    return found;
}

bool HomographyEstimator::initParameters() {
    if (!setPattern()) {
        return false;
    }
    
    patternSize = cv::Size(config().get<int>("points_per_row"),
                           config().get<int>("points_per_col"));
    computePatternPoints();

    return true;
}

bool HomographyEstimator::setPattern()
{
    auto pt = config().get<std::string>("pattern", "chessboard");

    if (pt == "chessboard") {
        pattern = Pattern::CHESSBOARD;
    } else if (pt == "circles") {
        pattern = Pattern::CIRCLES;
    } else if (pt == "circles_asymmetric") {
        pattern = Pattern::CIRCLES_ASYMMETRIC;
    } else {
        logger.error("pattern") << "Invalid calibration pattern '" << pt << "'";
        return false;
    }
    return true;
}

void HomographyEstimator::computePatternPoints()
{
    worldPoints.resize(0);

    float length = config().get<float>("pattern_length", 1);
    float offset_x = config().get<float>("pattern_offset_x", 0);
    float offset_y = config().get<float>("pattern_offset_y", 0);

    switch (pattern) {
        case Pattern::CHESSBOARD:
        case Pattern::CIRCLES:
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    worldPoints.emplace_back(offset_x + i * length,
                                             offset_y + j * length);
                }
            }
            break;
        case Pattern::CIRCLES_ASYMMETRIC:
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    worldPoints.emplace_back(offset_x + i * length,
                                             offset_y + (2 * j + i % 2) * length);
                }
            }
            break;
    }
}

bool HomographyEstimator::findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points)
{
    bool success = false;
    switch (pattern) {
        case Pattern::CHESSBOARD:
            success = cv::findChessboardCorners(img, patternSize, points);
            if (success) {
                //cv::cornerSubPix(img, points, cv::Size(11,11), cv::Size(-1,-1),
                //             cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
            }
            break;
        case Pattern::CIRCLES:
            success = cv::findCirclesGrid(img, patternSize, points, cv::CALIB_CB_SYMMETRIC_GRID);
            break;
        case Pattern::CIRCLES_ASYMMETRIC:
        {
            cv::Ptr<cv::FeatureDetector> blobDetector = cv::SimpleBlobDetector::create(getBlobDetectorParams()); //TODO warum einen neuen?
            success = cv::findCirclesGrid(img, patternSize, points, cv::CALIB_CB_ASYMMETRIC_GRID, blobDetector);
        }
            break;
    }
    return success;
}

void HomographyEstimator::computeEstimate()
{
    estimate = cv::findHomography(estimatePoints, outlinePoints);
}

bool HomographyEstimator::computeRefinement(){
    if(detectedPoints.size() != worldPoints.size()){
        logger.error("computeRefinement")<<"point count doesn't match!"<<detectedPoints.size()<<" "<<worldPoints.size();
        return false;
    }
    refinement = cv::findHomography(detectedPoints, worldPoints);
    return true;
}

cv::SimpleBlobDetector::Params HomographyEstimator::getBlobDetectorParams()
{
    // Blob detector
    cv::SimpleBlobDetector::Params blobParams;
    blobParams.filterByArea         = config().get<bool>("blob_filter_by_area", blobParams.filterByArea);
    blobParams.filterByCircularity  = config().get<bool>("blob_filter_by_circularity", blobParams.filterByCircularity);
    blobParams.filterByColor        = config().get<bool>("blob_filter_by_color", blobParams.filterByColor);
    blobParams.filterByConvexity    = config().get<bool>("blob_filter_by_convexity", blobParams.filterByConvexity);
    blobParams.filterByInertia      = config().get<bool>("blob_filter_by_inertia", blobParams.filterByInertia);

    blobParams.minArea              = config().get<float>("blob_min_area", blobParams.minArea);
    blobParams.maxArea              = config().get<float>("blob_max_area", blobParams.maxArea);

    blobParams.maxThreshold         = config().get<float>("blob_max_threshold", blobParams.maxThreshold);
    blobParams.minThreshold         = config().get<float>("blob_min_threshold", blobParams.minThreshold);
    blobParams.thresholdStep        = config().get<float>("blob_threshold_step", blobParams.thresholdStep);
    return blobParams;
}

void HomographyEstimator::computeTopView()
{
    // Choose top-view image size
    topViewSize = cv::Size(512, 512);

    // Choose corner points (in real-world coordinates)
    std::vector<cv::Point2f> coordinates;
    coordinates.emplace_back(0, -1.500);
    coordinates.emplace_back(0, 1.500);
    coordinates.emplace_back(3.000, -1.500);
    coordinates.emplace_back(3.000, 1.500);

    std::vector<cv::Point2f> pixels;
    pixels.emplace_back(0, topViewSize.height);
    pixels.emplace_back(0, 0);
    pixels.emplace_back(topViewSize.width, topViewSize.height);
    pixels.emplace_back(topViewSize.width, 0);

    cv::Mat H = cv::findHomography(pixels, coordinates);

    topView2cam = world2cam * H;
    logger.info("topView2cam") << topView2cam;
}

cv::Size HomographyEstimator::getSize()
{
    return cv::Size(image->width(), image->height());
}

void HomographyEstimator::saveHomography()
{
    // Save homography
    if(!isEnableSave()) {
        logger.error() << "Command line option --enable-save was not specified";
        return;
    }

    if(cam2world.rows != 3 || cam2world.cols != 3) {
        logger.error("cam2world") << "Invalid homography matrix dimensions: " << cam2world.rows << " x " << cam2world.cols;
        return;
    }

    std::ofstream out(saveLogDir("homography_estimator") + "homography.lconf");

    out << "cam2world = ";
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            out << cam2world.at<double>(r, c);
            if( r != 2 || c != 2) { //not ending with ,
                out << ",";
            }
        }
    }
    out << std::endl;

    out << "world2cam = ";
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            out << world2cam.at<double>(r, c);
            if( r != 2 || c != 2) { //not ending with ,
                out << ",";
            }
        }
    }
    out << std::endl;
}
