#include "Localization.h"
#include "WaferConfig.h"   
#include "ImageSimulator.h"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using namespace WaferConfig;

Localization::Localization() {
    model = new SubPixelModel();
}

Localization::~Localization() {
    if (model) delete model;
}

void Localization::createTemplate(const cv::Mat& image, cv::Rect roi) {
    if (roi.area() > 0 && (roi.x + roi.width <= image.cols) && (roi.y + roi.height <= image.rows)) {
        this->templ = image(roi).clone();
    }
    else {
        ImageSimulator sim;
        this->templ = sim.generateWaferImage(WaferConfig::WAFER_SIZE, 0, 0, 0, 0);
    }
}

cv::Point Localization::coarseLocalization(const cv::Mat& image) {
    if (templ.empty()) createTemplate(image, Rect(0, 0, 0, 0));

    int result_cols = image.cols - templ.cols + 1;
    int result_rows = image.rows - templ.rows + 1;
    if (result_cols <= 0 || result_rows <= 0) return Point(0, 0);

    Mat result;
    result.create(result_rows, result_cols, CV_32FC1);
    matchTemplate(image, templ, result, TM_CCOEFF_NORMED);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    return maxLoc;
}

cv::Point Localization::coarseLocalizationYolo(const cv::Mat& image, YoloDetector* detector) {
    if (!detector) return Point(0, 0);
    Rect box = detector->detect(image);
    return box.tl();
}

cv::Point2d Localization::fineLocalization(const cv::Mat& image, cv::Point coarsePos, SubPixelModel::ModelType type) {
    Point centerPos = coarsePos + Point(WaferConfig::WAFER_SIZE / 2, WaferConfig::WAFER_SIZE / 2);

    if (centerPos.x < 0 || centerPos.x >= image.cols || centerPos.y < 0 || centerPos.y >= image.rows) {
        return Point2d(-999.0, -999.0);
    }

    int outerRadius = OUTER_BOX_SIZE / 2;
    int innerRadius = INNER_BOX_SIZE / 2;

    auto measureEdge = [&](int offset, int direction) -> double {
        Point roiCenter;
        Rect roiRect;

        // 确保 ROI 足够长，以便让矩方法计算重心时有足够的背景参考
        int searchLen = ROI_SEARCH_LEN;

        if (direction == 0) { // X方向测量
            roiCenter = centerPos + Point(offset, 0);
            roiRect = Rect(roiCenter.x - searchLen / 2, roiCenter.y - ROI_SEARCH_WID / 2,
                searchLen, ROI_SEARCH_WID);
        }
        else { // Y方向测量
            roiCenter = centerPos + Point(0, offset);
            roiRect = Rect(roiCenter.x - ROI_SEARCH_WID / 2, roiCenter.y - searchLen / 2,
                ROI_SEARCH_WID, searchLen);
        }

        roiRect = roiRect & Rect(0, 0, image.cols, image.rows);
        if (roiRect.area() == 0) return -999.0;

        Mat roi = image(roiRect);
        Mat projectionMat;
        vector<double> profile;

        // 计算投影
        if (direction == 0) reduce(roi, projectionMat, 0, REDUCE_AVG, CV_64F);
        else                reduce(roi, projectionMat, 1, REDUCE_AVG, CV_64F);

        if (projectionMat.isContinuous()) {
            profile.assign((double*)projectionMat.datastart, (double*)projectionMat.dataend);
        }
        else {
            return -999.0;
        }

        // 梯度检查
        double pMin, pMax;
        minMaxLoc(projectionMat, &pMin, &pMax);
        // 如果对比度太低，视为无效
        if ((pMax - pMin) < EDGE_GRADIENT_THRESHOLD) return -999.0;

        // 计算亚像素位置 (相对于 ROI 起点)
        // 这里的 model->calculateEdge 已经改为调用 momentMethod
        double subPixelRel = model->calculateEdge(profile, type);

        if (subPixelRel == -999.0) return -999.0;

        return (direction == 0) ? (roiRect.x + subPixelRel) : (roiRect.y + subPixelRel);
        };

    double x_out_L = measureEdge(-outerRadius, 0);
    double x_out_R = measureEdge(outerRadius, 0);
    double x_in_L = measureEdge(-innerRadius, 0);
    double x_in_R = measureEdge(innerRadius, 0);

    // Y 方向
    double y_out_T = measureEdge(-outerRadius, 1);
    double y_out_B = measureEdge(outerRadius, 1);
    double y_in_T = measureEdge(-innerRadius, 1);
    double y_in_B = measureEdge(innerRadius, 1);

    if (x_out_L < 0 || x_out_R < 0 || x_in_L < 0 || x_in_R < 0 ||
        y_out_T < 0 || y_out_B < 0 || y_in_T < 0 || y_in_B < 0) {
        return Point2d(-999.0, -999.0);
    }

    double outerCenterX = (x_out_L + x_out_R) / 2.0;
    double innerCenterX = (x_in_L + x_in_R) / 2.0;
    double errorX = innerCenterX - outerCenterX;

    double outerCenterY = (y_out_T + y_out_B) / 2.0;
    double innerCenterY = (y_in_T + y_in_B) / 2.0;
    double errorY = innerCenterY - outerCenterY;

    return Point2d(errorX, errorY);
}