#pragma once
#include <opencv2/opencv.hpp>
#include "SubPixelModel.h"
#include "YoloDetector.h"

class Localization {
public:
    Localization();
    ~Localization();

    // 创建/设置模板
    void createTemplate(const cv::Mat& image, cv::Rect roi);

    // [传统] 粗定位
    cv::Point coarseLocalization(const cv::Mat& image);

    // [YOLO] 粗定位
    cv::Point coarseLocalizationYolo(const cv::Mat& image, YoloDetector* detector);

    // [精定位] 
    // 核心修改：返回值改为 cv::Point2d，x存储X方向误差，y存储Y方向误差
    cv::Point2d fineLocalization(const cv::Mat& image, cv::Point coarsePos, SubPixelModel::ModelType type);

private:
    SubPixelModel* model;
    cv::Mat templ;
};