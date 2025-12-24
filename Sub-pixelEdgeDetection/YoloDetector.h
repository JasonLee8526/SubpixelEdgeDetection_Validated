#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class YoloDetector {
public:
    // 构造函数：加载 ONNX 模型
    YoloDetector(const std::string& modelPath);

    // 检测函数：返回最佳置信度的物体框 (Best Box)
    // 如果未检测到，返回空 Rect(0,0,0,0)
    cv::Rect detect(const cv::Mat& image);

private:
    cv::dnn::Net net;

    // 预处理：Letterbox (保持长宽比填充，防止坐标畸变)
    cv::Mat formatToSquare(const cv::Mat& source);
};