#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ImageSimulator {
public:
    ImageSimulator();
    ~ImageSimulator();

    // 生成晶圆图像
    // size: 图像大小
    // shiftX, shiftY: 亚像素偏移量 (Truth)
    // noiseLevel: 噪声等级
    // angle: 旋转角度
    cv::Mat generateWaferImage(int size, double shiftX, double shiftY, double noiseLevel, double angle);
};