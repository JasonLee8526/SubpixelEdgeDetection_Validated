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
    cv::Mat generateWaferImage(int size, double shiftX, double shiftY, double noiseLevel, double angle,int targetScale=50,double spChance=0.0);

    // [新增] 批量生成训练数据集 (Images + YOLO Labels)
    // count: 生成数量
    // imagesPath: 图片保存目录 (例如 "datasets/train/images")
    // labelsPath: 标签保存目录 (例如 "datasets/train/labels")
    void generateDataset(int count, std::string imagesPath, std::string labelsPath);
};