#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @struct YoloLabel
 * @brief 存储 YOLO 格式的标签数据 (归一化后的中心坐标和宽高)
 */
struct YoloLabel {
    int class_id;      // 类别 ID (通常为 0)
    double x_center;   // 归一化中心 X (0-1)
    double y_center;   // 归一化中心 Y (0-1)
    double width;      // 归一化宽度 (0-1)
    double height;     // 归一化高度 (0-1)
};

/**
 * @class ImageSimulator
 * @brief 模拟生成标准及带有随机扰动的晶圆套刻标记图像，并自动生成标签。
 */
class ImageSimulator {
public:
    ImageSimulator(int imgWidth = 766, double physicalWidth_um = 50.0);

    // 生成标准图 (不带随机性，不输出标签)
    cv::Mat generateStandardWaferImage(int width, int height, double errorX_um, double errorY_um);

    /**
     * @brief 生成单张随机测试图像，并返回对应的 YOLO 标签。
     * @param width 图像宽
     * @param height 图像高
     * @return std::pair<cv::Mat, YoloLabel> 图像和对应的标签数据
     */
    std::pair<cv::Mat, YoloLabel> generateRandomTestImage(int width, int height);

    /**
     * @brief 批量生成数据集 (图片 + txt标签)。
     * @param n 生成数量
     * @param imagesDir 图片保存目录
     * @param labelsDir 标签保存目录
     */
    void generateDataset(int n, const std::string& imagesDir, const std::string& labelsDir);

    double PIX_TO_UM_FACTOR;
    double UM_TO_PIX_FACTOR;

private:
    cv::Mat drawBasePattern(int width, int height, double errorX_um, double errorY_um, int bgGray);
    void addGaussianNoise(cv::Mat& img, double mean, double stddev);
    void applyRandomRotation(cv::Mat& img, double angleDeg, int borderGray);
};