#define _USE_MATH_DEFINES // 确保 M_PI 可用
#include <cmath>
#include "ImageSimulator.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream> // 用于写txt文件
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

ImageSimulator::ImageSimulator(int imgWidth, double physicalWidth_um) {
    UM_TO_PIX_FACTOR = static_cast<double>(imgWidth) / physicalWidth_um;
    PIX_TO_UM_FACTOR = physicalWidth_um / static_cast<double>(imgWidth);
}

// 基础绘图逻辑 (灰底-黑框-同色内芯)
cv::Mat ImageSimulator::drawBasePattern(int width, int height, double errorX_um, double errorY_um, int bgGray) {
    cv::Mat img(height, width, CV_8UC1, cv::Scalar(bgGray));

    double scale = (double)width / 800.0;
    int outerBoxSize = static_cast<int>(400 * scale);
    int innerBoxSize = static_cast<int>(150 * scale);
    int errorX_px = static_cast<int>(std::round(errorX_um * UM_TO_PIX_FACTOR));
    int errorY_px = static_cast<int>(std::round(errorY_um * UM_TO_PIX_FACTOR));

    cv::Point center(width / 2, height / 2);

    // 绘制黑色实心外框 (20)
    int blackGray = 20;
    cv::Point outerTopLeft(center.x - outerBoxSize / 2, center.y - outerBoxSize / 2);
    cv::Point outerBottomRight(center.x + outerBoxSize / 2, center.y + outerBoxSize / 2);
    cv::rectangle(img, outerTopLeft, outerBottomRight, cv::Scalar(blackGray), cv::FILLED);

    // 绘制内芯 (与背景同色)
    int innerGray = bgGray;
    cv::Point innerTopLeft(center.x - innerBoxSize / 2 + errorX_px, center.y - innerBoxSize / 2 + errorY_px);
    cv::Point innerBottomRight(center.x + innerBoxSize / 2 + errorX_px, center.y + innerBoxSize / 2 + errorY_px);
    cv::rectangle(img, innerTopLeft, innerBottomRight, cv::Scalar(innerGray), cv::FILLED);

    return img;
}

cv::Mat ImageSimulator::generateStandardWaferImage(int width, int height, double errorX_um, double errorY_um) {
    cv::Mat img = drawBasePattern(width, height, errorX_um, errorY_um, 128);
    cv::Mat blurredImg;
    cv::GaussianBlur(img, blurredImg, cv::Size(5, 5), 0);
    return blurredImg;
}

void ImageSimulator::addGaussianNoise(cv::Mat& img, double mean, double stddev) {
    cv::Mat noise(img.size(), CV_16SC1);
    cv::randn(noise, cv::Scalar(mean), cv::Scalar(stddev));
    cv::Mat img16;
    img.convertTo(img16, CV_16SC1);
    cv::add(img16, noise, img16);
    img16.convertTo(img, CV_8UC1);
}

void ImageSimulator::applyRandomRotation(cv::Mat& img, double angleDeg, int borderGray) {
    if (std::abs(angleDeg) < 0.001) return;
    cv::Point2f center((float)img.cols / 2.0f, (float)img.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    cv::warpAffine(img, img, rotMat, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(borderGray));
}

// 核心修改：生成图片的同时计算 Bounding Box
std::pair<cv::Mat, YoloLabel> ImageSimulator::generateRandomTestImage(int width, int height) {
    static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));

    // 1. 随机参数
    std::uniform_int_distribution<int> distBg(80, 180);
    int bgGray = distBg(rng);

    std::uniform_real_distribution<double> distErr(-2.0, 2.0);
    double errX = distErr(rng);
    double errY = distErr(rng);

    // 2. 绘制基础图案
    cv::Mat img = drawBasePattern(width, height, errX, errY, bgGray);

    // 3. 计算原始 Bounding Box 尺寸 (未旋转前)
    double scale = (double)width / 800.0;
    double rawBoxSize = 400.0 * scale;

    // 给框加一点 padding (例如 10%)，确保 YOLO 框住整个物体及其边缘
    double padding = 1.1;
    double paddedBoxSize = rawBoxSize * padding;

    // 4. 应用模糊和旋转
    std::uniform_int_distribution<int> distBlur(0, 1);
    int ksize = (distBlur(rng) == 0) ? 3 : 5;
    cv::GaussianBlur(img, img, cv::Size(ksize, ksize), 0);

    std::uniform_real_distribution<double> distAngle(-2.0, 2.0);
    double angleDeg = distAngle(rng);
    applyRandomRotation(img, angleDeg, bgGray);

    std::uniform_real_distribution<double> distNoise(5.0, 20.0);
    double noiseStd = distNoise(rng);
    addGaussianNoise(img, 0, noiseStd);

    // 5. 计算旋转后的 Bounding Box (AABB)
    // 旋转中心的坐标 (归一化后是 0.5, 0.5)
    // 旋转后，一个正方形的外接矩形宽高计算公式：
    // NewW = W * |cos(theta)| + H * |sin(theta)|
    // 由于是正方形 W=H，公式简化为：NewSize = Size * (|cos| + |sin|)
    double angleRad = angleDeg * M_PI / 180.0;
    double newSize = paddedBoxSize * (std::abs(std::cos(angleRad)) + std::abs(std::sin(angleRad)));

    // 6. 生成 YOLO 标签
    YoloLabel label;
    label.class_id = 0; // 只有一类
    label.x_center = 0.5; // 因为我们在图像中心绘制并围绕中心旋转，所以中心始终是 0.5
    label.y_center = 0.5;

    // 归一化宽高 (除以图像总尺寸)
    label.width = newSize / width;
    label.height = newSize / height;

    // 边界检查，防止 padding 或旋转后超出 1.0
    if (label.width > 1.0) label.width = 1.0;
    if (label.height > 1.0) label.height = 1.0;

    return { img, label };
}

void ImageSimulator::generateDataset(int n, const std::string& imagesDir, const std::string& labelsDir) {
    // 创建目录
    if (!fs::exists(imagesDir)) fs::create_directories(imagesDir);
    if (!fs::exists(labelsDir)) fs::create_directories(labelsDir);

    std::cout << "[Dataset] 开始生成 " << n << " 组数据..." << std::endl;

    for (int i = 0; i < n; ++i) {
        // 生成数据
        auto result = generateRandomTestImage(800, 600);
        cv::Mat img = result.first;
        YoloLabel label = result.second;

        // 生成唯一文件名 ID
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string id = std::to_string(now_ms) + "_" + std::to_string(i);

        // 1. 保存图片
        std::string imgPath = imagesDir + "/" + id + ".png";
        cv::imwrite(imgPath, img);

        // 2. 保存 YOLO 格式标签 txt
        std::string txtPath = labelsDir + "/" + id + ".txt";
        std::ofstream outfile(txtPath);
        if (outfile.is_open()) {
            // 格式: class_id x_center y_center width height
            outfile << label.class_id << " "
                << std::fixed << std::setprecision(6) << label.x_center << " "
                << label.y_center << " "
                << label.width << " "
                << label.height << std::endl;
            outfile.close();
        }
        else {
            std::cerr << "无法写入标签文件: " << txtPath << std::endl;
        }
    }
    std::cout << "[Dataset] 生成完成！" << std::endl;
    std::cout << "图片路径: " << imagesDir << std::endl;
    std::cout << "标签路径: " << labelsDir << std::endl;
}