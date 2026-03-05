#include "ImageSimulator.h"
#include "WaferConfig.h" 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <vector>
#include <cstdlib> 
#include <fstream>  // 用于文件写入
#include <direct.h> // 用于创建目录 _mkdir
#include <cmath>
#include <iomanip>  // 用于设置输出精度
#include <iostream>
#include <chrono>
#include <sstream>
#include <random>

using namespace cv;
using namespace std;
using namespace WaferConfig;
namespace fs = std::filesystem;

ImageSimulator::ImageSimulator() {}

ImageSimulator::~ImageSimulator() {}

Mat ImageSimulator::generateWaferImage(int size, double shiftX, double shiftY, double noiseLevel, double angle,int targetScale, double saltPepperChance) {
    // =========================================================
    // 终极修正：使用 100倍 超采样 (Ultra Super Sampling)
    // 精度从 0.1px 提升至 0.01px，消除采样混叠导致的系统误差
    // =========================================================
       
    // =========================================================
    // 1. [新增] 随机亮度与对比度控制
    // =========================================================
    // 为了模拟不同的光照环境，我们不再使用固定的 180 和 50
    // 逻辑：背景偏亮，外框偏暗，但保证两者有足够的对比度以便算法能检测到边缘

    // 背景灰度：在 [150, 240] 之间随机
    int bgGray = 150 + rand() % 91;

    // 随机对比度：在 [60, 120] 之间
    int contrast = 60 + rand() % 61;

    // 外框灰度 = 背景 - 对比度
    int outerGray = bgGray - contrast;
    if (outerGray < 0) outerGray = 0;

    // 内芯灰度 = 背景灰度 (模拟“回”字形结构，中间空心透出背景)
    int innerGray = bgGray;

    // =========================================================
    // [修复] 动态调整超采样倍率 (Dynamic Super Sampling)
    // 原因：OpenCV 的 warpAffine 在图像尺寸超过 32768 时会崩溃
    // 逻辑：如果 size=800, scale=50 -> 40000 (Crash)。需限制最大维度。
    // =========================================================
    const int MAX_DIMENSION = 32000; // OpenCV 安全阈值

    // 如果预计尺寸超标，降低倍率
    if (size * targetScale > MAX_DIMENSION) {
        targetScale = MAX_DIMENSION / size;
        // 例如 size=800, targetScale 变为 40。800*40=32000 (安全)
    }
    // 保证最小精度 (至少10倍)
    if (targetScale < 1) targetScale = 1;

    // 1. 定义超高倍率
    // 100倍意味着 640x640 的图会变成 64000x64000 (内存爆炸)
    // 妥协方案：分块处理或适当降低到 20-50倍，或者优化逻辑
    // 为了演示完美效果，我们使用 50 倍 (既保证精度又不撑爆内存)
    const int SCALE = targetScale;
    int highResSize = size * SCALE;

    // 检查内存安全 (防止 size 过大导致分配失败)
    // 640 * 50 = 32000 -> 32000^2 * 1byte ~= 1GB (可以接受)
    Mat highResImg;
    try {
        highResImg.create(highResSize, highResSize, CV_8UC1);
        highResImg.setTo(Scalar(bgGray)); // 背景
    }
    catch (...) {
        // 如果内存不足，回退到 10倍
        cout << "[Warn] Memory low, fallback to 10x scale" << endl;
        return generateWaferImage(size, shiftX, shiftY, noiseLevel, angle);
    }

    Point2f center(highResSize / 2.0f, highResSize / 2.0f);

    // 偏移量放大 (精度 1/50 = 0.02px)
    Point2f innerCenter = center + Point2f((float)(shiftX * SCALE), (float)(shiftY * SCALE));

    // 3. 绘制外框 (放大版)
    int scaledOuterSize = OUTER_BOX_SIZE * SCALE;
    Rect outerRect(
        (int)(center.x - scaledOuterSize / 2),
        (int)(center.y - scaledOuterSize / 2),
        scaledOuterSize,
        scaledOuterSize
    );
    rectangle(highResImg, outerRect, Scalar(outerGray), -1);

    // 4. 绘制内芯 (放大版)
    int scaledInnerSize = INNER_BOX_SIZE * SCALE;
    Rect innerRect(
        (int)(innerCenter.x - scaledInnerSize / 2),
        (int)(innerCenter.y - scaledInnerSize / 2),
        scaledInnerSize,
        scaledInnerSize
    );
    rectangle(highResImg, innerRect, Scalar(innerGray), -1);

    // 5. 模拟旋转
    if (std::abs(angle) > 0.001) {
        Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
        // 使用 Nearest Neighbor 在超高分辨率下旋转，避免边缘模糊
        warpAffine(highResImg, highResImg, rotMat, highResImg.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar(180));
    }

    // 6. 下采样 (Downsampling)
    Mat finalImg;
    resize(highResImg, finalImg, Size(size, size), 0, 0, INTER_AREA);

    // 7. 模拟光学模糊
    // sigma=1.0 对应约 3-5 像素的边缘宽度，适合 Sigmoid 拟合
    GaussianBlur(finalImg, finalImg, Size(5, 5), 1.0);

    // 8. 添加噪声
    if (noiseLevel > 0) {
        Mat noise(finalImg.size(), finalImg.type());
        randn(noise, 0, noiseLevel);
        add(finalImg, noise, finalImg, noArray(), CV_8UC1);
    }

    // =========================================================
    // 9. [新增] 添加椒盐噪声 (Salt-and-Pepper Noise)
    // =========================================================
    // 椒盐噪声模拟灰尘(黑点)或坏点(白点)
    // 密度：假设 1% 的像素受到污染 (0.01)
    if (saltPepperChance > 0.0) {
        int totalPixels = finalImg.rows * finalImg.cols;
        int numSP = (int)(totalPixels * saltPepperChance); // 1% 的噪点

        for (int k = 0; k < numSP; ++k) {
            // 随机坐标
            int r = rand() % finalImg.rows;
            int c = rand() % finalImg.cols;

            // 随机决定是“椒”(黑, 0) 还是 “盐”(白, 255)
            if (rand() % 2 == 0) {
                finalImg.at<uchar>(r, c) = 0;   // Pepper
            }
            else {
                finalImg.at<uchar>(r, c) = 255; // Salt
            }
        }
    }

    return finalImg;
}


// [新增] 批量生成数据集
void ImageSimulator::generateDataset(int count, string imagesPath, string labelsPath) {
    // 1. 创建目录 (Windows _mkdir)
    // 如果路径包含多级目录，这里简单处理假设父目录存在，或者直接创建末级
    // 建议在外部调用时确保路径结构
    // 创建目录
    if (!fs::exists(imagesPath)) fs::create_directories(imagesPath);
    if (!fs::exists(labelsPath)) fs::create_directories(labelsPath);

    cout << "[Dataset] Generating " << count << " images..." << endl;
    cout << "   -> Images: " << imagesPath << endl;
    cout << "   -> Labels: " << labelsPath << endl;

    int targetSize = 640;
    // 使用 5倍 超采样，兼顾抗锯齿质量和生成速度
    // 640 * 5 = 3200，处理速度极快
    int trainScale = 5;

    for (int i = 0; i < count; ++i) {
        // --- A. 随机参数 ---
        double angle = (rand() % 101) / 10.0 - 5.0; // +/- 5 度
        double shiftX = (rand() % 400 - 200) / 100.0;
        double shiftY = (rand() % 400 - 200) / 100.0;
        double noise = 0.5 + (rand() % 250) / 100.0;
        double spChance = (rand() % 2 == 0) ? (rand() % 201) / 1000.0 : 0.0;

        // --- B. 直接生成目标尺寸图像 ---
        // 不再生成大图后裁剪，而是直接生成 640x640，物体在中心附近 (仅含 shift 偏移)
        Mat finalImg = generateWaferImage(targetSize, shiftX, shiftY, noise, angle, trainScale, spChance);

        // --- C. 保存图片 ---
        string baseName = "train_" + to_string(i);
        string imgFile = imagesPath + "/" + baseName + ".jpg";
        imwrite(imgFile, finalImg);

        // --- D. 计算 YOLO 标签 ---
        // 1. 中心坐标
        // 物体基础中心就在图像正中心
        double centerX = targetSize / 2.0;
        double centerY = targetSize / 2.0;

        // 注意：shiftX/Y 是内框相对于外框的偏移，外框本身是固定在中心的
        // YOLO 检测的是外框 (OuterBox)，所以外框的中心始终在图像中心 (受旋转影响极小，可视作不变)
        // 如果你需要 YOLO 能够处理物体不在视野中心的情况，必须保留之前的裁剪逻辑。
        // 但根据你的要求 "不需要模拟平移"，则物体中心固定。

        // 2. 计算旋转后的包围盒 (AABB)
        double rad = angle * CV_PI / 180.0;
        double s = (double)OUTER_BOX_SIZE;

        double bboxW = s * (abs(cos(rad)) + abs(sin(rad)));
        double bboxH = s * (abs(sin(rad)) + abs(cos(rad)));

        // 3. 归一化
        double normX = centerX / targetSize;
        double normY = centerY / targetSize;
        double normW = bboxW / targetSize;
        double normH = bboxH / targetSize;

        string labelFile = labelsPath + "/" + baseName + ".txt";
        ofstream outfile(labelFile);
        if (outfile.is_open()) {
            outfile << "0 " << fixed << setprecision(6)
                << normX << " " << normY << " " << normW << " " << normH << endl;
            outfile.close();
        }

        // 简单的进度显示
        if (i % 100 == 0) cout << "." << flush;
    }
    cout << "\n[Dataset] Done." << endl;
}