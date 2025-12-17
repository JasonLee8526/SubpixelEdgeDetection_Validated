#include "ImageUtils.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

// (新增) 包含用于时间戳和文件操作的头文件
#include <chrono>      // 用于时间
#include <iomanip>     // 用于 std::put_time
#include <sstream>     // 用于 std::stringstream
#include <filesystem>  // 用于创建文件夹 (需要 C++17)

// --- FilterUtils 实现 ---

cv::Mat FilterUtils::applyMedianFilter(const cv::Mat& input, int kernelSize) {
    CV_Assert(kernelSize % 2 == 1); // 核必须是奇数
    cv::Mat output;
    cv::medianBlur(input, output, kernelSize);
    return output;
}

// --- EnhancementUtils 实现 ---

cv::Mat EnhancementUtils::applyHistogramEqualization(const cv::Mat& input) {
    CV_Assert(input.type() == CV_8UC1); // 必须是8位灰度图
    cv::Mat output;
    cv::equalizeHist(input, output);
    return output;
}

// --- ImagePreprocessor 实现 ---

cv::Mat ImagePreprocessor::ensureGrayscale(const cv::Mat& input) {
    if (input.channels() == 3 || input.channels() == 4) {
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    else if (input.type() != CV_8UC1) {
        cv::Mat gray8;
        input.convertTo(gray8, CV_8UC1);
        return gray8;
    }
    return input.clone();
}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat& inputImage) {
    // 1. 灰度空间变换 (论文 3.2.1)
    cv::Mat grayImg = ensureGrayscale(inputImage);

    // 2. 滤波降噪 (论文 3.2.2) - 论文选择了中值滤波
    cv::Mat filteredImg = FilterUtils::applyMedianFilter(grayImg, 3);

    // 3. 图像增强 (论文 3.2.3) - 论文提到了直方图均衡化
    cv::Mat enhancedImg = EnhancementUtils::applyHistogramEqualization(filteredImg);

    // std::cout << "[Preprocessor] 图像预处理完成。" << std::endl;
    return enhancedImg;
}


// --- (新增) ImageIOUtils 实现 ---

void ImageIOUtils::saveImageWithTimestamp(const cv::Mat& image, const std::string& outputFolder) {
    // 1. 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // 2. 格式化为 "YYYYMMDD_HHMMSS"
    std::tm tm_buf;
    // (修复 Windows 平台) 使用 localtime_s 
#ifdef _WIN32
    localtime_s(&tm_buf, &in_time_t);
#else
    tm_buf = *std::localtime(&in_time_t);
#endif

    std::stringstream ss;
    ss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");

    // 3. 添加毫秒以确保唯一性
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    ss << "_" << std::setw(3) << std::setfill('0') << ms.count() << ".png";
    std::string filename = ss.str();

    // 4. 检查并创建文件夹 (需要 C++17)
    std::filesystem::path folderPath(outputFolder);
    try {
        if (!std::filesystem::exists(folderPath)) {
            std::filesystem::create_directories(folderPath);
            std::cout << "[ImageIOUtils] 已创建文件夹: " << outputFolder << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[ImageIOUtils] 错误：无法创建文件夹 " << outputFolder << ": " << e.what() << std::endl;
        return;
    }

    // 5. 构造完整路径并保存
    std::filesystem::path filePath = folderPath / filename;
    try {
        cv::imwrite(filePath.string(), image);
        // std::cout << "[ImageIOUtils] 图像已保存至: " << filePath.string() << std::endl;
    }
    catch (const cv::Exception& ex) {
        std::cerr << "[ImageIOUtils] 错误：无法保存图像 " << filePath.string() << ": " << ex.what() << std::endl;
    }
}
