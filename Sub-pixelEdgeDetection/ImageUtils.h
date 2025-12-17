#pragma once

#include <opencv2/opencv.hpp>
#include <string> // (新增) 包含 string 头文件

/**
 * @class FilterUtils
 * @brief (Prompt 5) 存放滤波降噪方法的公共扩展类。
 */
class FilterUtils {
public:
    /**
     * @brief 应用中值滤波 (论文3.2.2节选择的方法)。
     * @param input 输入图像。
     * @param kernelSize 滤波器核大小 (必须是奇数)。
     * @return cv::Mat 滤波后的图像。
     */
    static cv::Mat applyMedianFilter(const cv::Mat& input, int kernelSize = 3);
};

/**
 * @class EnhancementUtils
 * @brief (Prompt 5) 存放图像增强方法的公共扩展类。
 */
class EnhancementUtils {
public:
    /**
     * @brief 应用直方图均衡化 (论文3.2.3节提到的方法)。
     * @param input 输入图像 (必须是8位灰度图)。
     * @return cv::Mat 增强后的图像。
     */
    static cv::Mat applyHistogramEqualization(const cv::Mat& input);
};

/**
 * @class ImagePreprocessor
 * @brief (Prompt 5) 图像预处理主类。
 *
 * 负责调用各种工具类来执行完整的预处理流程。
 */
class ImagePreprocessor {
public:
    /**
     * @brief 执行完整的预处理流程 (论文3.2节)。
     * @param inputImage 原始输入图像 (可以是彩色或灰度)。
     * @return cv::Mat 预处理完成的8位灰度图像。
     */
    cv::Mat preprocess(const cv::Mat& inputImage);

private:
    /**
     * @brief 确保图像是8位灰度图 (论文3.2.1节)。
     * @param input 输入图像。
     * @return cv::Mat 8位灰度图像。
     */
    cv::Mat ensureGrayscale(const cv::Mat& input);
};


/**
 * @class ImageIOUtils
 * @brief (新增) 存放图像 I/O (保存/加载) 操作的公共类。
 */
class ImageIOUtils {
public:
    /**
     * @brief 将图像保存到指定文件夹，并使用当前时间戳命名。
     * @param image 要保存的 cv::Mat 图像。
     * @param outputFolder 目标文件夹 (例如 "simulated_images")。
     */
    static void saveImageWithTimestamp(const cv::Mat& image, const std::string& outputFolder);
};
