#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

/**
 * @struct Detection
 * @brief 存储单个目标的检测结果
 */
struct Detection {
    int class_id;       ///< 类别ID
    float confidence;   ///< 置信度
    cv::Rect box;       ///< 边界框 (x, y, w, h)
};

/**
 * @class YoloDetector
 * @brief (新增) 负责加载YOLO ONNX模型并执行推理
 * * 专门用于替代传统的模板匹配进行粗定位。
 * 使用 OpenCV DNN 模块加载 ONNX 模型。
 */
class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector() = default;

    /**
     * @brief 加载ONNX模型
     * @param modelPath ONNX文件路径
     * @param isGpu 是否使用GPU (需要OpenCV编译了CUDA支持)
     * @return bool 是否加载成功
     */
    bool loadModel(const std::string& modelPath, bool isGpu = false);

    /**
     * @brief 执行检测
     * @param img 输入图像 (BGR或Gray)
     * @param scoreThreshold 置信度阈值
     * @param nmsThreshold NMS阈值
     * @return std::vector<Detection> 检测到的目标列表
     */
    std::vector<Detection> detect(const cv::Mat& img, float scoreThreshold = 0.4f, float nmsThreshold = 0.4f);

    /**
     * @brief 获取模型输入尺寸
     */
    cv::Size getInputSize() const { return m_inputSize; }

private:
    /**
     * @brief 图像预处理 (Letterbox缩放)
     * 保持长宽比缩放，边缘填充灰色
     */
    cv::Mat formatToInput(const cv::Mat& source);

    cv::dnn::Net m_net;
    cv::Size m_inputSize; // YOLOv5 默认为 640x640
    std::vector<std::string> m_classNames;
};