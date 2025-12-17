#include "YoloDetector.h"
#include <iostream>

YoloDetector::YoloDetector() : m_inputSize(640, 640) {
    // 假设类别只有一类 "mark" 或者两类 "outer", "inner"
    // 这里为了通用，我们主要依赖边界框的大小关系来区分内外框
}

bool YoloDetector::loadModel(const std::string& modelPath, bool isGpu) {
    try {
        std::cout << "[YoloDetector] 正在加载模型: " << modelPath << " ..." << std::endl;
        m_net = cv::dnn::readNetFromONNX(modelPath);

        if (isGpu) {
            std::cout << "[YoloDetector] 尝试使用 CUDA 后端." << std::endl;
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else {
            std::cout << "[YoloDetector] 使用 CPU 后端." << std::endl;
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        return !m_net.empty();
    }
    catch (const cv::Exception& e) {
        std::cerr << "[YoloDetector] 模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YoloDetector::formatToInput(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);

    // 将图像拷贝到正方形画布中心或左上角
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& img, float scoreThreshold, float nmsThreshold) {
    std::vector<Detection> detections;

    // 1. 预处理
    cv::Mat inputImg;
    // 确保输入是3通道
    if (img.channels() == 1) {
        cv::cvtColor(img, inputImg, cv::COLOR_GRAY2BGR);
    }
    else {
        inputImg = img;
    }

    // Letterbox 处理 (简单版：填充到正方形)
    cv::Mat blob;
    int w = inputImg.cols;
    int h = inputImg.rows;
    int maxD = std::max(w, h);
    cv::Mat squared = cv::Mat::zeros(maxD, maxD, CV_8UC3);
    inputImg.copyTo(squared(cv::Rect(0, 0, w, h)));

    cv::dnn::blobFromImage(squared, blob, 1.0 / 255.0, m_inputSize, cv::Scalar(), true, false);

    // 2. 推理
    m_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

    // 3. 后处理 (解析 YOLOv5 输出)
    // YOLOv5 输出通常是 [1, 25200, 5 + classes] (对于 640x640)
    // 格式: x, y, w, h, obj_conf, class_scores...

    float* data = (float*)outputs[0].data;
    // 获取输出维度
    int rows = outputs[0].size[1]; // 25200
    int dimensions = outputs[0].size[2]; // 85 (if 80 classes) or 6 (if 1 class)

    // 计算缩放因子
    float x_factor = (float)maxD / m_inputSize.width;
    float y_factor = (float)maxD / m_inputSize.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= scoreThreshold) {
            float* classes_scores = data + 5;
            // 找到最大分数的类别
            cv::Mat scores(1, dimensions - 5, CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > scoreThreshold) {
                // 综合置信度
                float final_score = confidence * (float)max_class_score;

                float x = data[0];
                float y = data[1];
                float w_d = data[2];
                float h_d = data[3];

                int left = int((x - 0.5 * w_d) * x_factor);
                int top = int((y - 0.5 * h_d) * y_factor);
                int width = int(w_d * x_factor);
                int height = int(h_d * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(final_score);
                class_ids.push_back(class_id_point.x);
            }
        }
        data += dimensions;
    }

    // NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nms_result);

    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detections.push_back(result);
    }

    return detections;
}