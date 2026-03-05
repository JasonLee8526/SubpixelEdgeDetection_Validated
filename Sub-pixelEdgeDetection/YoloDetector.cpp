#include "YoloDetector.h"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

YoloDetector::YoloDetector(const string& modelPath) {
    try {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    catch (const cv::Exception& e) {
        cerr << "[YoloError] Error loading model: " << e.what() << endl;
    }
}

// 保持长宽比的预处理
Mat YoloDetector::formatToSquare(const Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = max(col, row);
    Mat result = Mat::zeros(_max, _max, source.type());
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

cv::Rect YoloDetector::detect(const cv::Mat& image) {
    // 0. 安全检查
    if (image.empty()) return Rect(0, 0, 0, 0);
    if (net.empty()) return Rect(0, 0, 0, 0);

    // ==========================================
    // [修复] 通道数适配
    // 原因：ImageSimulator 生成的是单通道灰度图 (CV_8UC1)
    // 但 YOLOv8 默认期望 3通道输入 (CV_8UC3)
    // 如果直接传单通道，net.forward 会因为维度不匹配崩溃
    // ==========================================
    Mat input_image = image;
    if (image.channels() == 1) {
        cvtColor(image, input_image, COLOR_GRAY2BGR);
    }

    // 1. 预处理
    Mat modelInput = formatToSquare(input_image);
    Mat blob;

    // YOLOv8 默认输入 640x640，归一化 0-1
    // swapRB=true: 将 BGR 转为 RGB (YOLO 训练时通常是 RGB)
    blobFromImage(modelInput, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);

    // 2. 推理 (此处之前报错)
    vector<Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    }
    catch (const cv::Exception& e) {
        cerr << "[YoloError] Inference failed: " << e.what() << endl;
        return Rect(0, 0, 0, 0);
    }

    // 3. 解析输出 (适配 YOLOv8 格式)
    Mat outputData = outputs[0];
    int dimensions = outputData.size[2];
    int rows = outputData.size[1];

    // 维度转置
    if (dimensions > rows) {
        outputData = outputData.reshape(1, rows);
        cv::transpose(outputData, outputData);
    }
    else {
        outputData = outputData.reshape(1, dimensions);
    }

    float* data = (float*)outputData.data;
    float x_factor = (float)modelInput.cols / 640.0f;
    float y_factor = (float)modelInput.rows / 640.0f;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < outputData.rows; ++i) {
        float* classes_scores = data + 4;
        Mat scores(1, outputData.cols - 4, CV_32FC1, classes_scores);
        Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > 0.45) { // 置信度阈值
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            // 还原坐标 (相对于 square input)
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back((float)max_class_score);
            class_ids.push_back(class_id.x);
        }
        data += outputData.cols;
    }

    // 4. NMS
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, 0.45, 0.45, nms_result);

    if (nms_result.empty()) {
        return Rect(0, 0, 0, 0);
    }

    // 返回最佳结果
    return boxes[nms_result[0]];
}