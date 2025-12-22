#include "YoloDetector.h"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 构造函数
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

// 保持长宽比的预处理 (Letterbox)
Mat YoloDetector::formatToSquare(const Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = max(col, row);
    Mat result = Mat::zeros(_max, _max, source.type());
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

// 检测函数
cv::Rect YoloDetector::detect(const cv::Mat& image) {
    if (net.empty()) return Rect(0, 0, 0, 0);

    // 1. 预处理
    Mat modelInput = formatToSquare(image);
    Mat blob;
    // YOLOv8 默认输入 640x640，归一化 0-1
    blobFromImage(modelInput, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);

    // 2. 推理
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 3. 解析输出 (适配 YOLOv8 [1, 84, 8400] 格式)
    Mat outputData = outputs[0];
    int dimensions = outputData.size[2];
    int rows = outputData.size[1];

    // 维度转置处理 (确保 outputData 是 [8400 x 84])
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

    // 遍历所有 Anchors (通常 8400 个)
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

    // 4. NMS (非极大值抑制)
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, 0.45, 0.45, nms_result);

    if (nms_result.empty()) {
        return Rect(0, 0, 0, 0);
    }

    // 5. 返回最佳结果 (置信度最高的一个)
    // 这里直接返回 Rect，解决了 Localization.cpp 中的类型转换错误
    return boxes[nms_result[0]];
}