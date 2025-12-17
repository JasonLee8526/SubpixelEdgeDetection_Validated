#include "Localization.h"
#include "ImageUtils.h"
#include "Utilities.h"
#include <iostream>
#include <algorithm>

Localization::Localization() : m_templateWindow(15), m_subPixelWindow(20), m_useYolo(false) {}

// [新增] 初始化YOLO
bool Localization::initYoloModel(const std::string& modelPath) {
    if (m_yoloDetector.loadModel(modelPath)) {
        m_useYolo = true;
        return true;
    }
    return false;
}

// ... (原有 createTemplate 实现保持不变) ...
bool Localization::createTemplate(const cv::Mat& standardImage, ImagePreprocessor& preprocessor) {
    // 模板获取逻辑保持不变，用于兼容对比
    cv::Mat processed = preprocessor.preprocess(standardImage);
    cv::Mat gradX = GradientUtils::applySobel(processed, 1, 0);
    cv::Mat gradY = GradientUtils::applySobel(processed, 0, 1);
    std::vector<double> rmsGradX = Utilities::calculateRMSGradient(gradX, 0);
    std::vector<double> rmsGradY = Utilities::calculateRMSGradient(gradY, 1);
    std::vector<int> peaksX = Utilities::findPeaks(rmsGradX, 50);
    std::vector<int> peaksY = Utilities::findPeaks(rmsGradY, 50);

    if (peaksX.size() < 4 || peaksY.size() < 4) {
        std::cerr << "错误：创建模板失败，未找到足够的峰值。" << std::endl;
        return false;
    }
    int x_start = std::max(0, peaksX[0] - m_templateWindow);
    int x_end = std::min((int)rmsGradX.size(), peaksX[3] + m_templateWindow + 1);
    m_templateRMS_X.assign(rmsGradX.begin() + x_start, rmsGradX.begin() + x_end);
    int y_start = std::max(0, peaksY[0] - m_templateWindow);
    int y_end = std::min((int)rmsGradY.size(), peaksY[3] + m_templateWindow + 1);
    m_templateRMS_Y.assign(rmsGradY.begin() + y_start, rmsGradY.begin() + y_end);
    return true;
}

// ... (原有 findTemplateMatchingPeaks 实现保持不变) ...
std::vector<int> Localization::findTemplateMatchingPeaks(const std::vector<double>& rmsData, const std::vector<double>& templateData) {
    if (templateData.empty()) return {};
    int n_data = rmsData.size();
    int n_template = templateData.size();
    if (n_data < n_template) return {};
    double max_corr = -2.0;
    int best_start_pos = 0;
    for (int i = 0; i <= n_data - n_template; ++i) {
        std::vector<double> window(rmsData.begin() + i, rmsData.begin() + i + n_template);
        double corr = Utilities::calculateSpearman(window, templateData);
        if (corr > max_corr) {
            max_corr = corr;
            best_start_pos = i;
        }
    }
    std::vector<double> relevant_data(rmsData.begin() + best_start_pos, rmsData.begin() + best_start_pos + n_template);
    std::vector<int> local_peaks = Utilities::findPeaks(relevant_data, 50);
    std::vector<int> global_peaks;
    for (int peak : local_peaks) {
        global_peaks.push_back(peak + best_start_pos);
    }
    while (global_peaks.size() < 4) global_peaks.push_back(-1);
    return std::vector<int>(global_peaks.begin(), global_peaks.begin() + 4);
}

// ... (原有 coarseLocalization 实现保持不变) ...
CoarseEdges Localization::coarseLocalization(const cv::Mat& processedImg) {
    cv::Mat gradX = GradientUtils::applySobel(processedImg, 1, 0);
    cv::Mat gradY = GradientUtils::applySobel(processedImg, 0, 1);
    std::vector<double> rmsGradX = Utilities::calculateRMSGradient(gradX, 0);
    std::vector<double> rmsGradY = Utilities::calculateRMSGradient(gradY, 1);
    std::vector<int> peaksX = findTemplateMatchingPeaks(rmsGradX, m_templateRMS_X);
    std::vector<int> peaksY = findTemplateMatchingPeaks(rmsGradY, m_templateRMS_Y);
    return { peaksX[0], peaksX[1], peaksX[2], peaksX[3],
            peaksY[0], peaksY[1], peaksY[2], peaksY[3] };
}

// [新增] YOLO粗定位实现
CoarseEdges Localization::coarseLocalizationYolo(const cv::Mat& originalImg) {
    CoarseEdges edges = { -1, -1, -1, -1, -1, -1, -1, -1 };

    if (!m_useYolo) {
        std::cerr << "错误: YOLO模型未加载，无法使用YOLO定位。" << std::endl;
        return edges;
    }

    // 1. 执行推理
    std::vector<Detection> detections = m_yoloDetector.detect(originalImg);

    if (detections.size() < 2) {
        std::cerr << "警告: YOLO检测到的目标少于2个，无法构成Box-in-Box结构。" << std::endl;
        return edges;
    }

    // 2. 逻辑解析：Box-in-Box 结构通常包含一个外框和一个内框
    // 我们可以通过面积大小来区分：面积大的是外框，面积小的是内框
    // 或者通过训练时的类别ID区分（如果训练了 "outer", "inner" 两类）

    // 这里假设通过面积排序：大的是外框，小的是内框
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.box.area() > b.box.area();
        });

    cv::Rect outerBox = detections[0].box;
    cv::Rect innerBox = detections[1].box;

    // 3. 将边界框坐标映射到 CoarseEdges
    // 约定：
    // x1: 外框左边缘, x2: 内框左边缘, x3: 内框右边缘, x4: 外框右边缘
    // y1: 外框上边缘, y2: 内框上边缘, y3: 内框下边缘, y4: 外框下边缘

    // 注意：YOLO回归的框可能不是很精确的贴合边缘（取决于标注质量），
    // 但作为"粗定位"（Coarse），只需要在亚像素窗口(例如20px)范围内即可。

    edges.x1 = outerBox.x;
    edges.x2 = innerBox.x;
    edges.x3 = innerBox.x + innerBox.width;
    edges.x4 = outerBox.x + outerBox.width;

    edges.y1 = outerBox.y;
    edges.y2 = innerBox.y;
    edges.y3 = innerBox.y + innerBox.height;
    edges.y4 = outerBox.y + outerBox.height;

    // 简单的合理性检查
    if (edges.x1 > edges.x2 || edges.x3 > edges.x4) {
        std::cerr << "警告: YOLO检测到的框嵌套逻辑异常 (X轴)。" << std::endl;
    }

    return edges;
}

std::vector<double> Localization::extractData(const std::vector<double>& rmsData, int center) {
    std::vector<double> data(2 * m_subPixelWindow + 1);
    for (int i = -m_subPixelWindow; i <= m_subPixelWindow; ++i) {
        int idx = center + i;
        if (idx < 0) idx = 0;
        if (idx >= rmsData.size()) idx = rmsData.size() - 1;
        data[i + m_subPixelWindow] = rmsData[idx];
    }
    return data;
}

// ... (FineLocalization 保持不变) ...
FineEdges Localization::fineLocalization(
    const cv::Mat& originalImg,
    const cv::Mat& gradImgX,
    const cv::Mat& gradImgY,
    const CoarseEdges& coarseEdges,
    int modelType) {

    // 灰度模型使用 RMS 灰度
    std::vector<double> rmsGrayX = Utilities::calculateRMSGray(originalImg, 0);
    std::vector<double> rmsGrayY = Utilities::calculateRMSGray(originalImg, 1);

    // 梯度模型使用 RMS 梯度
    std::vector<double> rmsGradX = Utilities::calculateRMSGradient(gradImgX, 0);
    std::vector<double> rmsGradY = Utilities::calculateRMSGradient(gradImgY, 1);

    std::vector<double> d_x1, d_x2, d_x3, d_x4, d_y1, d_y2, d_y3, d_y4;
    FitResult f_x1, f_x2, f_x3, f_x4, f_y1, f_y2, f_y3, f_y4;

    // 提取数据
    if (modelType == 0 || modelType == 3 || modelType == 4 || modelType == 5) { // 灰度模型
        d_x1 = extractData(rmsGrayX, coarseEdges.x1);
        d_x2 = extractData(rmsGrayX, coarseEdges.x2);
        d_x3 = extractData(rmsGrayX, coarseEdges.x3);
        d_x4 = extractData(rmsGrayX, coarseEdges.x4);
        d_y1 = extractData(rmsGrayY, coarseEdges.y1);
        d_y2 = extractData(rmsGrayY, coarseEdges.y2);
        d_y3 = extractData(rmsGrayY, coarseEdges.y3);
        d_y4 = extractData(rmsGrayY, coarseEdges.y4);
    }
    else { // 梯度模型
        d_x1 = extractData(rmsGradX, coarseEdges.x1);
        d_x2 = extractData(rmsGradX, coarseEdges.x2);
        d_x3 = extractData(rmsGradX, coarseEdges.x3);
        d_x4 = extractData(rmsGradX, coarseEdges.x4);
        d_y1 = extractData(rmsGradY, coarseEdges.y1);
        d_y2 = extractData(rmsGradY, coarseEdges.y2);
        d_y3 = extractData(rmsGradY, coarseEdges.y3);
        d_y4 = extractData(rmsGradY, coarseEdges.y4);
    }

    // 执行拟合
    switch (modelType) {
    case 0: // Sigmoid
    default:
        f_x1 = SubPixelModel::fitSigmoid(d_x1); f_x2 = SubPixelModel::fitSigmoid(d_x2);
        f_x3 = SubPixelModel::fitSigmoid(d_x3); f_x4 = SubPixelModel::fitSigmoid(d_x4);
        f_y1 = SubPixelModel::fitSigmoid(d_y1); f_y2 = SubPixelModel::fitSigmoid(d_y2);
        f_y3 = SubPixelModel::fitSigmoid(d_y3); f_y4 = SubPixelModel::fitSigmoid(d_y4);
        break;
    case 1: // Quadratic
        f_x1 = SubPixelModel::fitQuadratic(d_x1); f_x2 = SubPixelModel::fitQuadratic(d_x2);
        f_x3 = SubPixelModel::fitQuadratic(d_x3); f_x4 = SubPixelModel::fitQuadratic(d_x4);
        f_y1 = SubPixelModel::fitQuadratic(d_y1); f_y2 = SubPixelModel::fitQuadratic(d_y2);
        f_y3 = SubPixelModel::fitQuadratic(d_y3); f_y4 = SubPixelModel::fitQuadratic(d_y4);
        break;
    case 2: // Gaussian
        f_x1 = SubPixelModel::fitGaussian(d_x1); f_x2 = SubPixelModel::fitGaussian(d_x2);
        f_x3 = SubPixelModel::fitGaussian(d_x3); f_x4 = SubPixelModel::fitGaussian(d_x4);
        f_y1 = SubPixelModel::fitGaussian(d_y1); f_y2 = SubPixelModel::fitGaussian(d_y2);
        f_y3 = SubPixelModel::fitGaussian(d_y3); f_y4 = SubPixelModel::fitGaussian(d_y4);
        break;
    }

    FineEdges fineEdges;
    fineEdges.x1 = coarseEdges.x1 - m_subPixelWindow + (f_x1.success ? f_x1.edge_position : m_subPixelWindow);
    fineEdges.x2 = coarseEdges.x2 - m_subPixelWindow + (f_x2.success ? f_x2.edge_position : m_subPixelWindow);
    fineEdges.x3 = coarseEdges.x3 - m_subPixelWindow + (f_x3.success ? f_x3.edge_position : m_subPixelWindow);
    fineEdges.x4 = coarseEdges.x4 - m_subPixelWindow + (f_x4.success ? f_x4.edge_position : m_subPixelWindow);
    fineEdges.y1 = coarseEdges.y1 - m_subPixelWindow + (f_y1.success ? f_y1.edge_position : m_subPixelWindow);
    fineEdges.y2 = coarseEdges.y2 - m_subPixelWindow + (f_y2.success ? f_y2.edge_position : m_subPixelWindow);
    fineEdges.y3 = coarseEdges.y3 - m_subPixelWindow + (f_y3.success ? f_y3.edge_position : m_subPixelWindow);
    fineEdges.y4 = coarseEdges.y4 - m_subPixelWindow + (f_y4.success ? f_y4.edge_position : m_subPixelWindow);

    return fineEdges;
}