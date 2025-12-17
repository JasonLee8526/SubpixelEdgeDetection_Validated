#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "SubPixelModel.h"
#include "YoloDetector.h" // [新增] 引入YOLO头文件

/**
 * @struct CoarseEdges
 * @brief 存储8条边的像素级粗定位结果。
 */
struct CoarseEdges {
    int x1, x2, x3, x4;
    int y1, y2, y3, y4;
};

/**
 * @struct FineEdges
 * @brief 存储8条边的亚像素级精定位结果。
 */
struct FineEdges {
    double x1, x2, x3, x4;
    double y1, y2, y3, y4;
};

/**
 * @class Localization
 * @brief 负责粗定位和精定位的主类。
 */
class Localization {
public:
    Localization();

    /**
     * @brief [修改] 初始化YOLO模型
     * @param modelPath 模型路径
     */
    bool initYoloModel(const std::string& modelPath);

    /**
     * @brief 从标准图像创建 RMS 梯度模板 (原有方法)。
     */
    bool createTemplate(const cv::Mat& standardImage, class ImagePreprocessor& preprocessor);

    /**
     * @brief 执行像素级粗定位 (模板匹配 - 原有方法)。
     */
    CoarseEdges coarseLocalization(const cv::Mat& processedImg);

    /**
     * @brief [新增] 使用YOLO进行粗定位
     * * 优化策略：
     * 1. 使用YOLO检测出内框和外框。
     * 2. 根据框的几何位置提取8条边缘的坐标。
     * * @param originalImg 原始图像(YOLO通常需要3通道，或者内部转换)
     * @return CoarseEdges
     */
    CoarseEdges coarseLocalizationYolo(const cv::Mat& originalImg);

    /**
     * @brief 执行亚像素级精定位。
     */
    FineEdges fineLocalization(
        const cv::Mat& originalImg,
        const cv::Mat& gradImgX,
        const cv::Mat& gradImgY,
        const CoarseEdges& coarseEdges,
        int modelType);

    std::vector<double> extractData(const std::vector<double>& rmsData, int center);

private:
    std::vector<double> m_templateRMS_X;
    std::vector<double> m_templateRMS_Y;
    int m_templateWindow;
    int m_subPixelWindow;

    // [新增] YOLO检测器实例
    YoloDetector m_yoloDetector;
    bool m_useYolo;

    std::vector<int> findTemplateMatchingPeaks(const std::vector<double>& rmsData, const std::vector<double>& templateData);
};