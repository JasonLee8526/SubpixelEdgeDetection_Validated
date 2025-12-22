#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class SubPixelModel {
public:
    // 定义支持的拟合算法类型
    enum ModelType {
        Sigmoid,
        GrayMoment,
        SpatialMoment,
        Gaussian,
        Polynomial,
        ArcTan
    };

    SubPixelModel();
    ~SubPixelModel();

    // [核心接口] 统一入口：根据类型计算边缘位置
    // profile: 边缘区域的灰度投影数据
    // type: 算法类型
    // 返回值: 亚像素边缘相对于 profile 起点的偏移量
    double calculateEdge(const std::vector<double>& profile, ModelType type);

private:
    // 具体算法实现 (声明)
    double fitSigmoid(const std::vector<double>& data);
    double fitGaussian(const std::vector<double>& data);
    double momentMethod(const std::vector<double>& data);
    // 其他模型可以在此扩展...
};