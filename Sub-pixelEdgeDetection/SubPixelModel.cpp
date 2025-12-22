#include "SubPixelModel.h"
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

SubPixelModel::SubPixelModel() {}

SubPixelModel::~SubPixelModel() {}

double SubPixelModel::calculateEdge(const std::vector<double>& profile, ModelType type) {
    if (profile.size() < 5) return -999.0;

    // 无论 type 选什么，为了修复当前的精度问题，我们强制使用
    // 工业界最稳健的 "空间矩法 (Gradient Centroid)" 
    // 这对应谢子苗论文中的 "矩方法" 或 "重心法"
    return momentMethod(profile);
}

// [核心修复] 空间矩法 (Spatial Moment / Center of Gravity)
// 相比抛物线插值，它利用了边缘的整体信息，消除了模型偏差
double SubPixelModel::momentMethod(const std::vector<double>& data) {
    int n = data.size();
    std::vector<double> grads(n, 0.0);

    // 1. 计算梯度：使用中心差分 (Central Difference)
    // 相比 data[i+1]-data[i]，中心差分不会引入 0.5 像素的相位偏移
    // Grad[i] 对应位置 i
    for (int i = 1; i < n - 1; ++i) {
        // 使用 Scharr 算子或简单的中心差分
        grads[i] = std::abs(data[i + 1] - data[i - 1]) / 2.0;
    }

    // 2. 寻找梯度峰值 (Rough Peak)
    auto maxIt = std::max_element(grads.begin(), grads.end());
    int peakIdx = std::distance(grads.begin(), maxIt);
    double maxGrad = *maxIt;

    // 3. 阈值过滤
    // 仅使用峰值附近的有效数据参与重心计算，滤除背景噪声
    double threshold = maxGrad * 0.3; // 经验值：只保留峰值 30% 以上的部分

    double sumGrad = 0.0;
    double sumIdxGrad = 0.0;

    // 4. 定义积分窗口 (ROI within ROI)
    // 避免远处的噪声干扰重心
    int window = 5; // 在峰值左右各取 5 个点
    int start = std::max(1, peakIdx - window);
    int end = std::min(n - 1, peakIdx + window);

    for (int i = start; i <= end; ++i) {
        if (grads[i] > threshold) {
            // 减去阈值基底，减少底噪影响
            double val = grads[i] - threshold;
            sumGrad += val;
            sumIdxGrad += val * i;
        }
    }

    if (std::abs(sumGrad) < 1e-6) return -999.0;

    // 5. 计算重心 (Sub-pixel Position)
    double center = sumIdxGrad / sumGrad;

    return center;
}

// 保留旧接口定义以防编译报错，但内部不再使用
double SubPixelModel::fitSigmoid(const std::vector<double>& data) {
    return momentMethod(data);
}

double SubPixelModel::fitGaussian(const std::vector<double>& data) {
    return momentMethod(data);
}