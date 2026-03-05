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

//// [核心修复] 空间矩法 (Spatial Moment / Center of Gravity)
//// 相比抛物线插值，它利用了边缘的整体信息，消除了模型偏差
//double SubPixelModel::momentMethod(const std::vector<double>& data) {
//    int n = data.size();
//    std::vector<double> grads(n, 0.0);
//
//    // 1. 计算梯度：使用中心差分 (Central Difference)
//    // 相比 data[i+1]-data[i]，中心差分不会引入 0.5 像素的相位偏移
//    // Grad[i] 对应位置 i
//    for (int i = 1; i < n - 1; ++i) {
//        // 使用 Scharr 算子或简单的中心差分
//        grads[i] = std::abs(data[i + 1] - data[i - 1]) / 2.0;
//    }
//
//    // 2. 寻找梯度峰值 (Rough Peak)
//    auto maxIt = std::max_element(grads.begin(), grads.end());
//    int peakIdx = std::distance(grads.begin(), maxIt);
//    double maxGrad = *maxIt;
//
//    // 3. 阈值过滤
//    // 仅使用峰值附近的有效数据参与重心计算，滤除背景噪声
//    double threshold = maxGrad * 0.3; // 经验值：只保留峰值 30% 以上的部分
//
//    double sumGrad = 0.0;
//    double sumIdxGrad = 0.0;
//
//    // 4. 定义积分窗口 (ROI within ROI)
//    // 避免远处的噪声干扰重心
//    int window = 5; // 在峰值左右各取 5 个点
//    int start = std::max(1, peakIdx - window);
//    int end = std::min(n - 1, peakIdx + window);
//
//    for (int i = start; i <= end; ++i) {
//        if (grads[i] > threshold) {
//            // 减去阈值基底，减少底噪影响
//            double val = grads[i] - threshold;
//            sumGrad += val;
//            sumIdxGrad += val * i;
//        }
//    }
//
//    if (std::abs(sumGrad) < 1e-6) return -999.0;
//
//    // 5. 计算重心 (Sub-pixel Position)
//    double center = sumIdxGrad / sumGrad;
//
//    return center;
//}

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

//// [核心修复] 改进的空间矩法：抗残余椒盐噪声
//double SubPixelModel::momentMethod(const std::vector<double>& data) {
//    int n = data.size();
//    std::vector<double> grads(n, 0.0);
//
//    // 1. 计算梯度 (中心差分)
//    for (int i = 1; i < n - 1; ++i) {
//        grads[i] = std::abs(data[i + 1] - data[i - 1]) / 2.0;
//    }
//
//    // =========================================================
//    // [关键改进] 寻找最佳峰值 (Best Peak Search)
//    // 问题：残余椒盐噪点的梯度(255)往往大于真实边缘(100)，导致 max_element 找错。
//    // 策略：我们信任粗定位的结果，真实边缘一定在 ROI 中心附近。
//    //       因此，我们引入 "距离权重"，优先选择靠近中心的峰值。
//    // =========================================================
//
//    int roiCenter = n / 2;
//    int bestPeakIdx = -1;
//    double maxWeightedGrad = -1.0;
//
//    // 距离惩罚系数 (Sigma): 控制对中心偏离的容忍度
//    // 值越小，越倾向于选择中心的峰值；值越大，越允许边缘偏离
//    double sigmaSpace = 10.0;
//
//    for (int i = 1; i < n - 1; ++i) {
//        // 原始梯度
//        double g = grads[i];
//
//        // 距离中心的距离
//        double dist = std::abs(i - roiCenter);
//
//        // 高斯加权：距离越远，权重越低
//        // Weight = exp(-dist^2 / (2 * sigma^2))
//        double weight = std::exp(-(dist * dist) / (2 * sigmaSpace * sigmaSpace));
//
//        // 加权后的梯度分数
//        double score = g * weight;
//
//        if (score > maxWeightedGrad) {
//            maxWeightedGrad = score;
//            bestPeakIdx = i;
//        }
//    }
//
//    // 如果找不到有效峰值，或者加权分数太低（全是噪声）
//    if (bestPeakIdx == -1 || maxWeightedGrad < 5.0) return -999.0;
//
//    // ---------------------------------------------------------
//    // 后续逻辑保持不变：在最佳峰值附近开小窗口计算重心
//    // ---------------------------------------------------------
//
//    // 窗口大小：只取峰值左右 3 个点
//    // 椒盐噪声通常是孤立的，真实边缘是连续的，小窗口能避开远处的噪点
//    int window = 3;
//
//    int start = std::max(1, bestPeakIdx - window);
//    int end = std::min(n - 2, bestPeakIdx + window);
//
//    double sumGrad = 0.0;
//    double sumIdxGrad = 0.0;
//
//    // 阈值：基于原始梯度（非加权）设定
//    // 只保留峰值能量 40% 以上的部分
//    double peakVal = grads[bestPeakIdx];
//    double threshold = peakVal * 0.4;
//
//    for (int i = start; i <= end; ++i) {
//        if (grads[i] > threshold) {
//            double val = grads[i] - threshold;
//            sumGrad += val;
//            sumIdxGrad += val * i;
//        }
//    }
//
//    if (std::abs(sumGrad) < 1e-6) return -999.0;
//
//    return sumIdxGrad / sumGrad;
//}

// 保留旧接口定义以防编译报错，但内部不再使用
double SubPixelModel::fitSigmoid(const std::vector<double>& data) {
    return momentMethod(data);
}

double SubPixelModel::fitGaussian(const std::vector<double>& data) {
    return momentMethod(data);
}