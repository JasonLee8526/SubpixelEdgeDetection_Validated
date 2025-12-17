#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>

#include "ImageSimulator.h"
#include "ImageUtils.h" 
#include "Localization.h"
#include "Utilities.h"

// ... (ExperimentResult 结构体和 calculateStatistics 函数保持不变) ...
struct ExperimentResult {
    double mean_error_x = 0.0;
    double mean_error_y = 0.0;
    double variance_x = 0.0;
    double variance_y = 0.0;
};

void calculateStatistics(const std::vector<double>& errors, double& out_mean, double& out_variance) {
    if (errors.empty()) {
        out_mean = 0.0;
        out_variance = 0.0;
        return;
    }
    double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    out_mean = sum / errors.size();
    double sq_sum = 0.0;
    for (double err : errors) {
        sq_sum += (err - out_mean) * (err - out_mean);
    }
    if (errors.size() > 1) {
        out_variance = sq_sum / (errors.size() - 1);
    }
    else {
        out_variance = 0.0;
    }
}

// [修改] 增加 useYolo 参数
bool runSingleMeasurement(
    ImageSimulator& simulator,
    ImagePreprocessor& preprocessor,
    Localization& localizer,
    double true_dx_um,
    double true_dy_um,
    int modelType,
    bool useYolo,
    double& out_error_x,
    double& out_error_y)
{
    // 1. 生成图像
    cv::Mat testImg = simulator.generateStandardWaferImage(800, 600, true_dx_um, true_dy_um);

    // 2. 预处理
    cv::Mat processedImg = preprocessor.preprocess(testImg);
    cv::Mat grayImg;
    if (testImg.channels() != 1) cv::cvtColor(testImg, grayImg, cv::COLOR_BGR2GRAY);
    else grayImg = testImg.clone();
    cv::medianBlur(grayImg, grayImg, 3);

    // 准备梯度图
    cv::Mat gradX = GradientUtils::applySobel(processedImg, 1, 0);
    cv::Mat gradY = GradientUtils::applySobel(processedImg, 0, 1);

    // 3. 粗定位 [核心修改部分]
    CoarseEdges coarseEdges;
    if (useYolo) {
        // 使用 YOLO 进行粗定位
        // 注意：YOLO 通常在彩色图或原始图上效果更好，所以传入 testImg
        coarseEdges = localizer.coarseLocalizationYolo(testImg);
    }
    else {
        // 传统的模板匹配
        coarseEdges = localizer.coarseLocalization(processedImg);
    }

    if (coarseEdges.x1 == -1 || coarseEdges.y1 == -1) {
        // std::cerr << "粗定位失败！" << std::endl;
        return false;
    }

    // 4. 精定位
    FineEdges fineEdges = localizer.fineLocalization(grayImg, gradX, gradY, coarseEdges, modelType);

    // 5. 得到结果
    double measured_dx_px = ((fineEdges.x1 + fineEdges.x4) - (fineEdges.x2 + fineEdges.x3)) / 2.0;
    double measured_dy_px = ((fineEdges.y1 + fineEdges.y4) - (fineEdges.y2 + fineEdges.y3)) / 2.0;

    double measured_dx_um = measured_dx_px * simulator.PIX_TO_UM_FACTOR;
    double measured_dy_um = measured_dy_px * simulator.PIX_TO_UM_FACTOR;

    out_error_x = measured_dx_um - true_dx_um;
    out_error_y = measured_dy_um - true_dy_um;

    return true;
}

int main() {
    std::cout << "--- 复现论文第6章实验 (Prompt 8) + YOLO优化 ---" << std::endl;

    ImageSimulator simulator(766, 50.0);
    ImagePreprocessor preprocessor;
    Localization localizer;

    //std::string datasetRoot = "E:/杂项/TestImage/datasets/wafer_project";
    //simulator.generateDataset(500, datasetRoot + "/train/images", datasetRoot + "/train/labels");

    //// 生成 100 张验证集
    //simulator.generateDataset(100, datasetRoot + "/val/images", datasetRoot + "/val/labels");

    //std::cout << "数据准备完毕，可以直接开始 python train.py 了！" << std::endl;

    //return 0;

    // 配置
    bool USE_YOLO = false; // [开关] 设置为 true 启用 YOLO，false 使用模板匹配
    std::string yoloModelPath = "best.onnx"; // 请确保文件存在

    if (USE_YOLO) {
        std::cout << "正在初始化 YOLO 检测器..." << std::endl;
        if (!localizer.initYoloModel(yoloModelPath)) {
            std::cerr << "错误：无法加载 YOLO 模型 (" << yoloModelPath << ")。" << std::endl;
            std::cerr << "提示：请先使用 Python 训练 YOLOv5 模型并导出为 ONNX，或将 USE_YOLO 设为 false 以使用传统算法。" << std::endl;
            // 降级处理，或者直接退出
            // return -1; 
            std::cout << "切换回传统模板匹配模式。" << std::endl;
            USE_YOLO = false;
        }
        else {
            std::cout << "YOLO 模型加载成功。" << std::endl;
        }
    }

    if (!USE_YOLO) {
        std::cout << "正在创建 (0,0) 模板 (传统方法)..." << std::endl;
        cv::Mat templateImg = simulator.generateStandardWaferImage(800, 600, 0.0, 0.0);
        if (!localizer.createTemplate(templateImg, preprocessor)) {
            std::cerr << "无法启动实验：模板创建失败。" << std::endl;
            return -1;
        }
    }

    std::vector<std::pair<double, double>> errorGroups = {
        {0.0, 0.0}, {-0.5, -1.0}, {-1.0, 0.0}, {-1.0, 0.1},
        {-1.0, -0.1}, {-1.0, 0.5}, {-1.0, 1.0}, {-1.0, -1.0}
    };

    int runsPerGroup = 5;
    int modelToTest = 0; // Sigmoid

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n--- 开始实验 (" << (USE_YOLO ? "YOLO Coarse" : "Template Coarse") << " + Sigmoid Fine) ---" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(15) << "标准误差 (um)"
        << std::setw(18) << "平均误差 X (um)"
        << std::setw(18) << "平均误差 Y (um)"
        << std::setw(18) << "方差 X (10e-5)"
        << std::setw(18) << "方差 Y (10e-5)"
        << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    std::vector<double> total_errors_x;
    std::vector<double> total_errors_y;

    for (const auto& group : errorGroups) {
        double true_dx = group.first;
        double true_dy = group.second;

        std::vector<double> group_errors_x;
        std::vector<double> group_errors_y;

        for (int i = 0; i < runsPerGroup; ++i) {
            double error_x, error_y;
            // 传入 USE_YOLO 标志
            if (runSingleMeasurement(simulator, preprocessor, localizer, true_dx, true_dy, modelToTest, USE_YOLO, error_x, error_y)) {
                group_errors_x.push_back(error_x);
                group_errors_y.push_back(error_y);
                total_errors_x.push_back(std::abs(error_x));
                total_errors_y.push_back(std::abs(error_y));
            }
        }

        double mean_x, var_x, mean_y, var_y;
        calculateStatistics(group_errors_x, mean_x, var_x);
        calculateStatistics(group_errors_y, mean_y, var_y);

        std::string group_str = "(" + std::to_string(true_dx) + ", " + std::to_string(true_dy) + ")";
        std::cout << std::setw(15) << group_str
            << std::setw(18) << mean_x
            << std::setw(18) << mean_y
            << std::setw(18) << var_x * 100000.0
            << std::setw(18) << var_y * 100000.0
            << std::endl;
    }

    double total_mean_abs_error_x, total_var_x_all;
    double total_mean_abs_error_y, total_var_y_all;
    calculateStatistics(total_errors_x, total_mean_abs_error_x, total_var_x_all);
    calculateStatistics(total_errors_y, total_mean_abs_error_y, total_var_y_all);

    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(15) << "总平均绝对误差"
        << std::setw(18) << total_mean_abs_error_x
        << std::setw(18) << total_mean_abs_error_y
        << std::setw(18) << "---" << std::setw(18) << "---" << std::endl;

    std::cout << "\n实验完成。请按 Enter 键退出..." << std::endl;
    std::cin.get();

    return 0;
}