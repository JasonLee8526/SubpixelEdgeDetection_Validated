#define _CRT_SECURE_NO_WARNINGS // 禁用 fopen 等不安全函数的警告
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <direct.h> // 用于创建文件夹 (_mkdir)

#include "ImageSimulator.h"
#include "ImageUtils.h" 
#include "Localization.h"
#include "SubPixelModel.h"
#include "Utilities.h"
#include "WaferConfig.h" // 引入配置
#include "YoloDetector.h"
using namespace std;
using namespace cv;

#pragma region MyRegion
// ... (ExperimentResult 结构体和 calculateStatistics 函数保持不变) ...
//struct ExperimentResult {
//    double mean_error_x = 0.0;
//    double mean_error_y = 0.0;
//    double variance_x = 0.0;
//    double variance_y = 0.0;
//};
//
//void calculateStatistics(const std::vector<double>& errors, double& out_mean, double& out_variance) {
//    if (errors.empty()) {
//        out_mean = 0.0;
//        out_variance = 0.0;
//        return;
//    }
//    double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
//    out_mean = sum / errors.size();
//    double sq_sum = 0.0;
//    for (double err : errors) {
//        sq_sum += (err - out_mean) * (err - out_mean);
//    }
//    if (errors.size() > 1) {
//        out_variance = sq_sum / (errors.size() - 1);
//    }
//    else {
//        out_variance = 0.0;
//    }
//}
//
//// [修改] 增加 useYolo 参数
//bool runSingleMeasurement(
//    ImageSimulator& simulator,
//    ImagePreprocessor& preprocessor,
//    Localization& localizer,
//    double true_dx_um,
//    double true_dy_um,
//    int modelType,
//    bool useYolo,
//    double& out_error_x,
//    double& out_error_y)
//{
//    // 1. 生成图像
//    cv::Mat testImg = simulator.generateStandardWaferImage(800, 600, true_dx_um, true_dy_um);
//
//    // 2. 预处理
//    cv::Mat processedImg = preprocessor.preprocess(testImg);
//    cv::Mat grayImg;
//    if (testImg.channels() != 1) cv::cvtColor(testImg, grayImg, cv::COLOR_BGR2GRAY);
//    else grayImg = testImg.clone();
//    cv::medianBlur(grayImg, grayImg, 3);
//
//    // 准备梯度图
//    cv::Mat gradX = GradientUtils::applySobel(processedImg, 1, 0);
//    cv::Mat gradY = GradientUtils::applySobel(processedImg, 0, 1);
//
//    // 3. 粗定位 [核心修改部分]
//    CoarseEdges coarseEdges;
//    if (useYolo) {
//        // 使用 YOLO 进行粗定位
//        // 注意：YOLO 通常在彩色图或原始图上效果更好，所以传入 testImg
//        coarseEdges = localizer.coarseLocalizationYolo(testImg);
//    }
//    else {
//        // 传统的模板匹配
//        coarseEdges = localizer.coarseLocalization(processedImg);
//    }
//
//    if (coarseEdges.x1 == -1 || coarseEdges.y1 == -1) {
//        // std::cerr << "粗定位失败！" << std::endl;
//        return false;
//    }
//
//    // 4. 精定位
//    FineEdges fineEdges = localizer.fineLocalization(grayImg, gradX, gradY, coarseEdges, modelType);
//
//    // 5. 得到结果
//    double measured_dx_px = ((fineEdges.x1 + fineEdges.x4) - (fineEdges.x2 + fineEdges.x3)) / 2.0;
//    double measured_dy_px = ((fineEdges.y1 + fineEdges.y4) - (fineEdges.y2 + fineEdges.y3)) / 2.0;
//
//    double measured_dx_um = measured_dx_px * simulator.PIX_TO_UM_FACTOR;
//    double measured_dy_um = measured_dy_px * simulator.PIX_TO_UM_FACTOR;
//
//    out_error_x = measured_dx_um - true_dx_um;
//    out_error_y = measured_dy_um - true_dy_um;
//
//    return true;
//}
#pragma endregion



#pragma region MyRegion

//int main() {
//    std::cout << "--- 复现论文第6章实验 (Prompt 8) + YOLO优化 ---" << std::endl;
//
//    ImageSimulator simulator(766, 50.0);
//    ImagePreprocessor preprocessor;
//    Localization localizer;
//
//    //std::string datasetRoot = "E:/杂项/TestImage/datasets/wafer_project";
//    //simulator.generateDataset(500, datasetRoot + "/train/images", datasetRoot + "/train/labels");
//
//    //// 生成 100 张验证集
//    //simulator.generateDataset(100, datasetRoot + "/val/images", datasetRoot + "/val/labels");
//
//    //std::cout << "数据准备完毕，可以直接开始 python train.py 了！" << std::endl;
//
//    //return 0;
//
//    // 配置
//    bool USE_YOLO = false; // [开关] 设置为 true 启用 YOLO，false 使用模板匹配
//    std::string yoloModelPath = "best.onnx"; // 请确保文件存在
//
//    if (USE_YOLO) {
//        std::cout << "正在初始化 YOLO 检测器..." << std::endl;
//        if (!localizer.initYoloModel(yoloModelPath)) {
//            std::cerr << "错误：无法加载 YOLO 模型 (" << yoloModelPath << ")。" << std::endl;
//            std::cerr << "提示：请先使用 Python 训练 YOLOv5 模型并导出为 ONNX，或将 USE_YOLO 设为 false 以使用传统算法。" << std::endl;
//            // 降级处理，或者直接退出
//            // return -1; 
//            std::cout << "切换回传统模板匹配模式。" << std::endl;
//            USE_YOLO = false;
//        }
//        else {
//            std::cout << "YOLO 模型加载成功。" << std::endl;
//        }
//    }
//
//    if (!USE_YOLO) {
//        std::cout << "正在创建 (0,0) 模板 (传统方法)..." << std::endl;
//        cv::Mat templateImg = simulator.generateStandardWaferImage(800, 600, 0.0, 0.0);
//        if (!localizer.createTemplate(templateImg, preprocessor)) {
//            std::cerr << "无法启动实验：模板创建失败。" << std::endl;
//            return -1;
//        }
//    }
//
//    std::vector<std::pair<double, double>> errorGroups = {
//        {0.0, 0.0}, {-0.5, -1.0}, {-1.0, 0.0}, {-1.0, 0.1},
//        {-1.0, -0.1}, {-1.0, 0.5}, {-1.0, 1.0}, {-1.0, -1.0}
//    };
//
//    int runsPerGroup = 5;
//    int modelToTest = 0; // Sigmoid
//
//    std::cout << std::fixed << std::setprecision(6);
//    std::cout << "\n--- 开始实验 (" << (USE_YOLO ? "YOLO Coarse" : "Template Coarse") << " + Sigmoid Fine) ---" << std::endl;
//    std::cout << "----------------------------------------------------------------------------------" << std::endl;
//    std::cout << std::setw(15) << "标准误差 (um)"
//        << std::setw(18) << "平均误差 X (um)"
//        << std::setw(18) << "平均误差 Y (um)"
//        << std::setw(18) << "方差 X (10e-5)"
//        << std::setw(18) << "方差 Y (10e-5)"
//        << std::endl;
//    std::cout << "----------------------------------------------------------------------------------" << std::endl;
//
//    std::vector<double> total_errors_x;
//    std::vector<double> total_errors_y;
//
//    for (const auto& group : errorGroups) {
//        double true_dx = group.first;
//        double true_dy = group.second;
//
//        std::vector<double> group_errors_x;
//        std::vector<double> group_errors_y;
//
//        for (int i = 0; i < runsPerGroup; ++i) {
//            double error_x, error_y;
//            // 传入 USE_YOLO 标志
//            if (runSingleMeasurement(simulator, preprocessor, localizer, true_dx, true_dy, modelToTest, USE_YOLO, error_x, error_y)) {
//                group_errors_x.push_back(error_x);
//                group_errors_y.push_back(error_y);
//                total_errors_x.push_back(std::abs(error_x));
//                total_errors_y.push_back(std::abs(error_y));
//            }
//        }
//
//        double mean_x, var_x, mean_y, var_y;
//        calculateStatistics(group_errors_x, mean_x, var_x);
//        calculateStatistics(group_errors_y, mean_y, var_y);
//
//        std::string group_str = "(" + std::to_string(true_dx) + ", " + std::to_string(true_dy) + ")";
//        std::cout << std::setw(15) << group_str
//            << std::setw(18) << mean_x
//            << std::setw(18) << mean_y
//            << std::setw(18) << var_x * 100000.0
//            << std::setw(18) << var_y * 100000.0
//            << std::endl;
//    }
//
//    double total_mean_abs_error_x, total_var_x_all;
//    double total_mean_abs_error_y, total_var_y_all;
//    calculateStatistics(total_errors_x, total_mean_abs_error_x, total_var_x_all);
//    calculateStatistics(total_errors_y, total_mean_abs_error_y, total_var_y_all);
//
//    std::cout << "----------------------------------------------------------------------------------" << std::endl;
//    std::cout << std::setw(15) << "总平均绝对误差"
//        << std::setw(18) << total_mean_abs_error_x
//        << std::setw(18) << total_mean_abs_error_y
//        << std::setw(18) << "---" << std::setw(18) << "---" << std::endl;
//
//    std::cout << "\n实验完成。请按 Enter 键退出..." << std::endl;
//    std::cin.get();
//
//    return 0;
//}
#pragma endregion

struct TestCase {
    double shiftX;
    double shiftY;
    string description;
};

/// <summary>
/// 论文中的传统方法
/// </summary>
void TraditionalMethodTest() {
    cout << "================================================================================" << endl;
    cout << "   Sub-pixel Validation (Systematic Test Cases from Literature)    " << endl;
    cout << "================================================================================" << endl;

    ImageSimulator simulator;
    Localization localization;

    string saveDir = "TestImages";
    _mkdir(saveDir.c_str());

    // 1. 生成标准模板 (无偏移)
    cout << "[Init] Generating Standard Template..." << endl;
    Mat templateImg = simulator.generateWaferImage(WaferConfig::WAFER_SIZE, 0, 0, 0, 0);
    localization.createTemplate(templateImg, Rect(0, 0, 0, 0));

    // 2. 构建系统性测试用例 (模拟论文中的验证集)
    vector<TestCase> testCases;

    // Case Group 0: 零偏移基准
    testCases.push_back({ 0.0, 0.0, "Zero Reference" });

    // Case Group 1: X轴 亚像素线性测试 (0.1 - 1.0)
    // 验证算法对微小增量的敏感度
    for (int i = 1; i <= 10; ++i) {
        double val = i * 0.1;
        testCases.push_back({ val, 0.0, "X-Axis Step " + to_string(val).substr(0,3) });
    }

    // Case Group 2: Y轴 亚像素线性测试 (0.1 - 1.0)
    for (int i = 1; i <= 10; ++i) {
        double val = i * 0.1;
        testCases.push_back({ 0.0, val, "Y-Axis Step " + to_string(val).substr(0,3) });
    }

    // Case Group 3: 典型混合场景
    testCases.push_back({ 0.25, 0.25, "Quarter Shift" });
    testCases.push_back({ 0.50, 0.50, "Half Shift" });  // 0.5通常是很多插值算法的难点
    testCases.push_back({ 0.75, 0.75, "3/4 Shift" });
    testCases.push_back({ 1.50, 1.50, "Large Shift" });

    int numTests = testCases.size();
    int successCount = 0;
    double maxError = 0.0;

    cout << "\n[Start Testing] Running " << numTests << " systematic tests..." << endl;
    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;
    cout << "| ID | Desc             | True X  | True Y  | Meas X  | Meas Y  | Err X   | Err Y   | Status |" << endl;
    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;

    for (int i = 0; i < numTests; ++i) {
        double trueShiftX = testCases[i].shiftX;
        double trueShiftY = testCases[i].shiftY;

        // 保持 0 噪声和 0 旋转，专注于验证几何算法的正确性
        // 使用 50x 超采样 (ImageSimulator 内部实现)
        Mat testImg = simulator.generateWaferImage(640, trueShiftX, trueShiftY, 0.1,0,50,0.2);

        string filename = saveDir + "/Case_" + to_string(i) + ".png";
        imwrite(filename, testImg);

        Point coarsePos = localization.coarseLocalization(testImg);
        Point2d measured = localization.fineLocalization(testImg, coarsePos, SubPixelModel::Sigmoid);

        bool success = (measured.x != -999.0);
        double errX = 0.0, errY = 0.0;
        string statusStr = "FAIL";

        if (success) {
            errX = abs(measured.x - trueShiftX);
            errY = abs(measured.y - trueShiftY);
            maxError = max(maxError, max(errX, errY));

            // 严格标准: < 0.05 px
            if (errX < 0.05 && errY < 0.05) {
                statusStr = "PASS";
                successCount++;
            }
            else {
                statusStr = "WARN";
            }
        }

        cout << "| " << setw(2) << i << " | "
            << setw(16) << left << testCases[i].description << right << " | "
            << fixed << setprecision(3)
            << setw(7) << trueShiftX << " | "
            << setw(7) << trueShiftY << " | "
            << setw(7) << (success ? measured.x : 0) << " | "
            << setw(7) << (success ? measured.y : 0) << " | "
            << setw(7) << (success ? errX : 0) << " | "
            << setw(7) << (success ? errY : 0) << " | "
            << setw(6) << statusStr << " |" << endl;
    }

    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;
    cout << "\n[Summary] Success: " << successCount << "/" << numTests << endl;
    cout << "Max Error Observed: " << maxError << " px" << endl;

    if (maxError < 0.05) {
        cout << ">> SYSTEM VERIFIED: Algorithm matches paper's expected precision on standard steps." << endl;
    }
    else {
        cout << ">> STILL HAS ERROR: Look for patterns in the table above (e.g., is error higher at 0.5?)." << endl;
    }

    cout << "\nPress Enter to exit..." << endl;
    cin.get();

    return;
}

/// <summary>
/// 
/// </summary>
void GenerateTrainingImg() {
        ImageSimulator simulator;
    
        std::string datasetRoot = "E:/杂项/TestImage/datasets/wafer_project";
        simulator.generateDataset(1500, datasetRoot + "/train/images", datasetRoot + "/train/labels");
        simulator.generateDataset(300, datasetRoot + "/val/images", datasetRoot + "/val/labels");
    
        std::cout << "数据准备完毕，可以直接开始 python train.py 了！" << std::endl;
    
        return;
}

/// <summary>
/// 
/// </summary>
void Yolo8CoarseMthodTest() {
    cout << "================================================================================" << endl;
    cout << "   Sub-pixel Validation (YOLO Coarse Search + Sigmoid Fine Search)    " << endl;
    cout << "================================================================================" << endl;

    ImageSimulator simulator;
    Localization localization;
    YoloDetector* yolo = nullptr;

    // 1. 加载 YOLO 模型
    string modelPath = "best.onnx";
    // 简单检查文件是否存在
    FILE* f = fopen(modelPath.c_str(), "rb");
    if (f) {
        fclose(f);
        try {
            yolo = new YoloDetector(modelPath);
            cout << "[Init] YOLO model '" << modelPath << "' loaded successfully." << endl;
        }
        catch (const cv::Exception& e) {
            cout << "[Error] OpenCV DNN Exception: " << e.what() << endl;
        }
        catch (...) {
            cout << "[Error] Failed to initialize YOLO detector." << endl;
        }
    }
    else {
        cout << "[Warning] 'best.onnx' not found. Coarse search will FAIL." << endl;
    }

    string saveDir = "TestImages_YOLO";
    _mkdir(saveDir.c_str());

    // 2. 构建系统性测试用例 (X/Y 方向 0.1 - 1.0)
    vector<TestCase> testCases;

    // Case Group 0: 零偏移基准
    testCases.push_back({ 0.0, 0.0, "Zero Reference" });

    // Case Group 1: X轴 亚像素线性测试 (0.1 - 1.0)
    // 验证算法对微小增量的敏感度
    for (int i = 1; i <= 10; ++i) {
        double val = i * 0.1;
        testCases.push_back({ val, 0.0, "X-Axis Step " + to_string(val).substr(0,3) });
    }

    // Case Group 2: Y轴 亚像素线性测试 (0.1 - 1.0)
    for (int i = 1; i <= 10; ++i) {
        double val = i * 0.1;
        testCases.push_back({ 0.0, val, "Y-Axis Step " + to_string(val).substr(0,3) });
    }

    // Case Group 3: 典型混合场景
    testCases.push_back({ 0.25, 0.25, "Quarter Shift" });
    testCases.push_back({ 0.50, 0.50, "Half Shift" });  // 0.5通常是很多插值算法的难点
    testCases.push_back({ 0.75, 0.75, "3/4 Shift" });
    testCases.push_back({ 1.50, 1.50, "Large Shift" });

    int numTests = testCases.size();
    int successCount = 0;

    cout << "\n[Start Testing] Running " << numTests << " tests using YOLO..." << endl;
    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;
    cout << "| ID | Desc             | True X  | True Y  | Meas X  | Meas Y  | Err X   | Err Y   | Status |" << endl;
    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;

    for (int i = 0; i < numTests; ++i) {
        double trueShiftX = testCases[i].shiftX;
        double trueShiftY = testCases[i].shiftY;

        // 生成模拟图像
        // 使用 0.5 的噪声等级来模拟真实情况，验证 YOLO 的鲁棒性
        // 使用 scale=5 (快速模式) 生成图片
        Mat testImg = simulator.generateWaferImage(640, trueShiftX, trueShiftY, 0.5, 0,5,0.2);

        // 保存生成的图片用于检查
        string filename = saveDir + "/Yolo_Case_" + to_string(i) + ".png";
        imwrite(filename, testImg);

        // ==========================================
        // 核心变化: 使用 YOLO 进行粗定位
        // ==========================================
        Point coarsePos(0, 0);
        if (yolo) {
            // 调用我们之前修改好的 coarseLocalizationYolo
            // 它会自动将 YOLO 的中心转换为精定位所需的 TopLeft 锚点
            coarsePos = localization.coarseLocalizationYolo(testImg, yolo);
        }

        // 精定位 (保持 Sigmoid 不变)
        // 注意：如果 YOLO 返回 (0,0)，fineLocalization 内部会进行边界检查并返回 -999
        Point2d measured = localization.fineLocalization(testImg, coarsePos, SubPixelModel::Sigmoid);

        bool success = (measured.x != -999.0);
        double errX = 0.0, errY = 0.0;
        string statusStr = "FAIL";

        if (success) {
            errX = abs(measured.x - trueShiftX);
            errY = abs(measured.y - trueShiftY);

            // 判定标准: 单轴误差 < 0.1 px (考虑到 YOLO 框可能有微小抖动，稍微放宽一点点或保持 0.05)
            // 之前的纯几何验证可以达到 0.05，这里我们先看 0.1
            if (errX < 0.1 && errY < 0.1) {
                statusStr = "PASS";
                successCount++;
            }
            else {
                statusStr = "WARN";
            }
        }
        else {
            if (!yolo) statusStr = "NO_MDL";
            else statusStr = "MISS"; // YOLO 没检测到
        }

        // 格式化输出
        cout << "| " << setw(2) << i << " | "
            << setw(16) << left << testCases[i].description << right << " | "
            << fixed << setprecision(3)
            << setw(7) << trueShiftX << " | "
            << setw(7) << trueShiftY << " | "
            << setw(7) << (success ? measured.x : 0) << " | "
            << setw(7) << (success ? measured.y : 0) << " | "
            << setw(7) << (success ? errX : 0) << " | "
            << setw(7) << (success ? errY : 0) << " | "
            << setw(6) << statusStr << " |" << endl;
    }

    cout << setfill('-') << setw(110) << "-" << setfill(' ') << endl;
    cout << "\n[Summary] Success: " << successCount << "/" << numTests << endl;

    if (yolo) delete yolo;

    cout << "\nPress Enter to exit..." << endl;
    cin.get();

    return;
}

int main() {
    //训练数据
    //GenerateTrainingImg();

    //传统方法
    //TraditionalMethodTest();

    //yolo
    Yolo8CoarseMthodTest();
}


