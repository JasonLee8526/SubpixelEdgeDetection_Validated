#pragma once

// --- 晶圆套刻物理尺寸定义 ---
// 作用：确保 ImageSimulator 生成的图像与 Localization 寻找的边缘位置严格一致
// 避免因“参数硬编码不匹配”导致算法失效

namespace WaferConfig {
    // 基础图像参数
    const int WAFER_SIZE = 400;      // 标准模板图像大小 (400x400)

    // Box-in-Box 结构参数 (单位: 像素)
    const int OUTER_BOX_SIZE = 200;  // 外框边长
    const int INNER_BOX_SIZE = 100;  // 内框边长
    const int LINE_WIDTH = 15;       // 线条宽度

    // 精定位搜索参数
    const int ROI_SEARCH_LEN = 60;   // 搜索区域长度 (垂直于边缘方向)
    const int ROI_SEARCH_WID = 20;   // 搜索区域宽度 (平行于边缘方向)

    // 边缘判定阈值 (防止对纯黑背景进行拟合)
    const double EDGE_GRADIENT_THRESHOLD = 5.0;
}