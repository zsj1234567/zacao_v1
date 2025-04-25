import logging
from skimage.morphology import disk
from skimage.segmentation import watershed as skimage_watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
import json
from .hsv_analyzer import analyze_hsv_thresholds  # 使用相对路径导入
matplotlib.use('Agg')  # 设置后端为Agg，这必须在导入pyplot之前设置

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei',
                                          'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

# 尝试加载中文字体，如果失败则使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 测试中文字体是否可用
    fig = plt.figure(figsize=(1, 1))
    plt.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(fig)
    use_chinese = True
except:
    use_chinese = False
    print("警告: 无法加载中文字体，将使用英文显示")


class GrassAnalyzer:
    def __init__(self, calibration_points=None):
        """
        初始化草地分析器

        参数:
            calibration_points: 校准点坐标，用于定义1平方米区域
                                格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                                如果为None，将尝试从校准文件加载
        """
        self.calibration_points = calibration_points
        self.calibrated_image = None
        self.original_image = None
        self.mask = None
        self.instances = None
        self.debug_images = {}  # 用于存储调试图像
        self.calibration_dir = 'calibrations'  # 校准文件目录
        self.crop_region = None  # 剪裁区域坐标 (x_min, y_min, x_max, y_max)
        self.calibration_area_pixels = None  # 校准区域的实际面积（像素数）

    def load_image(self, image_path):
        """加载图像并尝试加载对应的校准文件 (支持非ASCII路径)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件 {image_path} 不存在")

        # self.original_image = cv2.imread(image_path)
        # 使用 imdecode 处理非 ASCII 路径
        try:
            # 先用 numpy 读取文件字节
            n = np.fromfile(image_path, dtype=np.uint8)
            # 然后用 OpenCV 从内存解码
            self.original_image = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception as e:
             raise ValueError(f"使用 numpy/cv2.imdecode 读取图像时出错 {image_path}: {e}")

        if self.original_image is None:
            raise ValueError(f"无法读取或解码图像 {image_path} (cv2.imdecode 返回 None)")

        # 转换为RGB颜色空间（OpenCV默认为BGR）
        self.original_image = cv2.cvtColor(
            self.original_image, cv2.COLOR_BGR2RGB)

        # 如果没有提供校准点，尝试从校准文件加载
        if self.calibration_points is None:
            self._load_calibration(image_path)

        return self.original_image

    def _load_calibration(self, image_path):
        """从校准文件加载校准点"""
        # 确保校准目录存在
        os.makedirs(self.calibration_dir, exist_ok=True)

        # 构建校准文件路径，保持与图像相同的目录结构
        rel_path = os.path.dirname(image_path)
        if rel_path == '':  # 如果图像在当前目录
            rel_path = '.'

        # 构建校准文件路径
        image_name = os.path.basename(image_path)
        calibration_file = os.path.join(
            self.calibration_dir, rel_path, f"{image_name}.json")

        # 尝试加载校准文件
        try:
            # 移除图像文件扩展名，使用基本文件名
            image_basename = os.path.splitext(image_name)[0]
            calibration_file_no_ext = os.path.join(
                self.calibration_dir, rel_path, f"{image_basename}.json")

            if os.path.exists(calibration_file_no_ext):
                with open(calibration_file_no_ext, 'r') as f:
                    self.calibration_points = json.load(f)
                print(f"已从 {calibration_file_no_ext} 加载校准点")
            else:
                print(f"未找到校准文件 {calibration_file_no_ext}")
                self.calibration_points = None
        except Exception as e:
            print(f"加载校准文件时出错: {e}")
            self.calibration_points = None

    def calibrate_image(self, points=None):
        """
        校准图像，按照校准点进行透视变换

        参数:
            points: 校准点坐标，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                   如果为None，则使用初始化时提供的点或者从校准文件加载
        """
        if points is not None:
            self.calibration_points = points
            print("使用外部提供的校准点进行校准。")
            # 如果外部提供了点，则不再尝试从文件加载
        elif self.calibration_points is None:
            # 仅当未提供外部点且内部点为 None 时才尝试加载文件
            # Load calibration implicitly tries to load if self.calibration_points is None
            # No specific call needed here if load_image was called first
            pass # Loading is attempted in load_image if points were None initially

        if self.calibration_points is None:
            # 如果没有提供校准点，使用整个图像
            h, w = self.original_image.shape[:2]
            self.calibration_points = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
            self.calibrated_image = self.original_image.copy()
            # 设置校准区域像素为整个图像的像素
            self.calibration_area_pixels = h * w
            print(f"警告: 未找到校准点，使用整个图像区域 ({w}x{h}) 进行分析。密度计算将基于此区域假定为1平方米。")
            return self.calibrated_image

        # 确保校准点是按顺时针排序的
        points_array = np.array(self.calibration_points)

        # 计算校准区域的边界框
        x_min = int(np.min(points_array[:, 0]))
        y_min = int(np.min(points_array[:, 1]))
        x_max = int(np.max(points_array[:, 0]))
        y_max = int(np.max(points_array[:, 1]))

        # 确保边界在图像范围内
        h, w = self.original_image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w-1, x_max)
        y_max = min(h-1, y_max)

        # 保存剪裁区域的坐标，用于后续计算
        self.crop_region = (x_min, y_min, x_max, y_max)

        # 记录校准区域的实际面积（像素数）
        self.calibration_area_pixels = (
            x_max - x_min + 1) * (y_max - y_min + 1)

        # 执行透视变换
        # 源点（输入图像中的四个角点）
        src_points = np.array(self.calibration_points, dtype=np.float32)

        # 计算目标区域的宽度和高度
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # 目标点（输出图像中的四个角点）
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        self.calibrated_image = cv2.warpPerspective(
            self.original_image, M, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # 打印校准信息
        print(f"校准区域: 左上({x_min}, {y_min}), 右下({x_max}, {y_max})")
        print(f"校准区域尺寸: {width} x {height} 像素")
        print(f"已应用透视变换")

        return self.calibrated_image

    def _auto_tune_parameters(self):
        """
        根据图像特性自动调整分割参数

        返回:
            参数字典
        """
        if self.calibrated_image is None:
            raise ValueError("请先校准图像")

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(self.calibrated_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 计算色调直方图
        h_hist = cv2.calcHist([h], [0], None, [180], [0, 180])

        # 找到主要的绿色峰值
        green_range = np.arange(35, 90)
        green_hist = h_hist[green_range]
        green_peak = green_range[np.argmax(green_hist)] if np.max(
            green_hist) > 0 else 60

        # 根据绿色峰值调整HSV范围
        green_width = 25  # 默认宽度
        lower_h = max(30, green_peak - green_width)
        upper_h = min(100, green_peak + green_width)

        # 根据饱和度和亮度分布调整阈值
        s_mean, s_std = np.mean(s), np.std(s)
        v_mean, v_std = np.mean(v), np.std(v)

        # 更严格的饱和度和亮度阈值
        lower_s = max(25, int(s_mean - 1.5 * s_std))
        upper_s = 255
        lower_v = max(25, int(v_mean - 1.5 * v_std))
        upper_v = 255

        # 获取校准区域的实际尺寸
        height, width = self.calibrated_image.shape[:2]
        calibration_size = width * height

        # 根据校准区域的实际尺寸调整参数
        min_region_size = max(20, int(50 * (calibration_size / 250000)))
        min_distance = max(5, int(10 * (height / 500)))

        # 返回自适应参数
        return {
            'lower_green': np.array([lower_h, lower_s, lower_v]),
            'upper_green': np.array([upper_h, upper_s, upper_v]),
            'min_region_size': min_region_size,
            'min_distance': min_distance
        }

    def segment_grass(self, method='hsv'):
        """
        分割图像中的草，主要基于S（饱和度）通道进行分割，并进行后处理优化

        参数:
            method: 分割方法，目前只支持'hsv'

        返回:
            草的掩码图像
        """
        if self.calibrated_image is None:
            raise ValueError("请先校准图像")

        # 1. 图像预处理：增强对比度
        # 转换到LAB颜色空间进行对比度增强
        lab = cv2.cvtColor(self.calibrated_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 对L通道进行CLAHE处理，增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 合并通道并转回RGB
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # 2. 计算图像整体亮度
        gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)

        # 3. 使用HSV分析器获取最佳阈值
        # 保存临时图像用于分析
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, 'temp_for_analysis.png')
        cv2.imwrite(temp_image_path, cv2.cvtColor(
            enhanced_rgb, cv2.COLOR_RGB2BGR))

        # 分析图像获取HSV阈值
        analysis_image, stats, suggested_thresholds = analyze_hsv_thresholds(
            temp_image_path)
        if analysis_image is not None:
            # 保存HSV分析结果到调试图像
            self.debug_images['hsv_analysis'] = analysis_image

        # 删除临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # 4. 转换为HSV并分离通道
        hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 5. 主要基于S通道进行分割，同时考虑H和V通道
        # 计算S通道的统计信息
        s_mean, s_std = np.mean(s), np.std(s)

        # 根据亮度动态调整S通道阈值
        if mean_brightness < 100:  # 暗图片
            lower_s = max(40, int(s_mean - 1.2 * s_std))  # 提高最小阈值
            upper_s = 255
        else:  # 亮图片
            lower_s = max(50, int(s_mean - 1.8 * s_std))  # 提高最小阈值
            upper_s = 255

        # 创建S通道掩码
        s_mask = cv2.threshold(s, lower_s, 255, cv2.THRESH_BINARY)[1]

        # 保存S通道掩码用于调试
        self.debug_images['s_channel_mask'] = s_mask

        # 6. 使用H通道进行绿色识别
        # 设置更严格的H范围，主要针对绿色
        lower_h = 35  # 提高下限
        upper_h = 85  # 降低上限
        h_mask = cv2.inRange(h, lower_h, upper_h)

        # 7. 使用V通道排除黑色干扰
        # 计算V通道的统计信息
        v_mean, v_std = np.mean(v), np.std(v)
        lower_v = max(40, int(v_mean - 1.5 * v_std))  # 提高最小亮度阈值
        upper_v = 255
        v_mask = cv2.inRange(v, lower_v, upper_v)

        # 8. 组合H、S和V通道的掩码
        self.mask = cv2.bitwise_and(s_mask, h_mask)
        self.mask = cv2.bitwise_and(self.mask, v_mask)

        # 保存组合掩码用于调试
        self.debug_images['combined_mask'] = self.mask

        # 保存原始掩码用于计算盖度
        self.hsv_mask = self.mask.copy()

        # 9. 改进掩码后处理
        # 使用更大的核进行形态学操作
        kernel = np.ones((7, 7), np.uint8)

        # 先进行闭运算连接相近区域
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # 再进行开运算去除噪声
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)

        # 10. 基于草叶特性的优化
        # 使用距离变换和分水岭算法优化边缘
        dist_transform = cv2.distanceTransform(self.mask, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        # 使用自适应阈值进行分水岭
        dist_mean = np.mean(dist_transform[dist_transform > 0])
        dist_std = np.std(dist_transform[dist_transform > 0])
        threshold = max(0.3, dist_mean - 0.5 * dist_std)

        # 创建标记
        _, sure_fg = cv2.threshold(dist_transform, threshold, 1, 0)
        sure_fg = sure_fg.astype(np.uint8) * 255

        # 寻找未知区域
        sure_bg = cv2.dilate(self.mask, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 应用分水岭
        markers = cv2.watershed(self.calibrated_image, markers.copy())
        watershed_mask = np.zeros_like(self.mask)
        watershed_mask[markers > 1] = 255

        # 11. 合并原始掩码和分水岭结果
        self.mask = cv2.bitwise_or(self.mask, watershed_mask)

        # 12. 移除小区域
        min_size = max(30, int(
            50 * (self.calibrated_image.shape[0] * self.calibrated_image.shape[1] / 250000)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8)

        # 创建输出掩码
        cleaned_mask = np.zeros_like(self.mask)

        # 从1开始，因为0是背景
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255

        self.mask = cleaned_mask

        # 13. 最终优化：使用形态学操作平滑边缘
        kernel_smooth = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel_smooth)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel_smooth)

        # 保存最终掩码用于调试
        self.debug_images['final_mask'] = self.mask

        return self.mask

    def calculate_coverage(self):
        """
        计算草的盖度，使用原始HSV掩码的结果

        返回:
            盖度百分比
        """
        if not hasattr(self, 'hsv_mask') or self.hsv_mask is None:
            # 如果没有hsv_mask，先分割草
            self.segment_grass(method='hsv')

        # 计算草的像素数（使用hsv_mask而不是最终的mask）
        grass_pixels = np.sum(self.hsv_mask > 0)

        # 计算总像素数
        total_pixels = self.hsv_mask.size

        # 计算盖度
        coverage = grass_pixels / total_pixels * 100

        return coverage

    def segment_instances(self, method='watershed'):
        """
        分割草的实例，基于HSV掩码进行分割以保持与盖度计算的一致性

        参数:
            method: 分割方法，'watershed'或'local_maxima'

        返回:
            标记的实例图像
        """
        if not hasattr(self, 'hsv_mask') or self.hsv_mask is None:
            # 如果没有hsv_mask，先分割草
            self.segment_grass(method='hsv')

        # 获取自适应参数
        params = self._auto_tune_parameters()

        # 对HSV掩码进行预处理，去除噪声并连接相近区域
        kernel = np.ones((3, 3), np.uint8)
        processed_mask = cv2.morphologyEx(
            self.hsv_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(
            processed_mask, cv2.MORPH_OPEN, kernel)

        # 首先进行距离变换
        dist_transform = cv2.distanceTransform(processed_mask, cv2.DIST_L2, 5)

        # 归一化距离变换结果
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        # 保存距离变换结果用于调试
        self.debug_images['dist_transform'] = dist_transform * 255

        if method == 'watershed':
            # 使用改进的分水岭算法，调整参数以适应HSV掩码

            # 使用自适应阈值
            dist_mean = np.mean(dist_transform[dist_transform > 0])
            dist_std = np.std(dist_transform[dist_transform > 0])
            # 调整阈值以获取合适数量的种子点
            threshold = max(0.2, dist_mean - 0.7 * dist_std)  # 降低阈值以检测更多潜在的草实例

            # 阈值化距离变换结果以获取种子点
            _, sure_fg = cv2.threshold(dist_transform, threshold, 1, 0)
            sure_fg = sure_fg.astype(np.uint8) * 255

            # 保存种子点用于调试
            self.debug_images['sure_fg'] = sure_fg

            # 应用形态学操作合并相近的种子点
            kernel = np.ones((3, 3), np.uint8)  # 使用小核以保留更多细节
            sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel)

            # 寻找未知区域
            sure_bg = cv2.dilate(processed_mask, np.ones(
                (3, 3), np.uint8), iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # 标记种子
            _, markers = cv2.connectedComponents(sure_fg)

            # 将标记加1以确保背景不是0
            markers = markers + 1

            # 将未知区域标记为0
            markers[unknown == 255] = 0

            # 应用分水岭算法
            markers = cv2.watershed(self.calibrated_image, markers.copy())

            # 存储实例标记
            self.instances = markers

        elif method == 'local_maxima':
            # 使用局部极大值方法，调整参数以适应HSV掩码

            # 调整最小距离参数
            min_distance = max(7, params['min_distance'])  # 减小最小距离以检测更多实例

            # 调整阈值以检测更多局部极大值
            coordinates = peak_local_max(dist_transform, min_distance=min_distance,
                                         threshold_abs=0.1, exclude_border=False)  # 降低阈值

            # 创建标记
            markers = np.zeros(dist_transform.shape, dtype=np.int32)
            for i, (x, y) in enumerate(coordinates):
                markers[x, y] = i + 1

            # 应用分水岭算法
            self.instances = skimage_watershed(-dist_transform,
                                               markers, mask=processed_mask)

        else:
            raise ValueError(f"不支持的实例分割方法: {method}")

        # 移除太小的实例
        min_size = params['min_region_size'] // 2  # 减小最小区域大小以保留更多实例
        unique_instances = np.unique(self.instances)

        # 创建新的实例标记
        new_instances = np.zeros_like(self.instances)
        new_id = 1  # 从1开始，0是背景

        # 移除太小的实例
        for instance_id in unique_instances:
            if instance_id <= 0:  # 跳过背景和边界
                continue

            # 计算当前实例的大小
            instance_mask = (self.instances == instance_id)
            size = np.sum(instance_mask)

            if size >= min_size:
                new_instances[instance_mask] = new_id
                new_id += 1

        self.instances = new_instances

        # 创建彩色实例图像用于调试
        instance_vis = np.zeros(
            (self.instances.shape[0], self.instances.shape[1], 3), dtype=np.uint8)
        for i in np.unique(self.instances):
            if i <= 0:  # 跳过背景和边界
                continue
            mask = self.instances == i
            color = np.random.randint(0, 256, 3)
            instance_vis[mask] = color

        self.debug_images['instances'] = instance_vis

        return self.instances

    def calculate_density(self, method='instance_count'):
        """
        计算草的密度，改进算法以避免将一株草的每一根叶子单独计算为一棵草

        参数:
            method: 密度计算方法，'instance_count'、'area_based'或'combined'

        返回:
            每平方米草的数量
        """
        if self.mask is None:
            self.segment_grass()

        # 增加检查确保 calibration_area_pixels 有效
        if self.calibration_area_pixels is None or self.calibration_area_pixels <= 0:
            print("[错误] 校准区域面积无效或未计算，无法计算密度。")
            # 尝试重新运行校准以计算面积
            if self.original_image is not None:
                print("尝试重新运行校准...")
                self.calibrate_image()
                if self.calibration_area_pixels is None or self.calibration_area_pixels <= 0:
                    return 0 # 或者 raise ValueError("无法获取有效的校准区域面积")
            else:
                return 0 # 或者 raise ValueError("无法获取有效的校准区域面积")

        # 获取校准区域的实际面积（像素数）
        if not hasattr(self, 'calibration_area_pixels'):
            # 如果没有校准区域信息，使用整个图像的面积
            self.calibration_area_pixels = self.calibrated_image.shape[0] * \
                self.calibrated_image.shape[1]

        # 假设校准区域是1平方米，计算每像素代表的实际面积（平方米）
        pixel_to_sqm = 1.0 / self.calibration_area_pixels

        if method == 'instance_count' or method == 'combined':
            if self.instances is None:
                self.segment_instances()

            # 计算实例数量
            unique_instances = np.unique(self.instances)
            num_instances = len([i for i in unique_instances if i > 0])

            if method == 'instance_count':
                return num_instances

        if method == 'area_based' or method == 'combined':
            grass_area = np.sum(self.mask > 0)
            avg_plant_size = 200
            area_based_count = max(1, int(grass_area / avg_plant_size))
            if method == 'area_based':
                return area_based_count

        if method == 'combined':
            if num_instances < 0.5 * area_based_count:
                logging.warning("[GrassAnalyzer] 实例分割可能不充分，使用组合方法估计密度")
                return max(num_instances, int(0.5 * (num_instances + area_based_count)))
            else:
                return num_instances

        raise ValueError(f"不支持的密度计算方法: {method}")

    def visualize_results(self, save_path=None, layout='default', save_debug=False, calculate_density=True):
        """可视化分析结果，包括原始图像、掩码、实例和密度信息，并选择性保存"""
        # logging.info(f"[Visualize] 开始生成结果图，目标路径: {save_path}, 布局: {layout}") # Remove start log

        if self.original_image is None:
            raise ValueError("请先加载图像")

        if self.calibrated_image is None or not hasattr(self, 'hsv_mask') or self.hsv_mask is None:
            raise ValueError("请先完成图像校准和草的分割")

        # 确保实例分割已完成（如果需要计算密度）
        if self.instances is None and layout == 'default' and calculate_density:
            self.segment_instances()

        # 计算盖度
        coverage = self.calculate_coverage()

        # 计算密度（如果需要）
        density = None
        if calculate_density and layout == 'default':
            density = self.calculate_density(method='instance_count')

        # 获取图片名称（如果有保存路径）
        image_name = ""
        if save_path:
            image_name = os.path.basename(save_path).split('_')[0]  # 提取图片名称
            if image_name:
                image_name = f" - {image_name}"

        # 根据布局样式选择不同的可视化方法
        if layout == 'simple':
            # 创建1x3的简化布局
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 根据是否支持中文选择标题
            if use_chinese:
                titles = [
                    f'原始图像{image_name}',
                    '校准后图像',
                    f'草的分割掩码 (HSV) - 盖度: {coverage:.2f}%'
                ]
            else:
                titles = [
                    f'Original Image{image_name}',
                    'Calibrated Image',
                    f'Grass Segmentation Mask (HSV) - Coverage: {coverage:.2f}%'
                ]

            # 1. 原始图像
            axes[0].imshow(self.original_image)
            axes[0].set_title(titles[0])
            axes[0].axis('off')

            # 2. 校准后图像
            axes[1].imshow(self.calibrated_image)
            axes[1].set_title(titles[1])
            axes[1].axis('off')

            # 3. 分割掩码（使用HSV掩码）
            axes[2].imshow(self.hsv_mask, cmap='gray')
            axes[2].set_title(titles[2])
            axes[2].axis('off')

        else:  # 默认3x2布局
            # 创建3x2的结果图像布局
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))

            # 根据是否支持中文选择标题
            if use_chinese:
                titles = [
                    f'原始图像{image_name}',
                    '校准后图像',
                    '草的分割掩码 (HSV)',
                    '实例分割' if calculate_density else '未计算实例分割',
                    f'草的盖度: {coverage:.2f}%',
                    f'草的密度: {density} 株/平方米' if calculate_density else '未计算密度'
                ]
            else:
                titles = [
                    f'Original Image{image_name}',
                    'Calibrated Image',
                    'Grass Segmentation Mask (HSV)',
                    'Instance Segmentation' if calculate_density else 'Instance Segmentation Not Calculated',
                    f'Grass Coverage: {coverage:.2f}%',
                    f'Grass Density: {density} plants/m²' if calculate_density else 'Density Not Calculated'
                ]

            # 1. 原始图像
            axes[0, 0].imshow(self.original_image)
            axes[0, 0].set_title(titles[0])
            axes[0, 0].axis('off')

            # 2. 校准后图像
            axes[0, 1].imshow(self.calibrated_image)
            axes[0, 1].set_title(titles[1])
            axes[0, 1].axis('off')

            # 3. 分割掩码（使用HSV掩码）
            axes[1, 0].imshow(self.hsv_mask, cmap='gray')
            axes[1, 0].set_title(titles[2])
            axes[1, 0].axis('off')

            # 4. 实例分割可视化
            if calculate_density and 'instances' in self.debug_images:
                axes[1, 1].imshow(self.debug_images['instances'])
                # 添加实例编号
                unique_instances = np.unique(self.instances)
                for i in unique_instances:
                    if i <= 0:  # 跳过背景和边界
                        continue
                    mask = self.instances == i
                    y, x = np.mean(np.where(mask), axis=1)
                    axes[1, 1].text(x, y, str(i), color='white',
                                    fontsize=10, ha='center', va='center')
            else:
                # 显示灰色背景，表示未计算
                axes[1, 1].imshow(np.ones_like(self.calibrated_image) * 200)
            axes[1, 1].set_title(titles[3])
            axes[1, 1].axis('off')

            # 5. 盖度可视化
            coverage_vis = self.calibrated_image.copy()
            green_mask = np.zeros_like(coverage_vis)
            green_mask[self.hsv_mask > 0] = [0, 255, 0]
            coverage_vis = cv2.addWeighted(
                coverage_vis, 0.7, green_mask, 0.3, 0)
            axes[2, 0].imshow(coverage_vis)
            axes[2, 0].set_title(titles[4])
            axes[2, 0].axis('off')

            # 6. 密度可视化（带有半透明实例标记）
            if calculate_density and self.instances is not None:
                # 创建基础图像
                density_vis = self.calibrated_image.copy()

                # 创建实例标记的彩色图像
                instance_overlay = np.zeros_like(density_vis)
                unique_instances = np.unique(self.instances)
                colors = plt.cm.rainbow(np.linspace(
                    0, 1, len([i for i in unique_instances if i > 0])))
                colors = (colors[:, :3] * 255).astype(np.uint8)

                instance_id = 0
                for i in unique_instances:
                    if i <= 0:
                        continue
                    mask = self.instances == i
                    instance_overlay[mask] = colors[instance_id]
                    instance_id += 1

                # 合并图像
                density_vis = cv2.addWeighted(
                    density_vis, 0.7, instance_overlay, 0.3, 0)

                # 添加实例编号
                for i in unique_instances:
                    if i <= 0:
                        continue
                    mask = self.instances == i
                    y, x = np.mean(np.where(mask), axis=1)
                    axes[2, 1].text(x, y, str(i), color='white',
                                    fontsize=10, ha='center', va='center',
                                    bbox=dict(facecolor='black', alpha=0.5))

                axes[2, 1].imshow(density_vis)
            else:
                # 显示灰色背景，表示未计算
                axes[2, 1].imshow(np.ones_like(self.calibrated_image) * 200)
            axes[2, 1].set_title(titles[5])
            axes[2, 1].axis('off')

        # 调整子图之间的间距
        plt.tight_layout()

        # 如果没有提供保存路径，使用默认路径
        if save_path is None:
            save_path = 'results/analysis_result.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # 确保目录存在

        # 初始化返回的路径字典
        saved_paths = {}

        # 保存合并后的图形
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved_paths['analysis_image'] = save_path # 主分析图
            logging.info(f"[Visualize] 结果图像已保存: {os.path.basename(save_path)}") # Simplified log
        except Exception as e:
            logging.error(f"[Visualize] 保存合并结果图像 '{save_path}' 时出错: {e}")
            save_path = None
        finally:
            plt.close(fig)

        # 保存调试图像 (如果 save_debug=True)
        if save_debug:
            output_dir = os.path.dirname(save_path)
            base_filename = os.path.splitext(os.path.basename(save_path))[0]
            base_filename = base_filename.replace('_analysis', '').replace('_default', '').replace('_simple', '')

            debug_images_to_save = {
                'original': self.original_image,
                'calibrated': self.calibrated_image,
                'hsv_mask': self.hsv_mask,
                'coverage_overlay': coverage_vis if 'coverage_vis' in locals() else None,
            }
            if calculate_density:
                 debug_images_to_save.update({
                     'instance_mask': self.instances if self.instances is not None else None,
                     'instance_overlay': self.debug_images.get('instances'),
                     'density_overlay': density_vis if 'density_vis' in locals() else None,
                 })

            for key, img_data in debug_images_to_save.items():
                if img_data is not None:
                    img_to_write = None
                    # 确保图像是 BGR 或 Grayscale uint8
                    if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                        # --- FIX: Convert RGB to BGR for ALL 3-channel images --- #
                        if np.issubdtype(img_data.dtype, np.floating):
                             img_data = (img_data * 255).astype(np.uint8)
                        # Assume input is RGB unless known otherwise (e.g., from cv2 functions)
                        # Convert to BGR for imwrite
                        img_to_write = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        # --- END FIX --- #
                    elif len(img_data.shape) == 2: # Grayscale or mask
                         if np.issubdtype(img_data.dtype, np.integer) and img_data.max() > 1: # e.g. instance mask
                             # 尝试转换为可显示的灰度图
                             img_to_write = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                         elif np.issubdtype(img_data.dtype, np.bool_) or img_data.max() <= 1: # binary mask (allow max 1)
                            img_to_write = (img_data * 255).astype(np.uint8)
                         else:
                            # Handle other integer types if necessary (e.g., already uint8 grayscale)
                            if img_data.dtype == np.uint8:
                                 img_to_write = img_data
                            else:
                                 print(f"[Visualize] Warning: Unhandled 2-channel image type for key '{key}': dtype={img_data.dtype}")
                                 continue # Skip saving this debug image
                    else:
                        print(f"[Visualize] Warning: Unhandled image shape for key '{key}': shape={img_data.shape}")
                        continue # Skip saving

                    # Check if image conversion was successful
                    if img_to_write is None:
                         print(f"[Visualize] Warning: Image data for key '{key}' could not be prepared for saving.")
                         continue

                    debug_save_path = os.path.join(output_dir, f"{base_filename}_{key}_debug.png")
                    normalized_debug_save_path = os.path.normpath(debug_save_path)

                    # --- 修改：使用 imencode 和 open 处理 Unicode 路径 --- #
                    try:
                        # 1. 将图像编码为 PNG 格式的字节流
                        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3] # 0-9, 3 is default
                        result, buffer = cv2.imencode('.png', img_to_write, encode_param)

                        if not result:
                            print(f"[Visualize] Error: cv2.imencode 调试图像 [{key}] 失败。")
                            continue

                        # 2. 使用 Python 的 open 以二进制模式写入文件
                        with open(normalized_debug_save_path, 'wb') as f:
                            f.write(buffer)

                        # 写入成功
                        saved_paths[f'{key}_debug_image'] = normalized_debug_save_path
                        # print(f"[Visualize] 调试图像 [{key}] 已保存到: {normalized_debug_save_path}") # 日志可以保持简洁

                    except Exception as e:
                        print(f"[Visualize] 使用 imencode/open 保存调试图像 [{key}] 到 '{normalized_debug_save_path}' 时发生异常: {e}")
                    # --- 结束修改 --- #

        # 返回包含所有已保存路径的字典
        return saved_paths

    def _save_debug_image(self, key, image_data, output_dir, base_filename):
        """Helper function to save individual debug images. (此函数不再直接使用，逻辑合并到visualize_results)"""
        if image_data is None:
            return None # 返回 None 而不是直接退出

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 构建保存路径
        save_path = os.path.join(output_dir, f"{base_filename}_{key}_debug.png")

        try:
            # 检查图像数据类型和格式
            if isinstance(image_data, np.ndarray):
                 # 确保图像是 BGR 或 Grayscale uint8
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    # 如果是浮点数，先转换
                    if np.issubdtype(image_data.dtype, np.floating):
                         image_data = (image_data * 255).astype(np.uint8)
                    # OpenCV 需要 BGR
                    # image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR) # 不一定需要，取决于来源
                elif len(image_data.shape) == 2: # Grayscale or mask
                     if np.issubdtype(image_data.dtype, np.bool_): # binary mask
                         image_data = (image_data * 255).astype(np.uint8)
                     elif np.issubdtype(image_data.dtype, np.floating):
                          image_data = (image_data * 255).astype(np.uint8)


                cv2.imwrite(save_path, image_data)
                print(f"[Debug Save] 调试图像 [{key}] 已保存到: {save_path}")
                return save_path
            else:
                print(f"[Debug Save] 调试图像 [{key}] 数据格式不正确，跳过保存。")
                return None
        except Exception as e:
            print(f"[Debug Save] 保存调试图像 [{key}] 时出错: {e}")
            return None

def main():
    # 创建草地分析器实例
    analyzer = GrassAnalyzer()

    # 获取数据集中的所有图像
    image_dir = 'datasets'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if not image_files:
        print("未找到图像文件")
        return

    # 处理第一张图像作为示例
    image_path = os.path.join(image_dir, image_files[0])
    print(f"处理图像: {image_path}")

    # 加载图像
    analyzer.load_image(image_path)

    # 校准图像（使用整个图像）
    analyzer.calibrate_image()

    # 分割草
    analyzer.segment_grass(method='hsv')

    # 计算盖度
    coverage = analyzer.calculate_coverage()
    print(f"草的盖度: {coverage:.2f}%")

    # 分割实例并计算密度
    analyzer.segment_instances(method='local_maxima')
    density = analyzer.calculate_density(method='combined')
    print(f"草的密度: {density} 株/平方米")

    # 可视化结果
    os.makedirs('results', exist_ok=True)
    # 获取图片名称并去除后缀
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    analyzer.visualize_results(
        save_path=f'results/{image_basename}_analysis.png')
    print(f"结果已保存到 results/{image_basename}_analysis.png")


if __name__ == "__main__":
    main()
