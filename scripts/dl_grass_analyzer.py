import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.cluster import KMeans

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 测试中文字体是否可用


def is_chinese_font_available():
    try:
        font = plt.matplotlib.font_manager.FontProperties(family='SimHei')
        if font.get_name() == 'SimHei':
            return True
        return False
    except:
        return False


# 检查中文字体可用性
CHINESE_FONT_AVAILABLE = is_chinese_font_available()


class DeepLearningGrassAnalyzer:
    """使用深度学习方法的草地分析器"""

    def __init__(self, calibration_points=None):
        """
        初始化草地分析器

        参数:
            calibration_points: 校准点列表，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        self.calibration_points = calibration_points
        self.original_image = None
        self.calibrated_image = None
        self.mask = None
        self.coverage = None
        self.density = None
        self.debug_images = {}  # 用于存储调试图像

        # 加载预训练模型
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载DeepLabV3模型用于语义分割
        self.segmentation_model = self.load_segmentation_model()

        # 图像预处理转换
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def load_segmentation_model(self):
        """加载预训练的语义分割模型"""
        try:
            # 尝试使用更新的API加载模型
            try:
                from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
                model = deeplabv3_resnet50(
                    weights=DeepLabV3_ResNet50_Weights.DEFAULT)
                print("成功加载DeepLabV3 ResNet50模型(新API)")
            except:
                # 如果新API不可用，使用旧API
                model = deeplabv3_resnet50(pretrained=True)
                print("成功加载DeepLabV3 ResNet50模型(旧API)")

            model.eval()
            model = model.to(self.device)
            return model
        except Exception as e:
            print(f"加载语义分割模型失败: {e}")
            print("将使用备用方法")
            return None

    def load_image(self, image_path):
        """
        加载图像 (支持非ASCII路径)

        参数:
            image_path: 图像文件路径
        """
        # self.original_image = cv2.imread(image_path)
        # 使用 imdecode 处理非 ASCII 路径
        try:
            n = np.fromfile(image_path, dtype=np.uint8)
            self.original_image = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception as e:
             raise ValueError(f"使用 numpy/cv2.imdecode 读取图像时出错 {image_path}: {e}")

        if self.original_image is None:
            # raise ValueError(f"无法加载图像: {image_path}")
            raise ValueError(f"无法读取或解码图像 {image_path} (cv2.imdecode 返回 None)")

        # 转换为RGB（OpenCV默认是BGR）
        self.original_image = cv2.cvtColor(
            self.original_image, cv2.COLOR_BGR2RGB)
        return self.original_image

    def calibrate_image(self, points=None):
        """
        校准图像，将四个点定义的区域转换为标准的1平方米区域

        参数:
            points: 校准点列表，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                   如果为None，则使用初始化时提供的点

        返回:
            校准后的图像
        """
        # 优先使用传入的 points 参数
        if points is not None:
            self.calibration_points = points
            print("使用外部提供的校准点进行校准。")
        # 如果没有传入 points，则保留 self.calibration_points (可能在 __init__ 中设置)

        # 如果没有校准点，使用整个图像
        if self.calibration_points is None:
            self.calibrated_image = self.original_image.copy()
            return self.calibrated_image

        # 源点
        src_points = np.array(self.calibration_points, dtype=np.float32)

        # 目标点 - 定义一个标准的1平方米区域（500x500像素）
        dst_points = np.array([
            [0, 0],
            [500, 0],
            [500, 500],
            [0, 500]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        self.calibrated_image = cv2.warpPerspective(
            self.original_image, M, (500, 500))

        return self.calibrated_image

    def _auto_tune_parameters(self):
        """根据图像特性自动调整参数"""
        # 分析图像特性
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
        green_width = 25  # 减小宽度，提高精度
        lower_h = max(0, green_peak - green_width)
        upper_h = min(180, green_peak + green_width)

        # 根据饱和度和亮度分布调整阈值
        s_mean, s_std = np.mean(s), np.std(s)
        v_mean, v_std = np.mean(v), np.std(v)

        # 更严格的饱和度和亮度阈值
        lower_s = max(30, int(s_mean - s_std))  # 最小饱和度为30
        upper_s = 255
        lower_v = max(30, int(v_mean - v_std))  # 最小亮度为30
        upper_v = 255

        # 返回自适应参数
        return {
            'lower_green': np.array([lower_h, lower_s, lower_v]),
            'upper_green': np.array([upper_h, upper_s, upper_v]),
            'dl_threshold': 0.15 if np.mean(s) > 100 else 0.1,  # 提高阈值
            # 根据图像大小调整
            'min_region_size': int(100 * (self.calibrated_image.size / 250000))
        }

    def _enhanced_color_segmentation(self):
        """增强的颜色分割方法，结合多种颜色空间和自适应阈值"""
        # 获取自适应参数
        params = self._auto_tune_parameters()

        # HSV空间分割
        hsv = cv2.cvtColor(self.calibrated_image, cv2.COLOR_RGB2HSV)
        hsv_mask = cv2.inRange(
            hsv, params['lower_green'], params['upper_green'])

        # LAB空间分割 - 更好地分离绿色植物
        lab = cv2.cvtColor(self.calibrated_image, cv2.COLOR_RGB2LAB)
        # 提取a通道（绿色-红色）
        _, a, _ = cv2.split(lab)
        # 阈值化a通道，a值小的是绿色
        _, a_mask = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY_INV)

        # 结合HSV和LAB结果
        combined_mask = cv2.bitwise_and(hsv_mask, a_mask)

        # 使用K-means进行颜色聚类
        pixels = self.calibrated_image.reshape(-1, 3).astype(np.float32)
        k = 5  # 聚类数
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 找出最可能是草的聚类
        # 计算每个聚类中心在HSV空间中的值
        centers_hsv = cv2.cvtColor(centers.reshape(1, k, 3).astype(
            np.uint8), cv2.COLOR_RGB2HSV).reshape(k, 3)

        # 找出绿色聚类 - 使用更严格的条件
        green_clusters = []
        for i in range(k):
            h, s, v = centers_hsv[i]
            # 检查是否在绿色范围内，增加饱和度和亮度的限制
            if params['lower_green'][0] <= h <= params['upper_green'][0] and s > 60 and v > 50:
                green_clusters.append(i)

        # 创建聚类掩码
        kmeans_mask = np.zeros(self.calibrated_image.shape[:2], dtype=np.uint8)
        labels = labels.reshape(self.calibrated_image.shape[:2])
        for cluster in green_clusters:
            kmeans_mask[labels == cluster] = 255

        # 结合所有掩码 - 使用逻辑与操作，更严格的条件
        final_mask = cv2.bitwise_and(combined_mask, kmeans_mask)

        # 如果结果太少，使用逻辑或操作
        if np.sum(final_mask > 0) < 0.05 * final_mask.size:
            final_mask = cv2.bitwise_or(combined_mask, kmeans_mask)

        # 应用形态学操作改善结果
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # 移除小区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8)
        min_size = params['min_region_size']

        # 创建输出掩码
        cleaned_mask = np.zeros_like(final_mask)

        # 从1开始，因为0是背景
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255

        # 保存调试图像
        self.debug_images['hsv_mask'] = hsv_mask
        self.debug_images['lab_mask'] = a_mask
        self.debug_images['kmeans_mask'] = kmeans_mask
        self.debug_images['combined_mask'] = combined_mask
        self.debug_images['final_mask'] = final_mask
        self.debug_images['cleaned_mask'] = cleaned_mask

        return cleaned_mask

    def segment_grass(self):
        """
        使用多模型集成进行草地分割

        返回:
            分割掩码
        """
        # 如果深度学习模型加载失败，使用增强的颜色分割
        if self.segmentation_model is None:
            print("使用增强的颜色分割方法")
            self.mask = self._enhanced_color_segmentation()
            return self.mask

        try:
            # 准备输入
            input_image = Image.fromarray(self.calibrated_image)
            input_tensor = self.preprocess(
                input_image).unsqueeze(0).to(self.device)

            # 获取自适应参数
            params = self._auto_tune_parameters()

            # 深度学习分割
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
                output_softmax = F.softmax(output, dim=0).cpu().numpy()

            # 增强的颜色分割
            color_mask = self._enhanced_color_segmentation()

            # 多类别集成
            # 尝试更多可能的植物相关类别
            potential_plant_classes = [15, 21, 22, 10, 8, 7, 17, 19]
            dl_mask = np.zeros_like(color_mask)

            # 使用加权投票机制
            for class_id in potential_plant_classes:
                if class_id < output_softmax.shape[0]:
                    class_prob = output_softmax[class_id]
                    # 使用自适应阈值
                    threshold = params['dl_threshold']
                    weight = 1.0 if class_id in [15, 21] else 0.7  # 主要类别权重更高
                    class_mask = (class_prob > threshold).astype(
                        np.float32) * weight * 255
                    dl_mask = np.maximum(dl_mask, class_mask).astype(np.uint8)

            # 结合策略：使用逻辑与操作，要求同时满足颜色特征和深度学习预测
            combined_mask = cv2.bitwise_and(color_mask, dl_mask)

            # 如果结合后的掩码太小，使用加权组合
            if np.sum(combined_mask > 0) < 0.05 * combined_mask.size:
                print("逻辑与结果太小，使用加权组合")
                # 使用加权组合
                alpha = 0.6  # 颜色特征权重
                beta = 0.4   # 深度学习权重
                combined_mask = cv2.addWeighted(
                    color_mask, alpha,
                    dl_mask, beta,
                    0
                )
                # 二值化
                _, combined_mask = cv2.threshold(
                    combined_mask, 50, 255, cv2.THRESH_BINARY)

            # 自适应判断
            if np.sum(combined_mask > 0) < 0.02 * combined_mask.size:
                print("深度学习分割结果不理想，使用增强的颜色特征分割")
                self.mask = color_mask
            else:
                self.mask = combined_mask

            # 形态学后处理
            kernel = np.ones((5, 5), np.uint8)
            self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
            self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

            # 移除小区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                self.mask, connectivity=8)
            min_size = params['min_region_size']

            # 创建输出掩码
            cleaned_mask = np.zeros_like(self.mask)

            # 从1开始，因为0是背景
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    cleaned_mask[labels == i] = 255

            self.mask = cleaned_mask

            # 检查分割结果是否合理
            coverage_percent = np.sum(self.mask > 0) / self.mask.size * 100
            print(f"分割结果覆盖率: {coverage_percent:.2f}%")
            if coverage_percent > 90:
                print("警告: 分割覆盖率过高，可能存在过度分割")
                # 如果覆盖率过高，使用更严格的颜色分割
                if coverage_percent > 95:
                    print("使用更严格的颜色分割")
                    # 更严格的HSV范围
                    hsv = cv2.cvtColor(
                        self.calibrated_image, cv2.COLOR_RGB2HSV)
                    strict_lower = np.array([40, 80, 50])
                    strict_upper = np.array([80, 255, 255])
                    strict_mask = cv2.inRange(hsv, strict_lower, strict_upper)

                    # 应用形态学操作
                    kernel = np.ones((5, 5), np.uint8)
                    strict_mask = cv2.morphologyEx(
                        strict_mask, cv2.MORPH_OPEN, kernel)
                    strict_mask = cv2.morphologyEx(
                        strict_mask, cv2.MORPH_CLOSE, kernel)

                    self.mask = strict_mask

            # 保存调试图像
            self.debug_images['dl_mask'] = dl_mask
            self.debug_images['combined_mask'] = combined_mask
            self.debug_images['final_mask'] = self.mask

            return self.mask

        except Exception as e:
            print(f"深度学习分割失败: {e}")
            print("使用增强的颜色分割方法")
            self.mask = self._enhanced_color_segmentation()
            return self.mask

    def calculate_coverage(self):
        """
        计算草的盖度

        返回:
            草的盖度百分比
        """
        if self.mask is None:
            self.segment_grass()

        # 计算草的像素数
        grass_pixels = np.sum(self.mask > 0)

        # 计算总像素数
        total_pixels = self.mask.shape[0] * self.mask.shape[1]

        # 计算盖度
        self.coverage = (grass_pixels / total_pixels) * 100

        return self.coverage

    def calculate_density(self):
        """
        使用语义分割结果估计草的密度

        返回:
            每平方米草的估计数量
        """
        if self.mask is None:
            self.segment_grass()

        # 使用连通组件分析估计草的数量
        # 这种方法假设每个连通区域代表一株或一簇草
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8)

        # 获取自适应参数
        params = self._auto_tune_parameters()
        min_size = params['min_region_size']

        # 计算有效草区域的数量（排除太小的区域）
        valid_regions = 0
        for i in range(1, num_labels):  # 从1开始，因为0是背景
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                valid_regions += 1

        # 如果连通区域太少，尝试使用更复杂的方法估计
        if valid_regions < 3 and np.sum(self.mask > 0) > 0.1 * self.mask.size:
            # 使用距离变换和局部极大值估计草的数量
            dist_transform = cv2.distanceTransform(self.mask, cv2.DIST_L2, 5)

            # 归一化距离变换
            cv2.normalize(dist_transform, dist_transform,
                          0, 1.0, cv2.NORM_MINMAX)

            # 找到局部极大值
            kernel = np.ones((7, 7), np.uint8)
            dilated = cv2.dilate(dist_transform, kernel)
            local_max = (dist_transform == dilated) & (dist_transform > 0.5)

            # 计算局部极大值的数量
            valid_regions = np.sum(local_max)

            # 保存调试图像
            self.debug_images['dist_transform'] = dist_transform * 255
            self.debug_images['local_max'] = local_max.astype(np.uint8) * 255

        self.density = valid_regions

        return self.density

    def visualize_results(self, save_path=None, layout='default', save_debug=False):
        """
        可视化分析结果

        参数:
            save_path: 保存结果图像的路径
            layout: 布局样式，'default'为2x2布局，'simple'为1x3简化布局
            save_debug: 是否保存调试图像

        返回:
            结果图像
        """
        if self.mask is None:
            self.segment_grass()

        if self.coverage is None:
            self.calculate_coverage()

        if layout == 'simple':
            # 创建1x3的简化布局
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 设置标题
            titles = [
                "原始图像" if CHINESE_FONT_AVAILABLE else "Original Image",
                "校准后的1平方米区域" if CHINESE_FONT_AVAILABLE else "Calibrated 1 Square Meter Area",
                f"草的分割掩码 - 盖度: {self.coverage:.2f}%" if CHINESE_FONT_AVAILABLE else f"Grass Segmentation Mask - Coverage: {self.coverage:.2f}%"
            ]

            # 1. 原始图像
            axes[0].imshow(self.original_image)
            axes[0].set_title(titles[0])
            axes[0].axis('off')

            # 2. 校准后图像
            axes[1].imshow(self.calibrated_image)
            axes[1].set_title(titles[1])
            axes[1].axis('off')

            # 3. 分割掩码
            axes[2].imshow(self.mask, cmap='gray')
            axes[2].set_title(titles[2])
            axes[2].axis('off')

        else:
            # 默认布局 - 2x2
            if self.density is None:
                self.calculate_density()

            # 创建图像网格
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # 原始图像
            axs[0, 0].imshow(self.original_image)
            title = "原始图像" if CHINESE_FONT_AVAILABLE else "Original Image"
            axs[0, 0].set_title(title)
            axs[0, 0].axis('off')

            # 校准后的图像
            axs[0, 1].imshow(self.calibrated_image)
            title = "校准后的1平方米区域" if CHINESE_FONT_AVAILABLE else "Calibrated 1 Square Meter Area"
            axs[0, 1].set_title(title)
            axs[0, 1].axis('off')

            # 分割结果
            overlay = self.calibrated_image.copy()
            # 创建半透明覆盖
            green_overlay = np.zeros_like(overlay)
            green_overlay[self.mask > 0] = [0, 255, 0]  # 绿色标记草
            # 混合原始图像和覆盖层
            overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)

            axs[1, 0].imshow(overlay)
            title = f"草的盖度: {self.coverage:.2f}%" if CHINESE_FONT_AVAILABLE else f"Grass Coverage: {self.coverage:.2f}%"
            axs[1, 0].set_title(title)
            axs[1, 0].axis('off')

            # 密度估计结果可视化
            # 使用连通组件标记
            num_labels, labels = cv2.connectedComponents(self.mask)

            # 为每个连通区域分配随机颜色
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
            labeled_img[labels == 0] = 0  # 背景设为黑色

            # 创建半透明覆盖
            density_overlay = cv2.addWeighted(
                self.calibrated_image, 0.7, labeled_img, 0.3, 0)

            # 显示密度估计结果
            axs[1, 1].imshow(density_overlay)
            title = f"草的密度: {self.density} 株/平方米" if CHINESE_FONT_AVAILABLE else f"Grass Density: {self.density} plants/m²"
            axs[1, 1].set_title(title)
            axs[1, 1].axis('off')

            # 添加总标题
            title = "草地分析结果 (语义分割方法)" if CHINESE_FONT_AVAILABLE else "Grass Analysis Results (Semantic Segmentation Method)"
            plt.suptitle(title, fontsize=16)

            plt.tight_layout()

        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到 {save_path}")

            # 保存调试图像
            if save_debug and self.debug_images:
                debug_dir = os.path.dirname(save_path)
                base_name = os.path.basename(save_path).split('.')[0]
                for name, img in self.debug_images.items():
                    debug_path = os.path.join(
                        debug_dir, f"{base_name}_{name}.png")
                    if len(img.shape) == 2 or img.shape[2] == 1:  # 灰度图像
                        plt.imsave(debug_path, img, cmap='gray')
                    else:  # 彩色图像
                        plt.imsave(debug_path, img)

        return fig

    def interactive_calibration(self):
        """交互式校准图像"""
        from calibration_tool import calibrate_image
        calibration_points = calibrate_image(self.original_image)
        self.calibration_points = calibration_points
        self.calibrate_image()
        return calibration_points
