import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib
import json
import matplotlib.cm as cm

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei',
                                          'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'


class LidarHeightAnalyzer:
    """
    基于激光雷达点云数据分析杂草高度的工具

    该分析器使用DBSCAN聚类算法将点云数据分为地面点和非地面点（杂草点），
    然后计算杂草的高度。
    """

    def __init__(self):
        """初始化激光雷达高度分析器"""
        self.point_cloud = None
        self.ground_points = None
        self.grass_points = None
        self.ground_height = None
        self.max_height = None
        self.grass_height = None
        self.valid_points_count = 0
        self.total_points_count = 0
        self.invalid_threshold = 1500.0  # 无效数据阈值
        self.clustering_method = 'adaptive'  # 聚类方法: 'dbscan', 'kmeans', 'adaptive'

    def load_point_cloud(self, file_path):
        """
        从文件加载点云数据

        参数:
            file_path: 点云数据文件路径

        返回:
            成功加载返回True，否则返回False
        """
        if not os.path.exists(file_path):
            print(f"[错误] 点云数据文件不存在: {file_path}")
            return False

        try:
            # 读取文件内容
            with open(file_path, 'r') as f:
                content = f.read()

            # 解析点云数据
            # 移除所有换行符并按逗号分割
            points_str = content.replace('\n', '').split(',')

            # 过滤空字符串并转换为浮点数
            points = [float(p.strip()) for p in points_str if p.strip()]

            # 记录总点数
            self.total_points_count = len(points)

            # 过滤无效数据（大于阈值的点）
            valid_points = [p for p in points if p < self.invalid_threshold]
            self.valid_points_count = len(valid_points)

            if self.valid_points_count == 0:
                print(f"[错误] 没有有效的点云数据")
                return False

            # 存储点云数据
            self.point_cloud = np.array(valid_points)

            print(
                f"[信息] 成功加载点云数据: {self.valid_points_count}/{self.total_points_count} 个有效点")
            return True

        except Exception as e:
            print(f"[错误] 加载点云数据失败: {str(e)}")
            return False

    def analyze_height(self, eps=0.3, min_samples=2):
        """
        分析点云数据，计算杂草高度

        参数:
            eps: DBSCAN算法的邻域半径参数
            min_samples: DBSCAN算法的最小样本数参数

        返回:
            分析成功返回True，否则返回False
        """
        if self.point_cloud is None or len(self.point_cloud) == 0:
            print("[错误] 未加载点云数据或数据为空")
            return False

        try:
            # 将点云数据重塑为适合聚类的形式
            X = self.point_cloud.reshape(-1, 1)

            # 根据选择的聚类方法进行聚类
            if self.clustering_method == 'dbscan':
                # 使用改进的DBSCAN参数
                labels = self._cluster_with_dbscan(X, eps, min_samples)
            elif self.clustering_method == 'kmeans':
                # 使用K-means聚类（假设有两类：地面和草）
                labels = self._cluster_with_kmeans(X)
            else:  # 'adaptive'
                # 尝试自适应选择最佳聚类方法
                labels = self._cluster_adaptive(X, eps, min_samples)

            # 如果聚类失败，返回False
            if labels is None:
                return False

            # 分析聚类结果
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1) if -1 in unique_labels else 0

            print(f"[信息] 聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")

            if n_clusters < 1:
                print("[警告] 未能有效聚类")
                return False

            # 如果只有一个聚类，尝试使用简单阈值法分离地面和草
            if n_clusters == 1 and -1 not in unique_labels:
                print("[信息] 只有一个聚类，尝试使用阈值法分离地面和草")
                return self._analyze_with_threshold()

            # 找出地面点（距离较大的点）和草点（距离较小的点）
            cluster_means = {}
            for label in unique_labels:
                if label != -1:  # 忽略噪声
                    cluster_means[label] = np.mean(X[labels == label])

            # 如果没有有效聚类，返回False
            if not cluster_means:
                print("[错误] 没有有效的聚类")
                return False

            # 距离较大的聚类为地面点，距离较小的聚类为草点
            ground_label = max(cluster_means, key=cluster_means.get)

            # 提取地面点和草点
            self.ground_points = self.point_cloud[labels == ground_label]

            # 草点是除了地面点和噪声点之外的所有点
            grass_mask = np.logical_and(labels != ground_label, labels != -1)
            self.grass_points = self.point_cloud[grass_mask] if np.any(
                grass_mask) else np.array([])

            # 计算地面高度（地面点的平均值）
            self.ground_height = np.mean(self.ground_points) if len(
                self.ground_points) > 0 else None

            # 计算最大高度（所有有效点中的最小值，因为距离越小高度越高）
            self.max_height = np.min(self.point_cloud) if len(
                self.point_cloud) > 0 else None

            # 计算草的高度（地面高度 - 最大高度）
            if self.ground_height is not None and self.max_height is not None:
                self.grass_height = self.ground_height - self.max_height
                print(
                    f"[信息] 地面高度: {self.ground_height:.3f}m, 最大高度点: {self.max_height:.3f}m")
                print(f"[信息] 计算的草高: {self.grass_height:.3f}m")
                return True
            else:
                print("[错误] 无法计算草高")
                return False

        except Exception as e:
            print(f"[错误] 分析高度失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _cluster_with_dbscan(self, X, eps, min_samples):
        """使用DBSCAN进行聚类"""
        try:
            # 标准化数据
            X_scaled = StandardScaler().fit_transform(X)

            # 使用DBSCAN进行聚类
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
            labels = db.labels_

            # 检查聚类结果
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 1:
                print("[警告] DBSCAN未能有效聚类，尝试调整参数")
                # 尝试不同的eps值
                for new_eps in [0.2, 0.4, 0.5, 0.6]:
                    if new_eps == eps:
                        continue
                    db = DBSCAN(eps=new_eps, min_samples=min_samples).fit(
                        X_scaled)
                    new_labels = db.labels_
                    new_n_clusters = len(set(new_labels)) - \
                        (1 if -1 in new_labels else 0)
                    if new_n_clusters >= 1:
                        print(f"[信息] 使用eps={new_eps}成功聚类")
                        return new_labels

                print("[警告] DBSCAN调整参数后仍未能有效聚类")
                return None

            return labels

        except Exception as e:
            print(f"[错误] DBSCAN聚类失败: {str(e)}")
            return None

    def _cluster_with_kmeans(self, X):
        """使用K-means进行聚类"""
        try:
            # 标准化数据
            X_scaled = StandardScaler().fit_transform(X)

            # 使用K-means聚类，假设有2个类别（地面和草）
            kmeans = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
            labels = kmeans.labels_

            return labels

        except Exception as e:
            print(f"[错误] K-means聚类失败: {str(e)}")
            return None

    def _cluster_adaptive(self, X, eps, min_samples):
        """自适应选择最佳聚类方法"""
        # 首先尝试DBSCAN
        dbscan_labels = self._cluster_with_dbscan(X, eps, min_samples)

        # 如果DBSCAN成功，使用DBSCAN结果
        if dbscan_labels is not None and len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) >= 1:
            print("[信息] 使用DBSCAN聚类结果")
            return dbscan_labels

        # 否则尝试K-means
        kmeans_labels = self._cluster_with_kmeans(X)

        # 如果K-means成功，使用K-means结果
        if kmeans_labels is not None:
            print("[信息] 使用K-means聚类结果")
            return kmeans_labels

        # 如果两种方法都失败，返回None
        print("[警告] 所有聚类方法都失败")
        return None

    def _analyze_with_threshold(self):
        """使用简单阈值法分析点云数据"""
        try:
            # 计算点云数据的均值和标准差
            mean = np.mean(self.point_cloud)
            std = np.std(self.point_cloud)

            # 使用高斯分布拟合数据
            # 假设距离较小的点（高于地面的点）是草点
            # 距离较大的点（接近地面的点）是地面点
            threshold = mean - 0.5 * std  # 使用均值减去0.5个标准差作为阈值

            # 分离地面点和草点
            self.ground_points = self.point_cloud[self.point_cloud >= threshold]
            self.grass_points = self.point_cloud[self.point_cloud < threshold]

            # 如果草点太少，调整阈值
            if len(self.grass_points) < 3:
                threshold = mean - 1.0 * std
                self.ground_points = self.point_cloud[self.point_cloud >= threshold]
                self.grass_points = self.point_cloud[self.point_cloud < threshold]

            # 计算地面高度（地面点的平均值）
            self.ground_height = np.mean(self.ground_points) if len(
                self.ground_points) > 0 else None

            # 计算最大高度（所有有效点中的最小值，因为距离越小高度越高）
            self.max_height = np.min(self.point_cloud) if len(
                self.point_cloud) > 0 else None

            # 计算草的高度
            if self.ground_height is not None and self.max_height is not None:
                self.grass_height = self.ground_height - self.max_height
                print(f"[信息] 使用阈值法: 阈值={threshold:.3f}m")
                print(
                    f"[信息] 地面高度: {self.ground_height:.3f}m, 最大高度点: {self.max_height:.3f}m")
                print(f"[信息] 计算的草高: {self.grass_height:.3f}m")
                return True
            else:
                print("[错误] 无法计算草高")
                return False

        except Exception as e:
            print(f"[错误] 阈值分析失败: {str(e)}")
            return False

    def visualize_results(self, save_path=None):
        """
        可视化分析结果

        参数:
            save_path: 结果图像保存路径，如果为None则不保存

        返回:
            可视化成功返回True，否则返回False
        """
        if self.point_cloud is None:
            print("[错误] 未加载点云数据")
            return False

        try:
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 绘制原始点云数据
            indices = np.arange(len(self.point_cloud))
            ax1.scatter(indices, self.point_cloud, c='blue',
                        s=30, alpha=0.6, label='点云数据')
            ax1.set_title('原始点云数据')
            ax1.set_xlabel('点索引')
            ax1.set_ylabel('距离 (m)')
            ax1.grid(True)

            # 添加直方图作为子图
            ax1_inset = ax1.inset_axes([0.6, 0.1, 0.35, 0.35])
            ax1_inset.hist(self.point_cloud, bins=15,
                           color='skyblue', alpha=0.7)
            ax1_inset.set_title('距离分布', fontsize=8)
            ax1_inset.tick_params(axis='both', which='major', labelsize=6)

            # 绘制聚类结果
            if self.ground_points is not None and self.grass_points is not None:
                # 绘制地面点
                if len(self.ground_points) > 0:
                    ground_indices = np.arange(len(self.ground_points))
                    ax2.scatter(ground_indices, self.ground_points,
                                c='green', s=30, alpha=0.6, label='地面点')

                # 如果有草点，绘制草点
                if len(self.grass_points) > 0:
                    # 为了在同一图上显示，我们使用不同的索引范围
                    grass_indices = np.arange(len(self.grass_points))
                    ax2.scatter(grass_indices, self.grass_points,
                                c='red', s=30, alpha=0.6, label='草点')

                # 绘制地面高度线
                if self.ground_height is not None:
                    ax2.axhline(y=self.ground_height, color='green',
                                linestyle='--', label=f'地面高度: {self.ground_height:.3f}m')

                # 绘制最大高度线
                if self.max_height is not None:
                    ax2.axhline(y=self.max_height, color='red',
                                linestyle='--', label=f'最大高度: {self.max_height:.3f}m')

                ax2.set_title('点云聚类结果')
                ax2.set_xlabel('点索引')
                ax2.set_ylabel('距离 (m)')
                ax2.grid(True)
                ax2.legend(loc='upper right')

                # 添加草高信息
                if self.grass_height is not None:
                    ax2.text(0.05, 0.05, f'草高: {self.grass_height:.3f}m',
                             transform=ax2.transAxes, fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.8))

                    # 在地面高度和最大高度之间添加双箭头标注草高
                    if self.ground_height is not None and self.max_height is not None:
                        # 在图的右侧添加箭头
                        x_pos = len(self.ground_points) * 0.9
                        ax2.annotate('', xy=(x_pos, self.max_height),
                                     xytext=(x_pos, self.ground_height),
                                     arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
                        # 添加草高文本
                        ax2.text(x_pos * 1.05, (self.ground_height + self.max_height) / 2,
                                 f'{self.grass_height:.3f}m',
                                 color='purple', fontweight='bold')

            # 调整布局
            plt.tight_layout()

            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[信息] 分析结果图像已保存到: {save_path}")

            # 关闭图形
            plt.close(fig)

            return True

        except Exception as e:
            print(f"[错误] 可视化结果失败: {str(e)}")
            return False

    def get_results(self):
        """
        获取分析结果

        返回:
            包含分析结果的字典
        """
        results = {
            "有效点数": self.valid_points_count,
            "总点数": self.total_points_count,
            "地面点数": len(self.ground_points) if self.ground_points is not None else 0,
            "草点数": len(self.grass_points) if self.grass_points is not None else 0,
            "地面高度(m)": round(self.ground_height, 3) if self.ground_height is not None else None,
            "最大高度(m)": round(self.max_height, 3) if self.max_height is not None else None,
            "草高(m)": round(self.grass_height, 3) if self.grass_height is not None else None
        }

        return results


def analyze_point_cloud(file_path, output_dir='results', eps=0.3, min_samples=2):
    """
    分析点云数据文件并生成结果

    参数:
        file_path: 点云数据文件路径
        output_dir: 结果保存目录
        eps: DBSCAN算法的邻域半径参数
        min_samples: DBSCAN算法的最小样本数参数

    返回:
        分析结果字典，如果分析失败则返回None
    """
    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建分析器
    analyzer = LidarHeightAnalyzer()

    # 加载点云数据
    if not analyzer.load_point_cloud(file_path):
        return None

    # 分析高度
    if not analyzer.analyze_height(eps=eps, min_samples=min_samples):
        return None

    # 生成结果文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    result_image_path = os.path.join(
        output_dir, f"{base_name}_height_analysis.png")

    # 可视化结果
    analyzer.visualize_results(save_path=result_image_path)

    # 返回分析结果
    return analyzer.get_results()


def main():
    """主函数，用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='激光雷达点云数据杂草高度分析工具')
    parser.add_argument('--file', type=str, required=True, help='点云数据文件路径')
    parser.add_argument('--output', type=str, default='results', help='结果保存目录')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='DBSCAN算法的邻域半径参数')
    parser.add_argument('--min-samples', type=int,
                        default=2, help='DBSCAN算法的最小样本数参数')

    args = parser.parse_args()

    # 分析点云数据
    results = analyze_point_cloud(
        args.file, args.output, args.eps, args.min_samples)

    if results:
        print("\n===== 分析结果 =====")
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

# 添加一个新的函数，用于处理数据库模式下的雷达数据分析
def analyze_lidar_from_db(db_manager, ld_lid, output_dir='results'):
    """
    从数据库中获取并分析特定lid的雷达数据
    
    参数:
        db_manager: 数据库管理器实例
        ld_lid: 要分析的雷达数据的lid
        output_dir: 结果保存目录
        
    返回:
        包含分析结果的字典，如果分析失败则返回None
    """
    if db_manager is None:
        print("[错误] 未提供数据库管理器")
        return None
        
    if ld_lid is None:
        print("[错误] 未提供有效的ld_lid")
        return None
        
    try:
        # 创建结果目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 直接使用ld_lid从数据库获取点云数据
        print(f"[信息] 使用ld_lid={ld_lid}获取雷达数据")
        X, Y, Z = db_manager.parse_ld_data(ld_lid)
        
        # 移除异常值
        Z_filtered, z_min, z_max = _remove_outliers(Z, percentile_low=10, percentile_high=90, z_threshold=3.0, method='combined')
        
        # 计算高度 (过滤后数据的最大值 - 平均值)
        # 注意：雷达数据中，值越小表示距离越远，所以高度计算需要取反
        height = float(np.nanmean(Z_filtered) - z_min)
        print(f"[信息] 雷达高度计算: 平均值={np.nanmean(Z_filtered):.2f}mm, 最小值={z_min:.2f}mm, 高度={height:.2f}mm")
        
        # 创建可视化图表
        heatmap_path = os.path.join(output_dir, f"heatmap_{ld_lid}.png")
        scatter_path = os.path.join(output_dir, f"3d_scatter_{ld_lid}.png")
        
        # 创建热力图和3D散点图
        _create_heatmap(Z, Z_filtered, ld_lid, output_dir)
        _create_3d_scatter_plot(X, Y, Z, Z_filtered, ld_lid, output_dir)
        
        print(f"[信息] 雷达数据分析完成 (lid={ld_lid}), 计算高度: {height:.2f}mm")
        
        # 返回结果
        return {
            "lidar_height_mm": height,
            "lidar_heatmap_path": heatmap_path,
            "lidar_scatter_path": scatter_path,
            "lidar_z_filtered_range": (z_min, z_max),
            "lidar_z_original_range": (Z.min(), Z.max())
        }
        
    except Exception as e:
        print(f"[错误] 分析雷达数据失败 (lid={ld_lid}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 辅助函数：移除异常值
def _remove_outliers(data, percentile_low=10, percentile_high=90, z_threshold=3.0, method='combined'):
    """
    移除异常值，返回处理后的数据和有效数值范围
    
    参数:
        data: 输入数据数组
        percentile_low: 下限百分位数，默认10%
        percentile_high: 上限百分位数，默认90%
        z_threshold: Z分数阈值，超过此值的点被视为异常值(默认3.0)
        method: 异常值检测方法，可选: 'percentile', 'z_score', 'iqr', 'combined'
        
    返回:
        Tuple[np.ndarray, float, float]: (处理后的数据，下限值，上限值)
    """
    # 创建数据副本以避免修改原始数据
    data_filtered = data.copy()
    data_flat = data.flatten()  # 用于统计计算
    
    if method == 'percentile' or method == 'combined':
        # 百分位数法 - 适合处理有偏分布
        p_low = np.percentile(data_flat, percentile_low)
        p_high = np.percentile(data_flat, percentile_high)
        
        if method == 'percentile':
            mask_percentile = (data < p_low) | (data > p_high)
            data_filtered[mask_percentile] = np.nan  # 标记为NaN后面再处理
            
    if method == 'z_score' or method == 'combined':
        # Z分数法 - 特别适合检测极端异常值
        data_mean = np.mean(data_flat)
        data_std = np.std(data_flat)
        z_scores = np.abs((data - data_mean) / (data_std + 1e-10))  # 加小值避免除零
        
        if method == 'z_score':
            mask_zscore = z_scores > z_threshold
            data_filtered[mask_zscore] = np.nan
            
    if method == 'iqr' or method == 'combined':
        # IQR法 - 鲁棒性好，不受极端值影响
        q1 = np.percentile(data_flat, 25)
        q3 = np.percentile(data_flat, 75)
        iqr = q3 - q1
        iqr_low = q1 - 1.5 * iqr
        iqr_high = q3 + 1.5 * iqr
        
        if method == 'iqr':
            mask_iqr = (data < iqr_low) | (data > iqr_high)
            data_filtered[mask_iqr] = np.nan
            
    if method == 'combined':
        # 综合检测法 - 结合上述三种方法的优点
        # 1. 基于百分位数的宽松限制
        # 2. Z分数检测极端异常值 
        # 3. IQR法提供鲁棒性检测
        mask_percentile = (data < p_low) | (data > p_high)
        mask_zscore = z_scores > z_threshold
        mask_iqr = (data < iqr_low) | (data > iqr_high)
        
        # 只要符合任一条件就认为是异常值
        mask_combined = mask_percentile | mask_zscore | mask_iqr
        data_filtered[mask_combined] = np.nan
    
    # 计算有效范围，用于设置热图色标范围
    valid_data = data_filtered[~np.isnan(data_filtered)]
    if len(valid_data) > 0:
        valid_min = np.min(valid_data)
        valid_max = np.max(valid_data)
    else:
        valid_min = np.min(data)
        valid_max = np.max(data)
        
    # 二次处理：将NaN替换为最近有效值或者边界值
    # 这样在可视化时不会出现空白区域
    if np.any(np.isnan(data_filtered)):
        nan_mask = np.isnan(data_filtered)
        data_filtered[nan_mask] = np.clip(data[nan_mask], valid_min, valid_max)
    
    # 返回过滤后的数据以及有效值范围
    return data_filtered, valid_min, valid_max

# 辅助函数：创建热力图
def _create_heatmap(Z, Z_filtered, lid, output_dir):
    """创建热力图可视化"""
    try:
        # 创建更美观的热力图，2x1布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
        
        # 获取过滤后数据的有效值范围
        valid_data = Z_filtered[~np.isnan(Z_filtered)]
        if len(valid_data) > 0:
            z_min = np.min(valid_data)
            z_max = np.max(valid_data)
        else:
            z_min = np.min(Z)
            z_max = np.max(Z)
        
        # === 左侧：处理异常值后的热力图 ===
        im1 = ax1.imshow(Z_filtered, cmap='jet', interpolation='bilinear', aspect='auto', 
                      vmin=z_min, vmax=z_max)
        ax1.set_title(f"处理异常值后的热力图 (ID: {lid})", fontsize=12, pad=15)
        ax1.set_xlabel('X方向采样点 (共160点)', fontsize=10)
        ax1.set_ylabel('Y方向采样点 (共60点)', fontsize=10)
        
        # 添加颜色条
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(f'深度值 (mm)\n[范围: {z_min:.2f} ~ {z_max:.2f}]', fontsize=10)
        
        # 添加网格线
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # === 右侧：原始热力图 ===
        im2 = ax2.imshow(Z, cmap='jet', interpolation='bilinear', aspect='auto')
        ax2.set_title(f"原始数据热力图 (ID: {lid})", fontsize=12, pad=15)
        ax2.set_xlabel('X方向采样点 (共160点)', fontsize=10)
        ax2.set_ylabel('Y方向采样点 (共60点)', fontsize=10)
        
        # 添加颜色条
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label(f'深度值 (mm)\n[范围: {Z.min():.2f} ~ {Z.max():.2f}]', fontsize=10)
        
        # 添加网格线
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 设置刻度
        for ax in [ax1, ax2]:
            x_ticks = np.linspace(0, Z.shape[1]-1, 5, dtype=int)
            y_ticks = np.linspace(0, Z.shape[0]-1, 4, dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
        
        # 添加总标题和说明
        fig.suptitle(f"激光雷达深度数据热力图 (数据ID: {lid})", fontsize=14)
        fig.text(0.5, 0.01, f"左图：移除极端异常值 (综合检测法) | 右图：原始数据", 
               ha='center', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"heatmap_{lid}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return plot_path
    except Exception as e:
        print(f"[错误] 创建热力图失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 辅助函数：创建3D散点图
def _create_3d_scatter_plot(X, Y, Z, Z_filtered, lid, output_dir):
    """创建3D散点图可视化"""
    try:
        # 创建更美观的3D散点图
        fig = plt.figure(figsize=(16, 12), dpi=100)  # 增加图形宽度
        
        # 创建两个子图：一个展示处理异常值后的数据，一个展示原始数据
        # 调整子图的相对宽度和位置
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)  # 增加水平间距
        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax2 = fig.add_subplot(gs[1], projection='3d')
        
        # 降采样以提高可视化效率
        step = 3  # 每隔3个点取一个样本
        xs = X[::step, ::step].flatten()
        ys = Y[::step, ::step].flatten()
        zs = Z[::step, ::step].flatten()
        zs_filtered = Z_filtered[::step, ::step].flatten()
        
        # 获取过滤后数据的有效值范围
        valid_data = Z_filtered[~np.isnan(Z_filtered)]
        if len(valid_data) > 0:
            z_min = np.min(valid_data)
            z_max = np.max(valid_data)
        else:
            z_min = np.min(Z)
            z_max = np.max(Z)
        
        # === 左侧图：处理异常值后的数据 ===
        norm_filtered = plt.Normalize(z_min, z_max)
        colors_filtered = cm.viridis(norm_filtered(zs_filtered))
        
        sc1 = ax1.scatter(xs, ys, zs_filtered, c=colors_filtered, marker='.', alpha=0.8)
        ax1.set_title(f"处理异常值后的点云图 (ID: {lid})", fontsize=12, pad=15)
        ax1.set_xlabel('X轴 (mm)', fontsize=10, labelpad=10)
        ax1.set_ylabel('Y轴 (mm)', fontsize=10, labelpad=10)
        ax1.set_zlabel('Z轴 - 深度 (mm)', fontsize=10, labelpad=10)
        
        # 添加颜色条并调整位置
        cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm_filtered, cmap=cm.viridis), 
                          ax=ax1, shrink=0.7, aspect=30, pad=0.1)
        cbar1.set_label(f'深度值 (mm)\n[范围: {z_min:.2f} ~ {z_max:.2f}]', fontsize=10, labelpad=10)
        
        # === 右侧图：原始数据 ===
        norm_original = plt.Normalize(Z.min(), Z.max())
        colors_original = cm.viridis(norm_original(zs))
        
        sc2 = ax2.scatter(xs, ys, zs, c=colors_original, marker='.', alpha=0.8)
        ax2.set_title(f"原始点云图 (ID: {lid})", fontsize=12, pad=15)
        ax2.set_xlabel('X轴 (mm)', fontsize=10, labelpad=10)
        ax2.set_ylabel('Y轴 (mm)', fontsize=10, labelpad=10)
        ax2.set_zlabel('Z轴 - 深度 (mm)', fontsize=10, labelpad=10)
        
        # 添加颜色条并调整位置
        cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm_original, cmap=cm.viridis), 
                           ax=ax2, shrink=0.7, aspect=30, pad=0.1)
        cbar2.set_label(f'深度值 (mm)\n[范围: {Z.min():.2f} ~ {Z.max():.2f}]', fontsize=10, labelpad=10)
        
        # 设置适当的视角
        ax1.view_init(elev=30, azim=45)
        ax2.view_init(elev=30, azim=45)
        
        # 添加异常值处理说明
        fig.suptitle(f"激光雷达点云数据可视化 (数据ID: {lid})", fontsize=14, y=0.95)
        fig.text(0.5, 0.02, f"左图：移除极端异常值 (综合检测法) | 右图：原始数据", 
                ha='center', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.04, 1, 0.93])
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"3d_scatter_{lid}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return plot_path
    except Exception as e:
        print(f"[错误] 创建3D散点图失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
