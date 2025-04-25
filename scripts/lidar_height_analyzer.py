import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib
import json

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
