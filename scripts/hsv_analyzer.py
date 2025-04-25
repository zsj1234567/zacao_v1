import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def analyze_hsv_thresholds(image_path):
    """
    分析图像的HSV阈值

    参数:
        image_path: 图像路径

    返回:
        tuple: (分析结果图像, HSV统计信息, 建议的HSV阈值)
    """
    try:
        print(f"\n正在处理图片: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像: {image_path}")
            return None, None, None

        print(f"图像尺寸: {image.shape}")

        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        print("HSV通道分离成功")

        # 计算直方图
        h_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
        v_hist = cv2.calcHist([v], [0], None, [256], [0, 256])

        print("直方图计算完成")

        # 创建图形
        plt.figure(figsize=(15, 10))

        # 显示原始图像
        plt.subplot(221)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')

        # 显示HSV直方图
        plt.subplot(222)
        plt.plot(h_hist, label='H')
        plt.plot(s_hist, label='S')
        plt.plot(v_hist, label='V')
        plt.title('HSV直方图')
        plt.legend()

        # 显示HSV通道
        plt.subplot(223)
        plt.imshow(h, cmap='hsv')
        plt.title('H通道')
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(s, cmap='gray')
        plt.title('S通道')
        plt.axis('off')

        # 将图形转换为图像
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        analysis_image = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)
        analysis_image = analysis_image.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # 计算HSV统计信息
        h_mean, h_std = np.mean(h), np.std(h)
        s_mean, s_std = np.mean(s), np.std(s)
        v_mean, v_std = np.mean(v), np.std(v)

        # 计算建议的HSV阈值
        suggested_thresholds = {
            'h': (max(30, int(h_mean - 2*np.std(h))), min(100, int(h_mean + 2*np.std(h)))),
            's': (max(20, int(s_mean - 1.5*np.std(s))), 255),
            'v': (max(20, int(v_mean - 1.5*np.std(v))), 255)
        }

        stats = {
            'h': {'mean': h_mean, 'std': h_std},
            's': {'mean': s_mean, 'std': s_std},
            'v': {'mean': v_mean, 'std': v_std}
        }

        return analysis_image, stats, suggested_thresholds

    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    try:
        # 分析ds3目录下的所有图片
        image_dir = "datasets/ds3"
        if not os.path.exists(image_dir):
            print(f"错误: 目录不存在: {image_dir}")
            sys.exit(1)

        print(f"开始分析目录: {image_dir}")
        image_files = [f for f in os.listdir(
            image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"错误: 在目录 {image_dir} 中没有找到图片文件")
            sys.exit(1)

        print(f"找到 {len(image_files)} 个图片文件")

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            analysis_image, stats, suggested_thresholds = analyze_hsv_thresholds(
                image_path)
            if analysis_image is not None:
                output_dir = 'results'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f'hsv_analysis_{os.path.basename(image_path)}.png')
                cv2.imwrite(output_path, analysis_image)
                print(f"分析结果已保存到: {output_path}")

                # 打印HSV统计信息
                print("\nHSV统计信息:")
                print(
                    f"H通道 - 均值: {stats['h']['mean']:.2f}, 标准差: {stats['h']['std']:.2f}")
                print(
                    f"S通道 - 均值: {stats['s']['mean']:.2f}, 标准差: {stats['s']['std']:.2f}")
                print(
                    f"V通道 - 均值: {stats['v']['mean']:.2f}, 标准差: {stats['v']['std']:.2f}")

                # 建议的HSV阈值
                print("\n建议的HSV阈值:")
                print(f"H: {suggested_thresholds['h']}")
                print(f"S: {suggested_thresholds['s']}")
                print(f"V: {suggested_thresholds['v']}")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
