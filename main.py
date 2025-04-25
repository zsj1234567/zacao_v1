import os
import json
import argparse
from scripts.calibration_tool import calibrate_image
from scripts.grass_analysis import GrassAnalyzer
# 导入深度学习分析器
from scripts.dl_grass_analyzer import DeepLearningGrassAnalyzer
# 导入点云高度分析器
from scripts.lidar_height_analyzer import LidarHeightAnalyzer, analyze_point_cloud
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime


# 简单的打印函数
def print_info(message):
    print(f"[信息] {message}")


def print_success(message):
    print(f"[成功] {message}")


def print_warning(message):
    print(f"[警告] {message}")


def print_error(message):
    print(f"[错误] {message}")


def print_section(title):
    print(f"\n===== {title} =====")


def print_dict_simple(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def spinner_simple(message, seconds):
    print(f"{message}...")
    time.sleep(seconds)
    print(f"{message}完成")


def print_file_info_simple(file_path):
    if not os.path.exists(file_path):
        print_error(f"文件不存在: {file_path}")
        return

    stats = os.stat(file_path)
    size = stats.st_size
    mtime = datetime.fromtimestamp(stats.st_mtime)

    if size < 1024:
        size_str = f"{size} B"
    elif size < 1048576:
        size_str = f"{size/1024:.2f} KB"
    else:
        size_str = f"{size/1048576:.2f} MB"

    print_info(f"文件: {os.path.basename(file_path)}")
    print_info(f"路径: {os.path.dirname(os.path.abspath(file_path))}")
    print_info(f"大小: {size_str}")
    print_info(f"修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


def find_all_images(directory):
    """
    递归查找目录及其子目录中的所有图像文件

    参数:
        directory: 要搜索的目录路径

    返回:
        图像文件路径列表
    """
    image_paths = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def main():
    # 打印欢迎信息
    print_section("草地盖度和密度分析工具")

    parser = argparse.ArgumentParser(description='草地盖度和密度分析工具')
    parser.add_argument('--image', type=str, help='要分析的图像路径或包含图像的文件夹路径')
    parser.add_argument('--calibrate', action='store_true', help='是否进行校准')
    parser.add_argument('--method', type=str, default='hsv',
                        choices=['hsv'],
                        help='分割方法: hsv (HSV颜色空间)')
    # 添加模型选择参数
    parser.add_argument('--model', type=str, default='traditional',
                        choices=['traditional', 'dl'], help='分析模型: traditional (传统方法) 或 dl (深度学习)')
    # 添加自定义结果保存目录参数
    parser.add_argument('--output-dir', type=str, default='results',
                        help='结果保存目录路径')
    # 添加分析图布局样式参数
    parser.add_argument('--layout', type=str, default='default',
                        choices=['default', 'simple'],
                        help='分析图布局样式: default (3x2布局) 或 simple (1x3简化布局)')
    # 添加保存哪些结果的参数
    parser.add_argument('--save-debug', action='store_true',
                        help='是否保存调试图像（包括HSV分析结果）')
    # 添加是否计算密度的参数
    parser.add_argument('--no-density', action='store_true',
                        help='不计算草的密度，仅计算盖度')
    # 添加是否分析点云高度的参数
    parser.add_argument('--analyze-height', action='store_true',
                        help='是否分析点云数据计算草高')
    # 添加点云数据目录参数
    parser.add_argument('--lidar-dir', type=str, default='datasets/ds3',
                        help='点云数据目录路径')
    # 添加DBSCAN算法参数
    parser.add_argument('--eps', type=float, default=0.3,
                        help='DBSCAN算法的邻域半径参数')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='DBSCAN算法的最小样本数参数')
    args = parser.parse_args()

    # 打印参数信息
    print_section("分析参数")
    param_dict = {
        "图像路径": args.image if args.image else "使用默认路径",
        "分析模型": args.model,
        "分割方法": args.method,
        "是否校准": "是" if args.calibrate else "否",
        "结果保存目录": args.output_dir,
        "布局样式": args.layout,
        "保存调试图像": "是" if args.save_debug else "否",
        "计算草密度": "否" if args.no_density else "是",
        "分析草高": "是" if args.analyze_height else "否"
    }

    if args.analyze_height:
        param_dict.update({
            "点云数据目录": args.lidar_dir,
            "DBSCAN邻域半径": args.eps,
            "DBSCAN最小样本数": args.min_samples
        })

    print_dict_simple(param_dict)

    # 默认数据集目录
    default_image_dir = 'datasets'

    # 确定要处理的图像
    image_paths = []

    if args.image:
        if os.path.exists(args.image):
            if os.path.isdir(args.image):
                # 如果是目录，递归处理目录中的所有图像
                print_info(f"递归处理目录: {args.image}")
                spinner_simple("正在搜索图像文件", 1)
                image_paths = find_all_images(args.image)

                if not image_paths:
                    print_error(f"目录 {args.image} 及其子目录中未找到图像文件")
                    return

                print_success(f"找到 {len(image_paths)} 个图像文件")
            else:
                # 如果是单个文件，只处理该文件
                print_info(f"处理单个图像: {args.image}")
                image_paths = [args.image]
        else:
            print_error(f"路径不存在: {args.image}")
            return
    else:
        # 如果未提供--image参数，使用默认数据集目录中的第一张图像
        if os.path.exists(default_image_dir):
            image_files = [f for f in os.listdir(default_image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if not image_files:
                print_error(f"默认目录 {default_image_dir} 中未找到图像文件")
                return

            # 默认处理第一张图像
            image_paths = [os.path.join(default_image_dir, image_files[0])]
            print_info(f"使用默认图像: {image_paths[0]}")
        else:
            print_error(f"默认目录 {default_image_dir} 不存在")
            return

    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    print_info(f"结果将保存到: {os.path.abspath(args.output_dir)}")

    # 处理每张图像
    for i, image_path in enumerate(image_paths):
        print_section(
            f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # 打印文件信息
        print_file_info_simple(image_path)

        # 检查是否需要校准
        calibration_dir = 'calibrations'
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_file = os.path.join(
            calibration_dir, f"{os.path.basename(image_path)}.json")

        if args.calibrate or not os.path.exists(calibration_file):
            print_info("进行图像校准...")
            spinner_simple("正在校准图像", 1)
            calibration_points = calibrate_image(image_path)
            if calibration_points is None:
                print_error("校准失败，跳过此图像")
                continue
            print_success("校准完成")

        # 根据选择的模型创建分析器
        if args.model == 'dl':
            print_info("使用深度学习方法进行分析...")
            analyzer = DeepLearningGrassAnalyzer()
        else:
            print_info("使用传统方法进行分析...")
            analyzer = GrassAnalyzer()

        # 加载图像（会自动加载校准文件）
        spinner_simple("正在加载图像", 1)
        analyzer.load_image(image_path)
        print_success("图像加载完成")

        # 校准图像
        spinner_simple("正在应用校准", 1)
        analyzer.calibrate_image()
        print_success("校准应用完成")

        # 分割草
        print_section("草地分割")
        spinner_simple("正在分割草地", 2)
        if args.model == 'dl':
            analyzer.segment_grass()
        else:
            analyzer.segment_grass(method=args.method)
        print_success("草地分割完成")

        # 计算盖度
        print_section("草地盖度计算")
        spinner_simple("正在计算草地盖度", 1)
        coverage = analyzer.calculate_coverage()
        print_success(f"草的盖度: {coverage:.2f}%")

        # 计算密度（如果需要）
        density = None
        if not args.no_density:
            print_section("草地密度计算")
            spinner_simple("正在计算草地密度", 2)
            density = analyzer.calculate_density()
            print_success(f"草的密度: {density} 株/平方米")
        else:
            print_info("跳过密度计算")

        # 分析点云高度（如果需要）
        height_results = None
        if args.analyze_height:
            print_section("草地高度分析")

            # 构建对应的点云数据文件路径
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            lidar_file = os.path.join(args.lidar_dir, f"{image_basename}.txt")

            if os.path.exists(lidar_file):
                print_info(f"找到对应的点云数据文件: {lidar_file}")
                spinner_simple("正在分析点云数据", 2)

                # 分析点云数据
                height_results = analyze_point_cloud(
                    lidar_file,
                    output_dir=args.output_dir,
                    eps=args.eps,
                    min_samples=args.min_samples
                )

                if height_results:
                    print_success(f"草的高度: {height_results['草高(m)']:.3f}m")
                else:
                    print_error("点云高度分析失败")
            else:
                print_error(f"未找到对应的点云数据文件: {lidar_file}")

        # 可视化结果
        print_section("结果可视化")
        model_suffix = "_dl" if args.model == 'dl' else "_traditional"
        layout_suffix = "_simple" if args.layout == 'simple' else ""

        # 生成唯一的结果文件名，避免同名文件覆盖
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        # 如果是从子文件夹中找到的图片，将相对路径信息编码到文件名中
        if args.image and os.path.isdir(args.image) and image_path.startswith(args.image):
            # 提取相对路径并替换路径分隔符为下划线
            rel_path = os.path.relpath(os.path.dirname(image_path), args.image)
            if rel_path != '.':  # 不是根目录
                # 将路径分隔符替换为下划线，避免文件名中包含路径分隔符
                rel_path = rel_path.replace(os.path.sep, '_')
                image_basename = f"{rel_path}_{image_basename}"

        result_path = os.path.join(
            args.output_dir, f"{image_basename}{model_suffix}{layout_suffix}_analysis.png")

        # 调用可视化函数，传入布局样式参数和密度计算标志
        spinner_simple("正在生成分析图像", 2)
        analyzer.visualize_results(save_path=result_path,
                                   layout=args.layout,
                                   save_debug=args.save_debug,
                                   calculate_density=not args.no_density)
        print_success(f"分析结果已保存到: {result_path}")

        # 汇总分析结果
        analysis_results = {
            "文件名": os.path.basename(image_path),
            "分析模型": args.model,
            "草地盖度": f"{coverage:.2f}%"
        }

        if density is not None:
            analysis_results["草地密度"] = f"{density} 株/平方米"

        # 添加高度分析结果
        if height_results and height_results.get("草高(m)") is not None:
            analysis_results["草地高度"] = f"{height_results['草高(m)']:.3f}m"

        # 打印分析结果摘要
        print_section("分析结果摘要")
        print_dict_simple(analysis_results)

    # 处理完成
    print_section("处理完成")
    print_success(f"已处理 {len(image_paths)} 个图像文件")
    print_success(f"结果已保存到目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
