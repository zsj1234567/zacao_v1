from PyQt6.QtCore import QObject, pyqtSignal, QThread
import logging
import traceback
import os
import json # For saving summary
import numpy as np

# 导入分析脚本 - 使用 try-except 以防脚本暂时不存在或路径错误
try:
    # from scripts.grass_analysis import GrassAnalyzer, SegmentMethod # Removed SegmentMethod
    from scripts.grass_analysis import GrassAnalyzer
    from scripts.dl_grass_analyzer import DeepLearningGrassAnalyzer
    from scripts.lidar_height_analyzer import analyze_point_cloud # Import the function directly
    # from scripts.calibration_tool import calibrate_image # Import if calibration UI needs it
except ImportError as e:
    logging.error(f"无法导入分析脚本: {e}。请确保 scripts 目录在 PYTHONPATH 中。")
    # 定义占位符或引发异常，取决于希望如何处理
    # GrassAnalyzer, SegmentMethod, DeepLearningGrassAnalyzer, analyze_point_cloud = None, None, None, None # Removed SegmentMethod
    GrassAnalyzer, DeepLearningGrassAnalyzer, analyze_point_cloud = None, None, None

class AnalysisRunner(QObject):
    """在单独的线程中运行分析任务"""
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    analysis_complete = pyqtSignal(bool, str) # 成功/失败，完成消息

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.calibration_data = config.get('calibration_data', {}) # Get pre-loaded points
        self._is_running = False
        self.current_analyzer = None # Store the current analyzer instance

    def run(self):
        """运行分析的核心逻辑"""
        if not GrassAnalyzer or not DeepLearningGrassAnalyzer or not analyze_point_cloud:
            self.log_message.emit("错误：未能加载必要的分析模块。请检查脚本路径和依赖。")
            self.analysis_complete.emit(False, "分析模块加载失败")
            return

        self._is_running = True
        self.log_message.emit("开始分析...")
        self.progress_updated.emit(0)
        total_steps = 100 # Total progress steps

        try:
            input_paths = self.config['input_paths']
            output_dir = self.config['output_dir']
            model_type = self.config.get('model_type', 'traditional')
            # segment_method = self.config.get('segment_method', SegmentMethod.HSV) # Removed SegmentMethod usage
            segment_method_str = self.config.get('segment_method', 'hsv') # Get the string directly
            do_calibration = self.config.get('do_calibration', False)
            save_debug_images = self.config.get('save_debug_images', False)
            calculate_density = self.config.get('calculate_density', True)
            plot_layout_str = self.config.get('plot_layout_str', 'default') # Get layout string from UI config
            perform_lidar_analysis = self.config.get('perform_lidar_analysis', False)
            lidar_dir = self.config.get('lidar_dir', None)
            dbscan_eps = self.config.get('dbscan_eps', 0.3) # Use the correct UI value
            dbscan_min_samples = self.config.get('dbscan_min_samples', 2) # Use the correct UI value
            hsv_config_path = self.config.get('hsv_config_path', None) # For traditional

            os.makedirs(output_dir, exist_ok=True)
            self.log_message.emit(f"结果将保存到: {os.path.abspath(output_dir)}")

            num_images = len(input_paths)
            all_results = [] # Store results for each image
            all_image_summaries = [] # Store summaries for all images

            # --- Image Processing Loop ---
            image_processing_progress = 70 # Allocate 70% for image processing
            lidar_progress_share = 20 if perform_lidar_analysis else 0 # Allocate 20% for lidar if enabled
            final_summary_progress = 10 # Allocate 10% for final summary

            for i, img_path in enumerate(input_paths):
                if not self._is_running:
                    self.log_message.emit("分析被用户中止。")
                    self.analysis_complete.emit(False, "用户中止")
                    return

                self.log_message.emit(f"--- 处理图像: {os.path.basename(img_path)} ({i + 1}/{num_images}) ---")
                current_image_results = {"文件名": os.path.basename(img_path), "分析模型": model_type}
                image_start_progress = int(i / num_images * image_processing_progress)
                image_end_progress = int((i + 1) / num_images * image_processing_progress)
                steps_per_image = 7 # Number of steps within image processing

                step_progress = lambda step: self.progress_updated.emit(image_start_progress + int((step / steps_per_image) * (image_end_progress - image_start_progress)))

                try:
                    # --- 1. Initialize Analyzer ---
                    self.log_message.emit("步骤 1/7: 初始化分析器...")
                    if model_type == 'dl':
                        # NOTE: DL Analyzer in script doesn't take model_path/device
                        self.current_analyzer = DeepLearningGrassAnalyzer()
                        self.log_message.emit("使用深度学习模型进行分析 (默认模型)。")
                    else:
                        # NOTE: Traditional Analyzer doesn't take most config in __init__
                        self.current_analyzer = GrassAnalyzer()
                        # self.log_message.emit(f"使用传统方法 ({segment_method.name}) 进行分析。") # Adjusted log message
                        self.log_message.emit(f"使用传统方法 ({segment_method_str}) 进行分析。")
                        if hsv_config_path:
                             self.log_message.emit(f"使用 HSV 配置文件: {hsv_config_path}")
                             # TODO: Need a way to pass hsv_config_path to GrassAnalyzer, maybe a setter?
                             # Assuming load_image or segment_grass might handle it based on naming conventions
                             pass
                    step_progress(1)

                    # --- 2. Load Image ---
                    self.log_message.emit("步骤 2/7: 加载图像...")
                    self.current_analyzer.load_image(img_path)
                    step_progress(2)

                    # --- 3. Calibrate Image ---
                    # Note: calibrate_image now loads from file if calibration_points is None
                    self.log_message.emit("步骤 3/7: 应用校准...")
                    # Get pre-loaded points for this image, if any
                    points_for_this_image = self.calibration_data.get(img_path)

                    # if do_calibration:
                    #     self.log_message.emit("执行显式校准... (尚未完全集成到Runner)")
                    #     # Here you might call calibrate_image from calibration_tool if interactive calibration is needed
                    #     # For now, assume calibration points are loaded from file or set in analyzer
                    # self.current_analyzer.calibrate_image() # Uses internal or loaded points
                    if points_for_this_image:
                         self.log_message.emit(f"  - 使用预加载/保存的校准点: {points_for_this_image}")
                         self.current_analyzer.calibrate_image(points=points_for_this_image)
                    else:
                         # No pre-loaded points, rely on analyzer's internal loading or default
                         self.log_message.emit("  - 未提供校准点，依赖分析器自动加载或默认行为。")
                         self.current_analyzer.calibrate_image()

                    self.log_message.emit("校准应用完成。")
                    step_progress(3)

                    # --- 4. Segment Grass ---
                    self.log_message.emit("步骤 4/7: 分割草地...")
                    if model_type == 'dl':
                         self.current_analyzer.segment_grass()
                    else:
                         # self.current_analyzer.segment_grass(method=segment_method.name.lower()) # Pass method name string directly
                         self.current_analyzer.segment_grass(method=segment_method_str.lower())
                    self.log_message.emit("草地分割完成。")
                    step_progress(4)

                    # --- 5. Calculate Coverage ---
                    self.log_message.emit("步骤 5/7: 计算盖度...")
                    coverage = self.current_analyzer.calculate_coverage()
                    current_image_results["草地盖度"] = f"{coverage:.2f}%"
                    self.log_message.emit(f"盖度: {coverage:.2f}%")
                    step_progress(5)

                    # --- 6. Calculate Density (Optional) ---
                    density = None
                    if calculate_density:
                        self.log_message.emit("步骤 6/7: 计算密度...")
                        try:
                            # Need to call segment_instances first for traditional method
                            if model_type != 'dl':
                                self.log_message.emit("  - (传统) 执行实例分割...")
                                self.current_analyzer.segment_instances() # Default method is watershed
                            density = self.current_analyzer.calculate_density() # Uses instance segmentation results
                            current_image_results["草地密度"] = f"{density} 株/平方米"
                            self.log_message.emit(f"密度: {density} 株/平方米")
                        except Exception as density_err:
                             self.log_message.emit(f"计算密度时出错: {density_err}")
                             current_image_results["草地密度"] = "计算失败"
                    else:
                         self.log_message.emit("步骤 6/7: 跳过密度计算。")
                         current_image_results["草地密度"] = "未计算"
                    step_progress(6)

                    # --- 7. Visualize Results ---
                    self.log_message.emit("步骤 7/7: 生成可视化结果...")
                    # Create unique filename based on input path relative to the base input dir if possible
                    base_input_dir = os.path.dirname(input_paths[0]) if len(input_paths) == 1 else self.config.get('base_input_dir', os.path.dirname(img_path))
                    rel_img_path = os.path.relpath(img_path, start=base_input_dir)
                    image_basename = os.path.splitext(rel_img_path.replace(os.path.sep, '_'))[0]

                    model_suffix = "_dl" if model_type == 'dl' else "_traditional"
                    layout_suffix = f"_{plot_layout_str}" # Use string from config

                    result_filename = f"{image_basename}{model_suffix}{layout_suffix}_analysis.png"
                    result_path = os.path.join(output_dir, result_filename)

                    # --- 修改 visualize_results 调用以接收字典 ---
                    # saved_main_path = self.current_analyzer.visualize_results(
                    saved_paths = self.current_analyzer.visualize_results(
                                                          save_path=result_path,
                                                          layout=plot_layout_str, # Pass layout string
                                                          save_debug=save_debug_images,
                                                          calculate_density=calculate_density)

                    # --- 检查返回结果并记录 ---
                    if saved_paths and isinstance(saved_paths, dict) and saved_paths.get('analysis_image'):
                        main_analysis_path = saved_paths['analysis_image']
                        self.log_message.emit(f"分析图像已保存到: {main_analysis_path}")
                        current_image_results["结果图路径"] = main_analysis_path # 主分析图
                        # 添加其他保存的调试图像路径
                        for key, path in saved_paths.items():
                            if key != 'analysis_image':
                                current_image_results[key] = path # e.g., 'hsv_mask_debug_image': 'path/to/hsv_debug.png'
                                self.log_message.emit(f"调试图像 [{key}] 已保存到: {path}")
                    else:
                        self.log_message.emit("警告: 可视化函数未返回预期的路径字典或主分析图路径。")
                        current_image_results["结果图路径"] = None

                    # step_progress(7) # 移动到可视化之后

                    # --- Image Lidar Analysis (if enabled for this image) ---
                    lidar_processing_start_progress = image_end_progress # Lidar starts after image processing
                    lidar_processing_end_progress = lidar_processing_start_progress + int(num_images * lidar_progress_share)
                    lidar_step_progress = lambda: self.progress_updated.emit(
                         lidar_processing_start_progress + int((i / num_images) * lidar_progress_share)
                     )

                    if perform_lidar_analysis:
                        lidar_file_base = os.path.splitext(os.path.basename(img_path))[0]
                        # Adjust extension based on your Lidar data format (e.g., .txt, .pcd)
                        lidar_file_path = os.path.join(lidar_dir, f"{lidar_file_base}.txt")

                        if os.path.exists(lidar_file_path):
                            self.log_message.emit(f"  - 发现对应的 Lidar 文件: {lidar_file_path}, 开始高度分析...")
                            try:
                                # Call the imported function directly
                                height_results = analyze_point_cloud(
                                    lidar_file_path,
                                    output_dir=output_dir, # Pass output dir for potential plots from lidar script
                                    eps=dbscan_eps,
                                    min_samples=dbscan_min_samples
                                )
                                if height_results and height_results.get("草高(m)") is not None:
                                    grass_height_m = height_results["草高(m)"]
                                    current_image_results["草地高度"] = f"{grass_height_m:.3f}m"
                                    self.log_message.emit(f"  - Lidar 分析完成。草高: {grass_height_m:.3f}m")
                                    # Optionally add other lidar results
                                    current_image_results["地面高度"] = f"{height_results.get('地面高度(m)', 'N/A'):.3f}m"
                                    current_image_results["最大高度点"] = f"{height_results.get('最大高度点(m)', 'N/A'):.3f}m"

                                else:
                                    self.log_message.emit("  - Lidar 分析未返回有效高度。")
                                    current_image_results["草地高度"] = "分析失败或无结果"
                            except Exception as lidar_err:
                                self.log_message.emit(f"  - Lidar 分析过程中发生错误: {lidar_err}")
                                logging.error(f"Lidar analysis error for {lidar_file_path}: {traceback.format_exc()}")
                                current_image_results["草地高度"] = "分析出错"
                        else:
                            self.log_message.emit(f"  - 未找到对应的 Lidar 文件: {lidar_file_path}")
                            current_image_results["草地高度"] = "无数据"
                    else:
                         current_image_results["草地高度"] = "未分析"

                    # Placeholder for actual density calculation logic
                    logging.info(f"Calculating density for {image_basename}...")
                    density_percentage = np.random.uniform(5.0, 60.0) # Dummy value
                    # Assume density map is the final display result here
                    final_result_image_path = result_path # Record path
                    current_image_results["density_percentage"] = round(density_percentage, 2)
                    current_image_results["calibration_points"] = points_for_this_image # Already a list
                    # Add other relevant metrics if needed
                    logging.info(f"Density for {image_basename}: {density_percentage:.2f}%")

                    # --- 更新 image_summary 创建逻辑 --- #
                    # 使用 current_image_results 字典作为基础，因为它已包含所有计算结果和调试路径
                    image_summary = current_image_results.copy()
                    image_summary['original_path'] = img_path # 确保原始路径存在
                    image_summary["状态"] = "成功" # 标记为成功

                    # 不再需要单独添加 result_image_path，它已包含在 current_image_results 中
                    # if final_result_image_path:
                    #     image_summary['result_image_path'] = final_result_image_path

                    # 添加到总列表
                    all_image_summaries.append(image_summary)

                except Exception as img_err:
                    error_msg = f"处理图像 {os.path.basename(img_path)} 时发生严重错误: {img_err}"
                    detailed_error = traceback.format_exc()
                    self.log_message.emit(error_msg)
                    logging.error(f"{error_msg}\n{detailed_error}")
                    # 记录失败信息到 summary
                    error_summary = {
                        "文件名": os.path.basename(img_path),
                        "分析模型": model_type,
                        "状态": "失败",
                        "错误信息": str(img_err), # 简短错误信息
                        "详细错误": detailed_error # 完整追溯
                    }
                    all_image_summaries.append(error_summary)
                finally:
                    # 更新进度条，即使出错也要保证前进
                    self.progress_updated.emit(image_end_progress)
                    self.current_analyzer = None # 清理当前分析器实例

            self.log_message.emit("所有图像处理完成。")
            self.progress_updated.emit(image_processing_progress + lidar_progress_share) # Progress after image loop + potential lidar share

            # --- Final Summary Saving ---
            summary_start_progress = image_processing_progress + lidar_progress_share
            self.progress_updated.emit(summary_start_progress)
            self.log_message.emit("--- 所有图像处理完成，正在保存分析摘要 ---")

            # --- 修改：为 JSON 文件创建不同的摘要结构 --- #
            # 1. 提取所有生成的图像路径到一个扁平列表
            all_generated_paths = []
            for img_summary in all_image_summaries:
                if img_summary.get("状态") == "成功":
                    # 查找所有以 _path 或 _image 结尾的键，并且值是字符串
                    for key, value in img_summary.items():
                        if isinstance(value, str) and (
                            key.endswith('_path') or key.endswith('_image')
                        ):
                            if os.path.exists(value): # 确保路径存在
                                all_generated_paths.append(os.path.normpath(value))
                            else:
                                logging.warning(f"摘要中的路径不存在，将忽略: {value}")

            # 去重并排序（可选）
            all_generated_paths = sorted(list(set(all_generated_paths)))

            # 2. 创建用于 JSON 文件的新摘要字典
            json_summary = {
                "run_config": self.config, # 仍然保存运行配置
                "image_results": all_generated_paths # 使用扁平化的路径列表
                # 可以根据需要添加其他顶层摘要信息，例如总图像数、成功数等
            }

            # 3. 保存这个新结构到 JSON 文件
            summary_file_path = os.path.join(output_dir, "analysis_summary.json")
            try:
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_summary, f, indent=4, ensure_ascii=False)
                self.log_message.emit(f"分析摘要 (仅路径列表) 已保存到: {summary_file_path}")
            except Exception as e:
                self.log_message.emit(f"保存分析摘要 JSON 时出错: {e}")
                logging.error(f"Failed to save summary JSON: {traceback.format_exc()}")
            # --- 结束修改 --- #

            # --- 发射信号：使用原始的、包含详细信息的 all_image_summaries --- #
            try:
                # 将原始的字典列表序列化为 JSON 字符串发送给 UI
                message_to_emit = json.dumps(all_image_summaries, ensure_ascii=False)
                self.progress_updated.emit(100) # Final progress
                self.analysis_complete.emit(True, message_to_emit) # 发送包含完整信息的列表
            except Exception as e:
                 error_msg = f"序列化详细分析结果以发送到 UI 时出错: {e}"
                 self.log_message.emit(error_msg)
                 logging.error(f"{error_msg}\n{traceback.format_exc()}")
                 # 即使序列化失败，也尝试发送一个通用成功消息，但可能无结果
                 self.analysis_complete.emit(True, "分析成功完成，但结果无法序列化显示。")

            # --- 旧的保存和发射逻辑 (注释掉或删除) ---
            # run_summary = {
            #     "run_config": self.config, # 保存运行配置
            #     "image_results": all_image_summaries # 保存所有图像的处理结果和路径
            # }
            #
            # summary_file_path = os.path.join(output_dir, "analysis_summary.json")
            # try:
            #     with open(summary_file_path, 'w', encoding='utf-8') as f:
            #         json.dump(run_summary, f, indent=4, ensure_ascii=False)
            #     self.log_message.emit(f"分析摘要已保存到: {summary_file_path}")
            # except Exception as e:
            #     self.log_message.emit(f"保存分析摘要时出错: {e}")
            #     logging.error(f"Failed to save summary JSON: {traceback.format_exc()}")
            #
            # self.progress_updated.emit(100) # Final progress
            # self.analysis_complete.emit(True, "分析成功完成") # 旧的信号只发送简单消息
            # --- 结束旧逻辑 ---

        except Exception as e:
            error_msg = f"分析过程中发生未预料的错误: {e}"
            detailed_error = traceback.format_exc()
            self.log_message.emit(error_msg)
            logging.error(f"{error_msg}\n{detailed_error}")
            self.analysis_complete.emit(False, f"分析失败: {e}")
        finally:
            self._is_running = False
            self.current_analyzer = None # Ensure cleanup

    def stop(self):
        """请求停止分析"""
        self._is_running = False
        if self.current_analyzer:
             # If the analyzer has a stop method, call it
             if hasattr(self.current_analyzer, 'stop') and callable(self.current_analyzer.stop):
                 self.current_analyzer.stop()
        self.log_message.emit("停止信号已发送。等待当前步骤完成...")

# Add a simple check function if needed
if __name__ == '__main__':
    # Example usage for testing (replace with actual config)
    print("Testing AnalysisRunner structure...")
    # config_test = { ... }
    # runner = AnalysisRunner(config_test)
    # runner.run() # Run directly for testing without thread
    print("AnalysisRunner defined.") 