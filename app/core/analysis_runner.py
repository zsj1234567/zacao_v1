from PyQt6.QtCore import QObject, pyqtSignal, QThread
import logging
import traceback
import os
import json # For saving summary
import numpy as np
import matplotlib.pyplot as plt
import copy

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
    analysis_file_completed = pyqtSignal(bool, str, dict) # 成功/失败，文件路径，分析结果

    def __init__(self, config: dict, db_manager_ref=None):
        super().__init__()
        self.config = config
        self.db_manager_ref = db_manager_ref
        self.calibration_data = config.get('calibration_data', {}) # Get pre-loaded points
        self._is_running = False
        self.current_analyzer = None # Store the current analyzer instance
        self.current_analysis_runner = None
        self.current_analysis_thread = None

    def run(self):
        """运行分析的核心逻辑"""
        if not GrassAnalyzer or not DeepLearningGrassAnalyzer or not analyze_point_cloud:
            try:
                self.log_message.emit("错误：未能加载必要的分析模块。请检查脚本路径和依赖。")
            except RuntimeError:
                return
            try:
                self.analysis_complete.emit(False, "分析模块加载失败")
            except RuntimeError:
                return
            return

        self._is_running = True
        try:
            self.log_message.emit("开始分析...")
        except RuntimeError:
            return
        self.progress_updated.emit(0)
        total_steps = 100 # Total progress steps

        # 移除所有不可序列化对象，防止pickle错误
        for k in ['db_manager', 'image_to_lid', 'DataProcessor']:
            if k in self.config:
                self.config.pop(k)

        try:
            input_paths = self.config['input_paths']
            output_dir = self.config['output_dir']
            model_type = self.config.get('model_type', 'traditional')
            segment_method_str = self.config.get('segment_method', 'hsv')
            do_calibration = self.config.get('do_calibration', False)
            save_debug_images = self.config.get('save_debug_images', False)
            calculate_density = self.config.get('calculate_density', True)
            plot_layout = self.config.get('plot_layout', 'default')
            perform_lidar_analysis = self.config.get('perform_lidar_analysis', False)
            lidar_dir = self.config.get('lidar_dir', None)
            dbscan_eps = self.config.get('dbscan_eps', 0.3)
            dbscan_min_samples = self.config.get('dbscan_min_samples', 2)
            hsv_config_path = self.config.get('hsv_config_path', None)

            os.makedirs(output_dir, exist_ok=True)
            try:
                self.log_message.emit(f"结果将保存到: {os.path.abspath(output_dir)}")
            except RuntimeError:
                return

            num_images = len(input_paths)
            all_results = [] # Store results for each image
            all_image_summaries = [] # Store summaries for all images

            # --- Image Processing Loop ---
            image_processing_progress = 70
            lidar_progress_share = 20 if perform_lidar_analysis else 0
            final_summary_progress = 10

            for i, img_path in enumerate(input_paths):
                if not self._is_running:
                    try:
                        self.log_message.emit("分析被用户中止。")
                    except RuntimeError:
                        return
                    try:
                        self.analysis_complete.emit(False, "用户中止")
                    except RuntimeError:
                        return
                    return

                try:
                    self.log_message.emit(f"--- 开始处理图像: {os.path.basename(img_path)} ({i + 1}/{num_images}) ---")
                except RuntimeError:
                    return
                current_image_results = {"文件名": os.path.basename(img_path), "分析模型": model_type}
                image_start_progress = int(i / num_images * image_processing_progress)
                image_end_progress = int((i + 1) / num_images * image_processing_progress)
                update_img_progress = lambda percent: self._safe_emit_progress(image_start_progress + int(percent * (image_end_progress - image_start_progress)))

                try:
                    # --- 1. Initialize Analyzer ---
                    if model_type == 'dl':
                        self.current_analyzer = DeepLearningGrassAnalyzer()
                    else:
                        self.current_analyzer = GrassAnalyzer()
                        if hsv_config_path:
                             pass
                    update_img_progress(0.1)

                    # --- 2. Load Image ---
                    self.current_analyzer.load_image(img_path)
                    update_img_progress(0.2)

                    # --- 3. Calibrate Image ---
                    points_for_this_image = self.calibration_data.get(img_path)
                    if points_for_this_image:
                         self.current_analyzer.calibrate_image(points=points_for_this_image)
                    else:
                         self.current_analyzer.calibrate_image()
                    update_img_progress(0.3)

                    # --- 4. Segment Grass ---
                    self.log_message.emit(f"  - 正在分割草地...")
                    if model_type == 'dl':
                         self.current_analyzer.segment_grass()
                    else:
                         self.current_analyzer.segment_grass(method=segment_method_str.lower())
                    update_img_progress(0.5)

                    # --- 5. Calculate Coverage ---
                    self.log_message.emit(f"  - 正在计算盖度...")
                    coverage = self.current_analyzer.calculate_coverage()
                    current_image_results["草地盖度"] = f"{coverage:.2f}%"
                    self.log_message.emit(f"  计算盖度: {coverage:.2f}%")
                    update_img_progress(0.6)

                    # --- 6. Calculate Density (Optional) ---
                    density = None
                    if calculate_density:
                        self.log_message.emit(f"  - 正在计算密度...")
                        try:
                            if model_type != 'dl':
                                self.current_analyzer.segment_instances()
                            density = self.current_analyzer.calculate_density()
                            current_image_results["草地密度"] = f"{density} 株/平方米"
                            self.log_message.emit(f"  计算密度: {density} 株/平方米")
                        except Exception as density_err:
                             self.log_message.emit(f"  计算密度时出错: {density_err}")
                             logging.warning(f"计算密度错误 for {img_path}: {density_err}")
                             current_image_results["草地密度"] = "计算失败"
                    else:
                         current_image_results["草地密度"] = "未计算"
                    update_img_progress(0.8)

                    # --- 7. Visualize Results ---
                    self.log_message.emit(f"  - 正在生成分析图...")
                    base_input_dir = os.path.dirname(input_paths[0]) if len(input_paths) == 1 else self.config.get('base_input_dir', os.path.dirname(img_path))
                    rel_img_path = os.path.relpath(img_path, start=base_input_dir)
                    image_basename = os.path.splitext(rel_img_path.replace(os.path.sep, '_'))[0]
                    model_suffix = "_dl" if model_type == 'dl' else "_traditional"
                    layout_suffix = f"_{plot_layout}"
                    result_filename = f"{image_basename}{model_suffix}{layout_suffix}_analysis.png"
                    result_path = os.path.join(output_dir, result_filename)

                    saved_paths = self.current_analyzer.visualize_results(
                                                          save_path=result_path,
                                                          layout=plot_layout,
                                                          save_debug=save_debug_images,
                                                          calculate_density=calculate_density)

                    if saved_paths and isinstance(saved_paths, dict) and saved_paths.get('analysis_image'):
                        main_analysis_path = saved_paths['analysis_image']
                        self.log_message.emit(f"  分析图像已保存: {os.path.basename(main_analysis_path)}")
                        current_image_results["结果图路径"] = main_analysis_path
                        for key, path in saved_paths.items():
                            if key != 'analysis_image':
                                current_image_results[key] = path
                    else:
                        self.log_message.emit("警告: 未能生成或保存主分析图像。")
                        logging.warning(f"Visualize results failed to return expected paths for {img_path}")
                        current_image_results["结果图路径"] = None
                    update_img_progress(0.9)

                    # --- 数据库模式下雷达分析（每张图片分析后立即处理和显示） ---
                    db_manager = self.db_manager_ref
                    try:
                        if db_manager is not None:
                            dl_records = db_manager.get_ld_data()
                            if i < len(dl_records):
                                dl_row = dl_records[i]
                                lid = dl_row['lid']
                                from scripts.db_manager import DataProcessor
                                processor = DataProcessor(db_manager, output_dir)
                                stats_file = os.path.join(output_dir, f"ld_stats_{lid}.json")
                                if not os.path.exists(stats_file):
                                    processor.process_ld_data()
                                if os.path.exists(stats_file):
                                    with open(stats_file, 'r', encoding='utf-8') as f:
                                        stats = json.load(f)
                                    z_height = stats.get('filtered_data', {}).get('z_height', None)
                                    if z_height is not None:
                                        current_image_results["草地高度"] = f"{z_height:.2f}mm"
                                    else:
                                        current_image_results["草地高度"] = "雷达高度缺失"
                                else:
                                    current_image_results["草地高度"] = "雷达统计文件缺失"
                                lidar_3d_path = os.path.join(output_dir, f"3d_scatter_{lid}.png")
                                lidar_heatmap_path = os.path.join(output_dir, f"heatmap_{lid}.png")
                                current_image_results["lidar_3dscatter_image"] = lidar_3d_path if os.path.exists(lidar_3d_path) else None
                                current_image_results["lidar_heatmap_image"] = lidar_heatmap_path if os.path.exists(lidar_heatmap_path) else None
                                self.log_message.emit(f"  已保存雷达高度热图和3D点云图: {os.path.basename(lidar_heatmap_path)}, {os.path.basename(lidar_3d_path)}")
                                del processor
                            else:
                                current_image_results["草地高度"] = "雷达数据缺失"
                                self.log_message.emit(f"  雷达数据分析失败: dl表数据不足")
                    except Exception as e:
                        current_image_results["草地高度"] = "雷达数据缺失"
                        self.log_message.emit(f"  雷达数据分析失败: {e}")

                    # --- 立即发射单张图片分析完成信号 ---
                    results = {}
                    allowed_keys = set([
                        'traditional_default_analysis', 'analysis_image', 'original_debug_image', 'calibrated_debug_image',
                        'hsv_mask_debug_image', 'coverage_overlay_debug_image', 'instance_mask_debug_image',
                        'instance_overlay_debug_image', 'density_overlay_debug_image',
                        '草地盖度', '草地密度', '草地高度', 'lidar_3dscatter_image', 'lidar_heatmap_image'
                    ])
                    if hasattr(self.current_analyzer, 'results') and isinstance(self.current_analyzer.results, dict):
                        for k, v in self.current_analyzer.results.items():
                            if k in allowed_keys:
                                results[k] = v
                    for k, v in current_image_results.items():
                        if k in allowed_keys:
                            results[k] = v
                    self.analysis_file_completed.emit(True, img_path, results)

                    image_summary = current_image_results.copy()
                    image_summary['original_path'] = img_path
                    image_summary["状态"] = "成功"
                    all_image_summaries.append(image_summary)

                    self.log_message.emit(f"--- 图像处理完成: {os.path.basename(img_path)} --- ")

                except Exception as img_err:
                    error_msg = f"处理图像 {os.path.basename(img_path)} 时发生严重错误: {img_err}"
                    detailed_error = traceback.format_exc()
                    self.log_message.emit(error_msg)
                    logging.error(f"{error_msg}\n{detailed_error}")
                    error_summary = {
                        "文件名": os.path.basename(img_path),
                        "分析模型": model_type,
                        "状态": "失败",
                        "错误信息": str(img_err),
                        "详细错误": detailed_error
                    }
                    all_image_summaries.append(error_summary)
                finally:
                    self.progress_updated.emit(image_end_progress)
                    self.current_analyzer = None

            # self.log_message.emit("所有图像处理完成。") # Redundant
            # Final progress update after loop
            final_loop_progress = image_processing_progress + lidar_progress_share
            self.progress_updated.emit(final_loop_progress)

            # --- Final Summary Saving ---
            # self.log_message.emit("--- 所有图像处理完成，正在保存分析摘要 ---") # Redundant
            all_generated_paths = []
            
            # 收集所有生成的图像路径
            for img_summary in all_image_summaries:
                if img_summary.get("状态") == "成功":
                    # 添加主要结果图路径
                    for key, value in img_summary.items():
                        if isinstance(value, str) and (key.endswith('_path') or key.endswith('_image')):
                            if os.path.exists(value):
                                all_generated_paths.append(os.path.normpath(value))
                            else:
                                logging.warning(f"摘要中的路径不存在，将忽略: {value}")
            
            # 添加原始图像路径
            for img_path in input_paths:
                if os.path.exists(img_path):
                    all_generated_paths.append(os.path.normpath(img_path))
            
            # 添加HSV分析图像
            hsv_analysis_files = [f for f in os.listdir(output_dir) if f.startswith('hsv_analysis_') and f.endswith('.png')]
            for hsv_file in hsv_analysis_files:
                hsv_path = os.path.join(output_dir, hsv_file)
                if os.path.exists(hsv_path):
                    all_generated_paths.append(os.path.normpath(hsv_path))
            
            # 添加LiDAR分析图像
            lidar_analysis_files = [f for f in os.listdir(output_dir) if f.endswith('_height_analysis.png')]
            for lidar_file in lidar_analysis_files:
                lidar_path = os.path.join(output_dir, lidar_file)
                if os.path.exists(lidar_path):
                    all_generated_paths.append(os.path.normpath(lidar_path))
            
            # 添加所有调试图像
            debug_files = [f for f in os.listdir(output_dir) if f.endswith('_debug.png')]
            for debug_file in debug_files:
                debug_path = os.path.join(output_dir, debug_file)
                if os.path.exists(debug_path):
                    all_generated_paths.append(os.path.normpath(debug_path))
            
            all_generated_paths = sorted(list(set(all_generated_paths)))
            safe_config = copy.deepcopy(self.config)
            for k in list(safe_config.keys()):
                if k in ('db_manager', 'image_to_lid'):
                    safe_config.pop(k)
            json_summary = {
                "run_config": safe_config,
                "image_results": all_generated_paths
            }
            summary_file_path = os.path.join(output_dir, "analysis_summary.json")
            try:
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_summary, f, indent=4, ensure_ascii=False)
                self.log_message.emit(f"分析摘要已保存: {summary_file_path}")
            except Exception as e:
                self.log_message.emit(f"保存分析摘要 JSON 时出错: {e}")
                logging.error(f"Failed to save summary JSON: {traceback.format_exc()}")

            # --- 发射信号：使用原始的、包含详细信息的 all_image_summaries --- #
            try:
                message_to_emit = json.dumps(all_image_summaries, ensure_ascii=False)
                self.progress_updated.emit(100) # Final progress
                self.analysis_complete.emit(True, message_to_emit)
            except Exception as e:
                 error_msg = f"序列化详细分析结果以发送到 UI 时出错: {e}"
                 self.log_message.emit(error_msg)
                 logging.error(f"{error_msg}\n{traceback.format_exc()}")
                 self.analysis_complete.emit(True, "分析成功完成，但结果无法序列化显示。")

        except Exception as e:
            error_msg = f"分析过程中发生未预料的错误: {e}"
            detailed_error = traceback.format_exc()
            self.log_message.emit(error_msg)
            logging.error(f"{error_msg}\n{detailed_error}")
            self.analysis_complete.emit(False, f"分析失败: {e}")
        finally:
            self._is_running = False
            self.current_analyzer = None

    def stop(self):
        self._is_running = False
        if self.current_analyzer:
             if hasattr(self.current_analyzer, 'stop') and callable(self.current_analyzer.stop):
                 self.current_analyzer.stop()
        self.log_message.emit("停止信号已发送。等待当前步骤完成...")

    def on_file_analysis_complete(self, success, message):
        """
        文件分析完成的回调
        
        参数:
            success: 是否成功
            message: 完成消息
        """
        # 获取当前正在分析的文件路径
        file_path = ""
        if self.current_analysis_runner and hasattr(self.current_analysis_runner, 'config'):
            file_path = self.current_analysis_runner.config.get('input_paths', ['未知文件'])[0]
        
        logging.info(f"文件分析完成: {file_path}, 成功: {success}, 消息: {message}")
        
        # 获取分析结果
        results = {}
        if hasattr(self.current_analysis_runner, 'current_analyzer') and self.current_analysis_runner.current_analyzer:
            analyzer = self.current_analysis_runner.current_analyzer
            if hasattr(analyzer, 'results'):
                results = analyzer.results
                
                # 确保结果中包含盖度、密度和高度数据
                if success:
                    # 提取盖度数据
                    if hasattr(analyzer, 'calculate_coverage') and callable(analyzer.calculate_coverage):
                        try:
                            coverage = analyzer.calculate_coverage()
                            results['草地盖度'] = f"{coverage:.2f}%"
                        except Exception as e:
                            logging.error(f"计算盖度时出错: {e}")
                            results['草地盖度'] = "N/A"
                    
                    # 提取密度数据
                    if hasattr(analyzer, 'calculate_density') and callable(analyzer.calculate_density):
                        try:
                            density = analyzer.calculate_density()
                            results['草地密度'] = f"{density} 株/平方米"
                        except Exception as e:
                            logging.error(f"计算密度时出错: {e}")
                            results['草地密度'] = "N/A"
                    
                    # 提取高度数据（如果有）
                    if 'grass_height' in results:
                        height = results['grass_height']
                        if isinstance(height, (int, float)):
                            results['草地高度'] = f"{height:.3f}m"
                    elif '草高(m)' in results:
                        height = results['草高(m)']
                        if isinstance(height, (int, float)):
                            results['草地高度'] = f"{height:.3f}m"
                
        # 添加到已分析文件记录
        if success:
            self.files_tracker.add_analyzed_file(file_path, results)
            
        # 发出信号
        self.analysis_file_completed.emit(success, file_path, results)
        
        # 清理资源
        if self.current_analysis_thread:
            self.current_analysis_thread.quit()
            self.current_analysis_thread.wait()
            self.current_analysis_thread = None
            self.current_analysis_runner = None

    def _safe_emit_progress(self, value):
        try:
            self.progress_updated.emit(value)
        except RuntimeError:
            return

# Add a simple check function if needed
if __name__ == '__main__':
    # Example usage for testing (replace with actual config)
    print("Testing AnalysisRunner structure...")
    # config_test = { ... }
    # runner = AnalysisRunner(config_test)
    # runner.run() # Run directly for testing without thread
    print("AnalysisRunner defined.") 