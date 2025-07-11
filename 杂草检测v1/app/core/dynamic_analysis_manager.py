import os
import time
import logging
import json
from copy import deepcopy
from datetime import datetime
from functools import partial
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from app.core.analyzed_files_tracker import AnalyzedFilesTracker
from app.core.analysis_runner import AnalysisRunner

class DynamicAnalysisManager(QObject):
    """
    动态分析管理器
    
    负责监控输入目录中的新文件并自动进行分析
    """
    
    # 信号定义
    new_files_found = pyqtSignal(list)  # 发现新文件时发出的信号
    analysis_started = pyqtSignal(str)  # 开始分析文件时发出的信号
    analysis_progress = pyqtSignal(int)  # 分析进度信号
    analysis_log = pyqtSignal(str)  # 分析日志信号
    analysis_file_completed = pyqtSignal(bool, str, dict)  # 单个文件分析完成信号 (成功与否, 文件路径, 结果)
    analysis_all_completed = pyqtSignal()  # 所有文件分析完成信号
    analysis_complete_signal = pyqtSignal(list)  # 分析完成信号
    
    def __init__(self, config=None):
        """
        初始化动态分析管理器
        
        参数:
            config: 分析配置字典
        """
        super().__init__()
        self.config = config or {}
        self.files_tracker = AnalyzedFilesTracker()
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_for_new_files)
        self.monitor_interval = 5000  # 默认5秒检查一次
        self.is_analyzing = False
        self.current_analysis_thread = None
        self.current_analysis_runner = None
        self.pending_files = []
        self.db_manager = None
        self.monitor_mode = 'local'  # 'local' or 'database'
        # 添加临时存储分析结果的字典
        self.temp_analysis_results = {}
        
    def set_config(self, config):
        """
        设置分析配置
        
        参数:
            config: 分析配置字典
        """
        self.config = config
        
    def set_monitor_interval(self, interval_ms):
        """
        设置监控间隔时间
        
        参数:
            interval_ms: 间隔时间（毫秒）
        """
        self.monitor_interval = interval_ms
        if self.is_monitoring():
            self.restart_monitoring()
            
    def start_monitoring(self):
        """开始监控新文件"""
        if not self.monitor_timer.isActive():
            self.monitor_timer.start(self.monitor_interval)
            logging.info(f"开始监控新文件，间隔: {self.monitor_interval}ms")
            
    def stop_monitoring(self):
        """停止监控新文件"""
        if self.monitor_timer.isActive():
            self.monitor_timer.stop()
            logging.info("停止监控新文件")
            
    def is_monitoring(self):
        """
        检查是否正在监控
        
        返回:
            bool: 是否正在监控
        """
        return self.monitor_timer.isActive()
        
    def restart_monitoring(self):
        """重启监控"""
        self.stop_monitoring()
        self.start_monitoring()
        
    def set_db_manager(self, db_manager):
        self.db_manager = db_manager

    def set_monitor_mode(self, mode):
        self.monitor_mode = mode
#todo 检测新文件
    def check_for_new_files(self):
        if self.is_analyzing:
            return
        if self.monitor_mode == 'database' and self.db_manager:
            jpeg_records = self.db_manager.get_jpeg_data()
            analyzed_jpeg_lids = self.db_manager.get_analyzed_jpeg_lids()
            new_records = [rec for rec in jpeg_records if rec['lid'] not in analyzed_jpeg_lids]
            new_files = []
            temp_dir = os.path.join(os.getcwd(), 'db_temp_images')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 获取所有ld记录的lid，并存入一个集合中以便快速查找
            ld_lids = {rec['lid'] for rec in self.db_manager.get_ld_data()}

            for rec in new_records:
                jpeg_lid = rec['lid']
                # 检查是否存在匹配的ld_lid
                if jpeg_lid in ld_lids:
                    # 使用相同的lid进行配对
                    ld_lid = jpeg_lid
                    img_path = self.db_manager.save_jpeg_to_file(jpeg_lid, output_dir=temp_dir)
                    new_files.append((img_path, jpeg_lid, ld_lid))
                else:
                    logging.warning(f"未找到与jpeg_lid {jpeg_lid} 匹配的ld数据")

            if new_files:
                self.new_files_found.emit([f[0] for f in new_files])
                self.pending_files.extend(new_files)
                logging.info(f"[DB动态分析] 发现 {len(new_files)} 个新数据库lid，添加到待分析队列")
                if not self.is_analyzing:
                    self.process_next_file()
            return
        # 本地模式原逻辑
        input_dir = self.config.get('input_dir')
        if not input_dir or not os.path.exists(input_dir):
            logging.warning(f"输入目录不存在或未设置: {input_dir}")
            return
        new_files = self.files_tracker.find_new_files(input_dir)
        if new_files:
            self.new_files_found.emit(new_files)
            self.pending_files.extend(new_files)
            logging.info(f"发现 {len(new_files)} 个新文件，添加到待分析队列")
            if not self.is_analyzing:
                self.process_next_file()

    def process_next_file(self):
        """处理队列中的下一个文件"""
        if not self.pending_files:
            logging.info("所有文件已分析完成")
            self.analysis_all_completed.emit()
            return
            
        # 获取下一个文件
        next_item = self.pending_files.pop(0)
        if self.monitor_mode == 'database':
            img_path, jpeg_lid, ld_lid = next_item
            self.analyze_file(img_path, jpeg_lid, ld_lid)
        else:
            self.analyze_file(next_item)
        
    def analyze_file(self, file_path, jpeg_lid=None, ld_lid=None):
        """
        分析单个文件
        
        参数:
            file_path: 文件路径
            jpeg_lid: JPEG的LID
            ld_lid: LD的LID
        """
        if self.is_analyzing:
            logging.warning(f"当前有分析任务正在进行，文件 {file_path} 将稍后处理")
            if isinstance(file_path, tuple) and len(file_path) == 3:
                # 如果是元组形式的待处理项，完整保存
                if file_path not in self.pending_files:
                    self.pending_files.append(file_path)
            else:
                # 普通字符串路径
                if file_path not in self.pending_files:
                    self.pending_files.append(file_path)
            return
            
        self.is_analyzing = True
        self.analysis_started.emit(file_path)
        logging.info(f"开始分析文件: {file_path}")
        
        # 创建分析配置 - 使用深拷贝确保不影响原始配置
        analysis_config = deepcopy(self.config)
        analysis_config['input_paths'] = [file_path]
        
        # 传递lid信息到config - 确保这些信息被正确传递
        if jpeg_lid is not None and ld_lid is not None:
            # 保存到配置中
            analysis_config['jpeg_lid'] = jpeg_lid
            analysis_config['ld_lid'] = ld_lid
            # 同时保存到类实例变量中，以便于回调函数访问
            self.config['jpeg_lid'] = jpeg_lid
            self.config['ld_lid'] = ld_lid
            logging.info(f"[DynamicAnalysis] 配置LID信息: jpeg_lid={jpeg_lid}, ld_lid={ld_lid}")
        
        # 处理校准模式
        if analysis_config.get('do_calibration', False):
            calibration_mode = analysis_config.get('calibration_mode', 'manual')
            logging.info(f"使用校准模式: {calibration_mode}")
            
            # 检查校准文件是否存在
            calib_file = self._get_calibration_file_path(file_path)
            if os.path.exists(calib_file):
                try:
                    with open(calib_file, 'r') as f:
                        points = json.load(f)
                        if isinstance(points, list) and len(points) == 4:
                            # 添加校准数据到配置
                            if 'calibration_data' not in analysis_config:
                                analysis_config['calibration_data'] = {}
                            analysis_config['calibration_data'][file_path] = points
                            logging.info(f"已加载校准点: {points}")
                        else:
                            logging.warning(f"校准文件 {os.path.basename(calib_file)} 格式无效")
                except Exception as e:
                    logging.error(f"加载校准文件出错: {e}")
            else:
                logging.info(f"未找到校准文件: {calib_file}")
                # 如果是自动校准模式，可以在这里添加自动校准的代码
                if calibration_mode == 'auto':
                    logging.info("使用自动校准模式，自动生成校准点")
                    # 创建校准目录
                    calib_dir = os.path.dirname(calib_file)
                    os.makedirs(calib_dir, exist_ok=True)
                    
                    # 自动生成校准点（使用图像四角）
                    try:
                        # 读取图像获取尺寸
                        import cv2
                        import numpy as np
                        img = cv2.imread(file_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            # 生成四个角点
                            auto_points = [
                                [0, 0],                  # 左上
                                [width - 1, 0],          # 右上
                                [width - 1, height - 1], # 右下
                                [0, height - 1]          # 左下
                            ]
                            # 保存校准点
                            with open(calib_file, 'w') as f:
                                json.dump(auto_points, f)
                            
                            # 添加到配置
                            if 'calibration_data' not in analysis_config:
                                analysis_config['calibration_data'] = {}
                            analysis_config['calibration_data'][file_path] = auto_points
                            logging.info(f"已自动生成校准点: {auto_points}")
                        else:
                            logging.warning(f"无法读取图像进行自动校准: {file_path}")
                    except Exception as e:
                        logging.error(f"自动校准失败: {e}")
                else:
                    # 手动模式但没有校准文件，发出警告
                    logging.warning("手动校准模式下未找到校准文件，将使用原始尺寸")
        
        # 创建分析线程和运行器
        self.current_analysis_thread = QThread()
        self.current_analysis_runner = AnalysisRunner(analysis_config, db_manager_ref=self.db_manager)
        self.current_analysis_runner.moveToThread(self.current_analysis_thread)
        
        # 连接信号
        self.current_analysis_thread.started.connect(self.current_analysis_runner.run)
        self.current_analysis_runner.progress_updated.connect(self.analysis_progress)
        self.current_analysis_runner.log_message.connect(self.analysis_log)
        self.current_analysis_runner.analysis_complete.connect(self.on_file_analysis_complete)
        self.current_analysis_runner.analysis_file_completed.connect(self.on_file_analysis_file_completed)
        
        # 启动分析线程
        self.current_analysis_thread.start()
# todo 动态分析写入数据库
    def on_file_analysis_complete(self, result):
        """单张文件分析完成后的回调"""
        file_path = ""
        if self.current_analysis_runner and hasattr(self.current_analysis_runner, 'config'):
            file_path = self.current_analysis_runner.config.get('input_paths', ['未知文件'])[0]
        logging.info(f"[DynamicAnalysis] 文件分析完成: {file_path}, 结果: {result}")
        
        # 构造完整的image_result字典，包含所有分析字段和图像路径
        image_result = result.copy() if isinstance(result, dict) else {}
        image_result['original_path'] = file_path
        
        # 如果有current_analysis_runner，尝试从其中提取当前的结果
        if self.current_analysis_runner and hasattr(self.current_analysis_runner, 'current_analyzer'):
            analyzer = self.current_analysis_runner.current_analyzer
            if analyzer:
                # 尝试直接调用分析器方法获取结果
                try:
                    if hasattr(analyzer, 'calculate_coverage') and callable(analyzer.calculate_coverage):
                        coverage = analyzer.calculate_coverage()
                        image_result['草地盖度'] = f"{coverage:.2f}%"
                        logging.info(f"[DynamicAnalysis] 直接计算盖度: {coverage:.2f}%")
                except Exception as e:
                    logging.error(f"[DynamicAnalysis] 计算盖度出错: {e}")
                
                try:
                    if hasattr(analyzer, 'calculate_density') and callable(analyzer.calculate_density):
                        density = analyzer.calculate_density()
                        image_result['草地密度'] = f"{density} 株/平方米"
                        logging.info(f"[DynamicAnalysis] 直接计算密度: {density} 株/平方米")
                except Exception as e:
                    logging.error(f"[DynamicAnalysis] 计算密度出错: {e}")
                
                # 尝试从结果字典获取高度数据
                if hasattr(analyzer, 'results') and isinstance(analyzer.results, dict):
                    for key, value in analyzer.results.items():
                        if key not in image_result:
                            image_result[key] = value
                            logging.info(f"[DynamicAnalysis] 从analyzer.results中获取: {key}={value}")
        
        # 保存分析结果到临时变量，以便后续写入数据库
        if self.monitor_mode == 'database' and self.db_manager:
            jpeg_lid = self.config.get('jpeg_lid')
            ld_lid = self.config.get('ld_lid')
            if jpeg_lid is not None and ld_lid is not None:
                # 提取并保存关键结果
                coverage = None
                height = None
                density = None
                
                # 尝试从不同的可能键中获取盖度信息
                if '草地盖度' in image_result:
                    coverage = image_result['草地盖度']
                elif 'grass_coverage' in image_result:
                    coverage = image_result['grass_coverage']
                
                # 尝试从不同的可能键中获取高度信息
                if '草地高度' in image_result:
                    height = image_result['草地高度']
                elif 'grass_height' in image_result:
                    height = image_result['grass_height']
                
                # 尝试从不同的可能键中获取密度信息
                if '草地密度' in image_result:
                    density = image_result['草地密度']
                elif 'grass_density' in image_result:
                    density = image_result['grass_density']
                #todo这个地方是两个相同的id
                # 保存到临时变量
                self.temp_analysis_results[(jpeg_lid, ld_lid)] = {
                    'coverage': coverage,
                    'height': height,
                    'density': density
                }
                logging.info(f"[DynamicAnalysis] 已保存临时分析结果: jpeg_lid={jpeg_lid}, ld_lid={ld_lid}, coverage={coverage}, height={height}, density={density}")
        
        # 写入analyzed_files.json，无论成功与否
        if result:
            logging.info(f"[DynamicAnalysis] 写入analyzed_files.json: {file_path}, results: {image_result}")
            self.files_tracker.add_analyzed_file(file_path, image_result)
        else:
            logging.info(f"[DynamicAnalysis] 写入analyzed_files.json: {file_path}, results: N/A")
            self.files_tracker.add_analyzed_file(file_path, {'草地盖度': 'N/A', '草地密度': 'N/A', '草地高度': 'N/A', 'original_path': file_path})
        
        # 发出信号，保证与普通分析流程一致
        self.analysis_complete_signal.emit([image_result])
        # 其它逻辑不变
        self.analysis_file_completed.emit(True, file_path, image_result)  # 确保这里总是发送成功信号和完整结果
        
        if self.current_analysis_thread:
            self.current_analysis_thread.quit()
            self.current_analysis_thread.wait()
            self.current_analysis_thread = None
            self.current_analysis_runner = None
        
        self.is_analyzing = False
        
        if self.pending_files:
            self.process_next_file()
        else:
            self.analysis_all_completed.emit()
            
        if self.monitor_mode == 'database' and self.db_manager:
            try:
                # 确保结果数据传递到UI 界面
                self.analysis_complete_signal.emit([image_result])
            except Exception as e:
                logging.error(f"[动态分析][DB写入] 结果写入数据库失败: {e}")
            
    def stop_current_analysis(self):
        """停止当前分析任务"""
        if self.is_analyzing and self.current_analysis_runner:
            self.current_analysis_runner.stop()
            logging.info("已请求停止当前分析任务")
            
    def clear_pending_files(self):
        """清空待分析文件队列"""
        count = len(self.pending_files)
        self.pending_files = []
        logging.info(f"已清空待分析队列，共移除 {count} 个文件")
        
    def get_analyzed_files_count(self):
        """
        获取已分析文件数量
        
        返回:
            int: 已分析文件数量
        """
        return len(self.files_tracker.get_analyzed_files()) 
        
    def _get_calibration_file_path(self, image_path):
        """
        获取给定图像对应的校准文件路径
        将校准文件保存在图片所在目录下的 calibrations 文件夹中
        """
        # 获取图片所在目录
        image_dir = os.path.dirname(image_path)
        # 在图片目录下创建 calibrations 文件夹
        calib_dir = os.path.join(image_dir, 'calibrations')
        
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(calib_dir, f"{img_basename}.json")

    def on_file_analysis_file_completed(self, success, file_path, results):
        """
        直接用AnalysisRunner的analysis_file_completed信号参数处理分析结果，避免依赖current_analyzer生命周期
        """
        logging.info(f"[DynamicAnalysis] [analysis_file_completed] 文件: {file_path}, 成功: {success}, results: {results}")
        if self.db_manager:
            jpeg_lid = self.config.get('jpeg_lid')
            ld_lid = self.config.get('ld_lid')
            # if jpeg_lid is None or ld_lid is None:根据表的lid查询设备号，根据雷达的lid查询设备号，根据设备号比较，如果设备号一样就将图片lid和雷达lid放到一起
            #     logging.warning(f"[DB写入] lid映射失败，写入NULL: jpeg_lid={jpeg_lid}, ld_lid={ld_lid}")
            # 确保表存在
            create_sql = '''
            CREATE TABLE IF NOT EXISTS nk_analysis_results (
             "id" int4 NOT NULL,
             "lqn" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
             "jpeg_lid" int8,
             "ld_lid" int8,
             "coverage" numeric,
             "height" numeric,
             "density" numeric,
             "gnaw" bool,
             "lct" timestamp(6) NOT NULL
            )'''
            self.db_manager.execute_update(create_sql)
            # 提取分析结果
            # 辅助函数：从字符串中提取数值（去除单位
            def extract_number(val):
                if val is None:
                    return None
                if isinstance(val, str):
                    val = val.replace('%', '').replace('mm', '').replace('株/平方米', '').strip()
                try:
                    return float(val)
                except Exception:
                    return None
            coverage = extract_number(results.get('草地盖度'))
            height = extract_number(results.get('草地高度'))
            density = extract_number(results.get('草地密度'))
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #TODO 判断图片lid与雷达lid那个为空，


            update_sql = '''
            UPDATE nk_analysis_results SET coverage=%s, height=%s, density=%s, lct=%s
            WHERE jpeg_lid=%s AND ld_lid=%s
            '''
            try:
                updated = self.db_manager.execute_update(update_sql, (coverage, height, density, now, jpeg_lid, ld_lid))
                if updated == 0:
                    insert_sql = '''
                    INSERT INTO nk_analysis_results (jpeg_lid, ld_lid, coverage, height, density, lct)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    '''
                    self.db_manager.execute_update(insert_sql, (jpeg_lid, ld_lid, coverage, height, density, now))
                logging.info(f"[DB写入] 写入: jpeg_lid={jpeg_lid}, ld_lid={ld_lid}, coverage={coverage}, height={height}, density={density}")
            except Exception as e:
                logging.error(f"[DB写入] 写入失败: {e}")
        else:
            # 本地模式
            if success:
                self.files_tracker.add_analyzed_file(file_path, results)
            else:
                self.files_tracker.add_analyzed_file(file_path, {'草地盖度': 'N/A', '草地密度': 'N/A', '草地高度': 'N/A', 'original_path': file_path})
        logging.info(f"[DynamicAnalysis] [analysis_file_completed] 发射analysis_file_completed信号: {file_path}, results: {results}")
        self.analysis_file_completed.emit(success, file_path, results)

    def analyze_all_by_device(self, output_dir="output_images"):
        """
        按设备号批量分析所有图片盖度和雷达高度，并写入数据库（非阻塞队列版）
        """
        if not self.db_manager:
            logging.error("未设置db_manager，无法进行数据库分析")
            self.analysis_log.emit("错误：未设置数据库管理器")
            return

        try:
            self.analysis_log.emit("正在查询所有设备号...")
            sql = '''
                SELECT lqn FROM nk_ope_data_lc301b_jpeg
                UNION
                SELECT lqn FROM nk_open_data_lc301b_ld
            '''
            device_rows = self.db_manager.execute_query(sql)
            device_ids = [row['lqn'] for row in device_rows]
            if not device_ids:
                self.analysis_log.emit("未找到任何设备号")
                return

            self.analysis_log.emit(f"找到 {len(device_ids)} 个设备号")

            # 统计总任务数用于进度计算
            total_tasks = 0
            device_task_counts = {}
            all_tasks = []
            for lqn in device_ids:
                jpeg_rows = self.db_manager.execute_query(
                    "SELECT lid FROM nk_ope_data_lc301b_jpeg WHERE lqn = %s", (lqn,))
                ld_rows = self.db_manager.execute_query(
                    "SELECT lid FROM nk_open_data_lc301b_ld WHERE lqn = %s", (lqn,))
                jpeg_ids = [row['lid'] for row in jpeg_rows]
                ld_ids = [row['lid'] for row in ld_rows]
                device_task_counts[lqn] = len(jpeg_ids) + len(ld_ids)
                total_tasks += device_task_counts[lqn]
                # 生成所有分析任务
                for jpeg_lid in jpeg_ids:
                    all_tasks.append({
                        'type': 'image',
                        'jpeg_lid': jpeg_lid,
                        'lqn': lqn
                    })
                for ld_lid in ld_ids:
                    all_tasks.append({
                        'type': 'lidar',
                        'ld_lid': ld_lid,
                        'lqn': lqn
                    })
                logging.info(f"[任务生成] 设备号 {lqn}: 图片任务 {len(jpeg_ids)} 个, 雷达任务 {len(ld_ids)} 个")
            if total_tasks == 0:
                self.analysis_log.emit("未找到任何需要分析的数据")
                return
            self.analysis_log.emit(f"总共需要分析 {total_tasks} 个数据项")
            logging.info(f"[任务生成] 总任务数: {total_tasks}, 任务列表: {[task['type'] for task in all_tasks]}")
            os.makedirs(output_dir, exist_ok=True)
            self._device_analysis_total_tasks = total_tasks
            self._device_analysis_completed_tasks = 0
            self._device_analysis_output_dir = output_dir
            self._device_analysis_queue = all_tasks
            self._run_next_device_analysis_task()
        except Exception as e:
            logging.error(f"按设备号批量分析失败: {e}")
            self.analysis_log.emit(f"批量分析失败: {e}")
            self.analysis_all_completed.emit()

    def _run_next_device_analysis_task(self):
        """
        队列驱动：启动下一个分析任务，分析完成后自动继续
        """
        if not hasattr(self, '_device_analysis_queue') or not self._device_analysis_queue:
            self.analysis_progress.emit(100)
            self.analysis_log.emit(f"所有设备号分析完成！共处理 {getattr(self, '_device_analysis_completed_tasks', 0)} 个数据项")
            self.analysis_all_completed.emit()
            return
        task = self._device_analysis_queue.pop(0)
        logging.info(f"[队列处理] 当前任务: type={task['type']}, task={task}")
        output_dir = getattr(self, '_device_analysis_output_dir', "output_images")
        # 构造分析配置
        analysis_config = deepcopy(self.config)
        analysis_config['output_dir'] = output_dir
        if task['type'] == 'image':
            jpeg_lid = task['jpeg_lid']
            lqn = task['lqn']
            self.analysis_log.emit(f"分析图片 {jpeg_lid} (设备号: {lqn})")
            logging.info(f"[队列处理] 开始处理图片任务: jpeg_lid={jpeg_lid}, lqn={lqn}")
            img_path = self.db_manager.save_jpeg_to_file(jpeg_lid, output_dir=output_dir)
            analysis_config['input_paths'] = [img_path]
            analysis_config['jpeg_lid'] = jpeg_lid
            analysis_config['lqn'] = lqn
            analysis_config['ld_lid'] = None
            analysis_config['perform_lidar_analysis'] = False
            callback = partial(self._on_device_analysis_completed, jpeg_lid=jpeg_lid, lqn=lqn, ld_lid=None)
        else:
            ld_lid = task['ld_lid']
            lqn = task['lqn']
            self.analysis_log.emit(f"分析雷达 {ld_lid} (设备号: {lqn})")
            logging.info(f"[队列处理] 开始处理雷达任务: ld_lid={ld_lid}, lqn={lqn}")
            analysis_config['input_paths'] = []
            analysis_config['ld_lid'] = ld_lid
            analysis_config['lqn'] = lqn
            analysis_config['perform_lidar_analysis'] = True
            analysis_config['jpeg_lid'] = None
            callback = partial(self._on_device_analysis_completed, jpeg_lid=None, lqn=lqn, ld_lid=ld_lid)
        # 创建分析线程和运行器
        analysis_thread = QThread()
        analysis_runner = AnalysisRunner(analysis_config, db_manager_ref=self.db_manager)
        analysis_runner.moveToThread(analysis_thread)
        analysis_thread.started.connect(analysis_runner.run)
        analysis_runner.log_message.connect(self.analysis_log)
        analysis_runner.analysis_file_completed.connect(callback)
        # 分析完成后自动继续下一个
        analysis_runner.analysis_file_completed.connect(lambda *args: self._on_device_analysis_task_finished())
        analysis_thread.start()
        # 保存引用，防止被GC
        self._current_analysis_thread = analysis_thread
        self._current_analysis_runner = analysis_runner

    def _on_device_analysis_task_finished(self):
        # 更新进度
        self._device_analysis_completed_tasks += 1
        progress = int((self._device_analysis_completed_tasks / self._device_analysis_total_tasks) * 100)
        self.analysis_progress.emit(progress)
        logging.info(f"[任务完成] 已完成 {self._device_analysis_completed_tasks}/{self._device_analysis_total_tasks} 个任务, 进度: {progress}%")
        # 释放线程引用
        self._current_analysis_thread.quit()
        self._current_analysis_thread.wait()
        self._current_analysis_thread = None
        self._current_analysis_runner = None
        logging.info(f"[任务完成] 线程已释放，剩余队列长度: {len(self._device_analysis_queue)}")
        # 继续下一个
        self._run_next_device_analysis_task()

    def _on_device_analysis_completed(self, success, file_path, results, jpeg_lid=None, lqn=None, ld_lid=None):
        """
        设备分析完成回调函数
        """
        try:
            logging.info(f"[分析完成回调] success={success}, file_path={file_path}, jpeg_lid={jpeg_lid}, ld_lid={ld_lid}, lqn={lqn}, results={results}")
            if not success:
                logging.warning(f"分析失败: file_path={file_path}")
                return
            # 确保表存在
            create_sql = '''
            CREATE TABLE IF NOT EXISTS nk_analysis_results (
              id SERIAL PRIMARY KEY,
              lqn varchar(255) NOT NULL,
              jpeg_lid int8,
              ld_lid int8,
              coverage numeric,
              height numeric,
              density numeric,
              gnaw bool,
              lct timestamp(6) NOT NULL
            )'''
            try:
                self.db_manager.execute_update(create_sql)
            except Exception as e:
                logging.error(f"[DB建表异常] {e}", exc_info=True)
                return
            # 提取分析结果
            def extract_number(val):
                if val is None:
                    return None
                if isinstance(val, str):
                    val = val.replace('%', '').replace('mm', '').replace('株/平方米', '').strip()
                try:
                    return float(val)
                except Exception:
                    return None
            coverage = extract_number(results.get('草地盖度'))
            height = extract_number(results.get('草地高度'))
            density = extract_number(results.get('草地密度'))
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 日志：准备写入数据库
            logging.info(f"[DB写入] 准备写入: jpeg_lid={jpeg_lid}, ld_lid={ld_lid}, lqn={lqn}, coverage={coverage}, height={height}, density={density}, time={now}")
            # 根据分析类型写入数据库
            if jpeg_lid is not None:
                # 图片分析结果
                update_sql = '''
                UPDATE nk_analysis_results SET coverage=%s, density=%s, lqn=%s, lct=%s
                WHERE jpeg_lid=%s
                '''
                try:
                    updated = self.db_manager.execute_update(update_sql, (coverage, density, lqn, now, jpeg_lid))
                    if updated == 0:
                        insert_sql = '''
                        INSERT INTO nk_analysis_results (jpeg_lid, lqn, coverage, density, lct)
                        VALUES (%s, %s, %s, %s, %s)
                        '''
                        self.db_manager.execute_update(insert_sql, (jpeg_lid, lqn, coverage, density, now))
                    logging.info(f"[DB写入] 图片结果写入完成: jpeg_lid={jpeg_lid}, lqn={lqn}, coverage={coverage}, density={density}")
                except Exception as e:
                    logging.error(f"[DB写入异常] 图片: {e}", exc_info=True)
            elif ld_lid is not None:
                # 雷达分析结果
                update_sql = '''
                UPDATE nk_analysis_results SET height=%s, lqn=%s, lct=%s
                WHERE ld_lid=%s
                '''
                try:
                    updated = self.db_manager.execute_update(update_sql, (height, lqn, now, ld_lid))
                    if updated == 0:
                        insert_sql = '''
                        INSERT INTO nk_analysis_results (ld_lid, lqn, height, lct)
                        VALUES (%s, %s, %s, %s)
                        '''
                        self.db_manager.execute_update(insert_sql, (ld_lid, lqn, height, now))
                    logging.info(f"[DB写入] 雷达结果写入完成: ld_lid={ld_lid}, lqn={lqn}, height={height}")
                except Exception as e:
                    logging.error(f"[DB写入异常] 雷达: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"[设备分析] 回调处理失败: {e}", exc_info=True)
        