import os
import time
import logging
import json
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
        
    def check_for_new_files(self):
        """检查是否有新文件需要分析"""
        if self.is_analyzing:
            return  # 如果正在分析，跳过检查
            
        input_dir = self.config.get('input_dir')
        if not input_dir or not os.path.exists(input_dir):
            logging.warning(f"输入目录不存在或未设置: {input_dir}")
            return
            
        new_files = self.files_tracker.find_new_files(input_dir)
        if new_files:
            self.new_files_found.emit(new_files)
            self.pending_files.extend(new_files)
            logging.info(f"发现 {len(new_files)} 个新文件，添加到待分析队列")
            
            # 如果当前没有分析任务，立即开始分析
            if not self.is_analyzing:
                self.process_next_file()
                
    def process_next_file(self):
        """处理队列中的下一个文件"""
        if not self.pending_files:
            logging.info("所有文件已分析完成")
            self.analysis_all_completed.emit()
            return
            
        # 获取下一个文件
        next_file = self.pending_files.pop(0)
        self.analyze_file(next_file)
        
    def analyze_file(self, file_path):
        """
        分析单个文件
        
        参数:
            file_path: 文件路径
        """
        if self.is_analyzing:
            logging.warning(f"当前有分析任务正在进行，文件 {file_path} 将稍后处理")
            if file_path not in self.pending_files:
                self.pending_files.append(file_path)
            return
            
        self.is_analyzing = True
        self.analysis_started.emit(file_path)
        logging.info(f"开始分析文件: {file_path}")
        
        # 创建分析配置
        analysis_config = self.config.copy()
        analysis_config['input_paths'] = [file_path]
        
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
                    logging.info("使用自动校准模式，跳过手动校准步骤")
                    # 这里可以添加自动校准的代码
                    # 目前先留空，直接进入分析流程
                else:
                    # 手动模式但没有校准文件，发出警告
                    logging.warning("手动校准模式下未找到校准文件，将使用原始尺寸")
        
        # 创建分析线程和运行器
        self.current_analysis_thread = QThread()
        self.current_analysis_runner = AnalysisRunner(analysis_config)
        self.current_analysis_runner.moveToThread(self.current_analysis_thread)
        
        # 连接信号
        self.current_analysis_thread.started.connect(self.current_analysis_runner.run)
        self.current_analysis_runner.progress_updated.connect(self.analysis_progress)
        self.current_analysis_runner.log_message.connect(self.analysis_log)
        self.current_analysis_runner.analysis_complete.connect(self.on_file_analysis_complete)
        
        # 启动分析线程
        self.current_analysis_thread.start()
        
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
            
        self.is_analyzing = False
        
        # 处理下一个文件
        if self.pending_files:
            self.process_next_file()
        else:
            self.analysis_all_completed.emit()
            
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