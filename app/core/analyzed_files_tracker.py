import os
import json
import logging
from datetime import datetime

class AnalyzedFilesTracker:
    """
    用于跟踪已分析过的文件
    
    功能：
    1. 记录已分析过的文件名称
    2. 检查文件是否已被分析
    3. 添加新分析的文件到记录中
    """
    
    def __init__(self, record_file_path="app/data/analyzed_files.json"):
        """
        初始化跟踪器
        
        参数:
            record_file_path: 记录文件的路径
        """
        self.record_file_path = record_file_path
        self.analyzed_files = {}
        self._ensure_record_file_exists()
        self._load_records()
    
    def _ensure_record_file_exists(self):
        """确保记录文件存在，如果不存在则创建"""
        directory = os.path.dirname(self.record_file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"创建目录: {directory}")
            except Exception as e:
                logging.error(f"创建目录失败: {e}")
                
        if not os.path.exists(self.record_file_path):
            try:
                with open(self.record_file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                logging.info(f"创建分析记录文件: {self.record_file_path}")
            except Exception as e:
                logging.error(f"创建分析记录文件失败: {e}")
    
    def _load_records(self):
        """从文件加载已分析记录"""
        try:
            with open(self.record_file_path, 'r', encoding='utf-8') as f:
                self.analyzed_files = json.load(f)
            logging.info(f"已加载分析记录，共 {len(self.analyzed_files)} 条")
        except Exception as e:
            logging.error(f"加载分析记录失败: {e}")
            self.analyzed_files = {}
    
    def _save_records(self):
        """保存记录到文件"""
        try:
            with open(self.record_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.analyzed_files, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存分析记录，共 {len(self.analyzed_files)} 条")
        except Exception as e:
            logging.error(f"保存分析记录失败: {e}")
    
    def is_file_analyzed(self, file_path):
        """
        检查文件是否已被分析
        
        参数:
            file_path: 文件路径
            
        返回:
            bool: 是否已分析
        """
        file_name = os.path.basename(file_path)
        return file_name in self.analyzed_files
    
    def add_analyzed_file(self, file_path, analysis_results=None):
        """
        添加已分析文件到记录
        
        参数:
            file_path: 文件路径
            analysis_results: 分析结果，可选
        """
        file_name = os.path.basename(file_path)
        self.analyzed_files[file_name] = {
            "file_path": file_path,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": analysis_results
        }
        self._save_records()
        logging.info(f"已添加分析记录: {file_name}")
    
    def get_analyzed_files(self):
        """
        获取所有已分析文件
        
        返回:
            dict: 已分析文件记录
        """
        return self.analyzed_files
    
    def find_new_files(self, directory):
        """
        查找目录中未分析的新文件
        
        参数:
            directory: 要搜索的目录
            
        返回:
            list: 未分析的文件路径列表
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logging.warning(f"目录不存在或不是有效目录: {directory}")
            return []
            
        new_files = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    if not self.is_file_analyzed(file_path):
                        new_files.append(file_path)
        
        logging.info(f"在 {directory} 中找到 {len(new_files)} 个未分析的新文件")
        return new_files 