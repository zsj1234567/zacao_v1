#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import pool
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
from PIL import Image
import io
import struct
import os
from datetime import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import stats

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class PostgresManager:
    """PostgreSQL数据库连接管理器"""
    
    def __init__(self, 
                 dbname: str,
                 user: str,
                 password: str,
                 host: str = 'localhost',
                 port: str = '5432',
                 min_conn: int = 1,
                 max_conn: int = 10):
        """
        初始化数据库连接池
        
        Args:
            dbname: 数据库名称
            user: 用户名
            password: 密码
            host: 主机地址
            port: 端口号
            min_conn: 最小连接数
            max_conn: 最大连接数
        """
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                database=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            logging.info("数据库连接池创建成功")
        except Exception as e:
            logging.error(f"创建数据库连接池失败: {str(e)}")
            raise

    def get_connection(self):
        """获取数据库连接"""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """归还连接到连接池"""
        self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        执行查询操作
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 将结果转换为字典列表
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            logging.error(f"查询执行失败: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)

    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        执行更新操作（INSERT, UPDATE, DELETE）
        
        Args:
            query: SQL更新语句
            params: 更新参数
            
        Returns:
            受影响的行数
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"更新执行失败: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)

    def close(self):
        """关闭连接池"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logging.info("数据库连接池已关闭")

    # 添加专门的表读取方法
    def get_atm_data(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        读取 nk_ope_data_lc301b_atm 表数据
        
        Args:
            limit: 限制返回的记录数
            offset: 跳过的记录数
            
        Returns:
            查询结果列表
        """
        query = "SELECT * FROM nk_ope_data_lc301b_atm"
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_jpeg_data(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        读取 nk_ope_data_lc301b_jpeg 表数据
        
        Args:
            limit: 限制返回的记录数
            offset: 跳过的记录数
            
        Returns:
            查询结果列表
        """
        query = "SELECT * FROM nk_ope_data_lc301b_jpeg"
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_ptm_data(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        读取 nk_ope_data_lc301b_ptm 表数据
        
        Args:
            limit: 限制返回的记录数
            offset: 跳过的记录数
            
        Returns:
            查询结果列表
        """
        query = "SELECT * FROM nk_ope_data_lc301b_ptm"
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_ld_data(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        读取 nk_open_data_lc301b_ld 表数据
        
        Args:
            limit: 限制返回的记录数
            offset: 跳过的记录数
            
        Returns:
            查询结果列表
        """
        query = "SELECT * FROM nk_open_data_lc301b_ld"
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_table_columns(self, table_name: str) -> List[str]:
        """
        获取指定表的列名
        
        Args:
            table_name: 表名
            
        Returns:
            列名列表
        """
        query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position
        """
        results = self.execute_query(query, (table_name,))
        return [row['column_name'] for row in results]

    def save_jpeg_to_file(self, lid: int, output_dir: str = "output_images") -> str:
        """
        将JPEG数据保存为图片文件
        
        Args:
            lid: 数据ID
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 查询JPEG数据
            query = "SELECT ljpg, lct FROM nk_ope_data_lc301b_jpeg WHERE lid = %s"
            results = self.execute_query(query, (lid,))
            
            if not results:
                raise ValueError(f"未找到ID为{lid}的图片数据")
                
            jpeg_data = results[0]['ljpg']
            timestamp = results[0]['lct']
            
            # 构建输出文件名（使用时间戳）
            filename = f"image_{lid}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 将二进制数据转换为图片并保存
            image = Image.open(io.BytesIO(jpeg_data))
            image.save(filepath)
            
            logging.info(f"图片已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"保存图片失败: {str(e)}")
            raise

    def parse_ld_data(self, lid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        解析激光检测数据
        
        Args:
            lid: 数据ID
            
        Returns:
            tuple: (X值数组, Y值数组, Z值数组)，每个数组形状为(60, 160)
        """
        try:
            # 查询激光数据
            query = "SELECT ldn FROM nk_open_data_lc301b_ld WHERE lid = %s"
            results = self.execute_query(query, (lid,))
            
            if not results:
                raise ValueError(f"未找到ID为{lid}的激光数据")
                
            ld_data = results[0]['ldn']
            
            # 创建numpy数组来存储解析后的数据
            X = np.zeros((60, 160), dtype=np.float32)
            Y = np.zeros((60, 160), dtype=np.float32)
            Z = np.zeros((60, 160), dtype=np.float32)
            
            # 解析二进制数据
            for i in range(60):
                for j in range(160):
                    idx = (i * 160 + j) * 12  # 每个点12字节
                    x = struct.unpack('f', ld_data[idx:idx+4])[0]
                    y = struct.unpack('f', ld_data[idx+4:idx+8])[0]
                    z = struct.unpack('f', ld_data[idx+8:idx+12])[0]
                    
                    X[i,j] = x
                    Y[i,j] = y
                    Z[i,j] = z
            
            return X, Y, Z
            
        except Exception as e:
            logging.error(f"解析激光数据失败: {str(e)}")
            raise

    def get_image_to_jpeg_lid_map(self, output_dir="output_images"):
        """
        返回图片路径到JPEG lid的映射 {img_path: jpeg_lid}
        """
        jpeg_records = self.get_jpeg_data()
        image_to_jpeg_lid = {}
        for rec in jpeg_records:
            lid = rec['lid']
            img_path = self.save_jpeg_to_file(lid, output_dir=output_dir)
            image_to_jpeg_lid[img_path] = lid
        return image_to_jpeg_lid

    def get_image_to_ld_lid_map(self, output_dir="output_images"):
        """
        返回图片路径到LD lid的映射 {img_path: ld_lid}
        """
        jpeg_records = self.get_jpeg_data()
        ld_records = self.get_ld_data()
        image_to_ld_lid = {}
        for idx, rec in enumerate(jpeg_records):
            lid = rec['lid']
            img_path = self.save_jpeg_to_file(lid, output_dir=output_dir)
            if idx < len(ld_records):
                image_to_ld_lid[img_path] = ld_records[idx]['lid']
            else:
                image_to_ld_lid[img_path] = None
        return image_to_ld_lid

class DataProcessor:
    """数据处理器：处理和整理从数据库获取的数据"""
    
    def __init__(self, db_manager: PostgresManager, output_base_dir: str = "output_data"):
        """
        初始化数据处理器
        
        Args:
            db_manager: PostgreSQL数据库管理器实例
            output_base_dir: 输出数据的基础目录
        """
        self.db_manager = db_manager
        self.output_base_dir = output_base_dir
        self.setup_directories()
        
    def _remove_outliers(self, data: np.ndarray, percentile_low: float = 10, percentile_high: float = 90, 
                        z_threshold: float = 3.0, method: str = 'combined') -> Tuple[np.ndarray, float, float]:
        """
        移除异常值，返回处理后的数据和有效数值范围
        
        Args:
            data: 输入数据数组
            percentile_low: 下限百分位数，默认10%
            percentile_high: 上限百分位数，默认90%
            z_threshold: Z分数阈值，超过此值的点被视为异常值(默认3.0)
            method: 异常值检测方法，可选: 'percentile', 'z_score', 'iqr', 'combined'
            
        Returns:
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
        
    def setup_directories(self):
        """只创建主输出目录，不创建任何子目录，所有输出直接放在output_base_dir根目录下"""
        os.makedirs(self.output_base_dir, exist_ok=True)
        self.dirs = {
            '3d_plots': self.output_base_dir,
            'heatmaps': self.output_base_dir,
            'json': self.output_base_dir
        }
            
    def process_all_data(self):
        """处理所有表的数据"""
        # 处理ATM数据
        atm_data = self.process_atm_data()
        
        # 处理JPEG数据
        jpeg_data = self.process_jpeg_data()
        
        # 处理PTM数据
        ptm_data = self.process_ptm_data()
        
        # 处理LD数据
        ld_data = self.process_ld_data()
        
        # 生成汇总报告
        self.generate_summary_report(atm_data, jpeg_data, ptm_data, ld_data)
        
    def process_atm_data(self) -> pd.DataFrame:
        """处理大气数据"""
        logging.info("开始处理ATM数据...")
        
        # 获取所有ATM数据
        atm_data = self.db_manager.get_atm_data()
        df_atm = pd.DataFrame(atm_data)
        
        # 保存为CSV
        csv_path = os.path.join(self.dirs['csv'], 'atm_data.csv')
        df_atm.to_csv(csv_path, index=False)
        logging.info(f"ATM数据已保存到: {csv_path}")
        
        return df_atm
        
    def process_jpeg_data(self) -> pd.DataFrame:
        """处理图像数据"""
        logging.info("开始处理JPEG数据...")
        
        # 获取所有JPEG数据
        jpeg_data = self.db_manager.get_jpeg_data()
        
        # 提取基本信息（不包含二进制数据）
        jpeg_info = []
        for record in jpeg_data:
            jpeg_info.append({
                'lid': record['lid'],
                'lqn': record['lqn'],
                'lct': record['lct']
            })
            
            # 保存图片
            try:
                image_path = self.db_manager.save_jpeg_to_file(
                    record['lid'],
                    output_dir=self.dirs['images']
                )
                logging.info(f"保存图片: {image_path}")
            except Exception as e:
                logging.error(f"保存图片失败 (lid={record['lid']}): {str(e)}")
        
        # 转换为DataFrame并保存
        df_jpeg = pd.DataFrame(jpeg_info)
        csv_path = os.path.join(self.dirs['csv'], 'jpeg_info.csv')
        df_jpeg.to_csv(csv_path, index=False)
        
        return df_jpeg
        
    def process_ptm_data(self) -> pd.DataFrame:
        """处理PTM数据"""
        logging.info("开始处理PTM数据...")
        
        # 获取所有PTM数据
        ptm_data = self.db_manager.get_ptm_data()
        df_ptm = pd.DataFrame(ptm_data)
        
        # 保存为CSV
        csv_path = os.path.join(self.dirs['csv'], 'ptm_data.csv')
        df_ptm.to_csv(csv_path, index=False)
        logging.info(f"PTM数据已保存到: {csv_path}")
        
        return df_ptm
        
    def process_ld_data(self) -> pd.DataFrame:
        """处理激光检测数据，所有输出直接放在output_base_dir根目录下"""
        logging.info("开始处理LD数据...")
        ld_data = self.db_manager.get_ld_data()
        ld_info = []
        for record in ld_data:
            lid = record['lid']
            ld_info.append({
                'lid': lid,
                'lqn': record['lqn'],
                'lct': record['lct']
            })
            try:
                X, Y, Z = self.db_manager.parse_ld_data(lid)
                Z_filtered, z_min, z_max = self._remove_outliers(
                    Z, percentile_low=10, percentile_high=90,
                    z_threshold=3.0, method='combined'
                )
                stats = {
                    'lid': lid,
                    'original_data': {
                        'x_range': (float(X.min()), float(X.max())),
                        'y_range': (float(Y.min()), float(Y.max())),
                        'z_range': (float(Z.min()), float(Z.max())),
                        'z_mean': float(Z.mean()),
                        'z_std': float(Z.std()),
                        'z_median': float(np.median(Z)),
                        'z_q1': float(np.percentile(Z, 25)),
                        'z_q3': float(np.percentile(Z, 75)),
                        'z_iqr': float(np.percentile(Z, 75) - np.percentile(Z, 25))
                    },
                    'filtered_data': {
                        'x_range': (float(X.min()), float(X.max())),
                        'y_range': (float(Y.min()), float(Y.max())),
                        'z_range': (float(z_min), float(z_max)),
                        'z_mean': float(np.nanmean(Z_filtered)),
                        'z_std': float(np.nanstd(Z_filtered)),
                        'z_median': float(np.nanmedian(Z_filtered)),
                        'z_q1': float(np.nanpercentile(Z_filtered, 25)),
                        'z_q3': float(np.nanpercentile(Z_filtered, 75)),
                        'z_iqr': float(np.nanpercentile(Z_filtered, 75) - np.nanpercentile(Z_filtered, 25)),
                        'z_height': float(z_max - np.nanmean(Z_filtered))
                    },
                    'shape': {
                        'rows': Z.shape[0],
                        'cols': Z.shape[1],
                        'total_points': Z.size
                    },
                    'outlier_info': {
                        'outlier_count': int(np.sum(np.isnan(Z_filtered))),
                        'outlier_percent': float(np.sum(np.isnan(Z_filtered)) / Z.size * 100)
                    }
                }
                stats_file = os.path.join(self.output_base_dir, f"ld_stats_{lid}.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=4)
                # 只生成3D点云图和热力图，直接保存在根目录
                self._create_3d_scatter_plot(X, Y, Z, Z_filtered, lid, self.output_base_dir)
                self._create_heatmap(Z, Z_filtered, lid, self.output_base_dir)
            except Exception as e:
                logging.error(f"处理LD数据失败 (lid={lid}): {str(e)}")
                logging.exception("详细错误信息")
        import pandas as pd
        df_ld = pd.DataFrame(ld_info)
        return df_ld
        


    def _create_3d_scatter_plot(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, Z_filtered: np.ndarray,
                              lid: int, output_dir: str):
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
            
            logging.info(f"3D散点图已保存到: {plot_path}")
            return plot_path
        except Exception as e:
            logging.error(f"创建3D散点图失败: {str(e)}")
            logging.exception("详细错误信息")
    
    def _create_heatmap(self, Z: np.ndarray, Z_filtered: np.ndarray,
                         lid: int, output_dir: str):
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
            
            # 保存一个额外的差异热力图，显示处理前后的差异
            self._create_difference_heatmap(Z, Z_filtered, lid, output_dir)
            
            # 保存异常值分布图
            self._create_outlier_map(Z, Z_filtered, lid, output_dir)
            
            logging.info(f"热力图已保存到: {plot_path}")
            return plot_path
        except Exception as e:
            logging.error(f"创建热力图失败: {str(e)}")
            logging.exception("详细错误信息")
    
    def _create_difference_heatmap(self, Z_original: np.ndarray, Z_filtered: np.ndarray,
                                 lid: int, output_dir: str):
        """创建显示处理前后差异的热力图"""
        try:
            # 计算差异
            diff = np.abs(Z_original - Z_filtered)
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            
            # 使用热力图显示差异
            im = ax.imshow(diff, cmap='hot', interpolation='bilinear', aspect='auto')
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('差异值 (mm)', fontsize=10)
            
            # 设置标题和标签
            ax.set_title(f"异常值处理前后的差异图 (ID: {lid})", fontsize=12, pad=15)
            ax.set_xlabel('X方向采样点 (共160点)', fontsize=10)
            ax.set_ylabel('Y方向采样点 (共60点)', fontsize=10)
            
            # 添加说明
            non_zero_diff = np.count_nonzero(diff)
            total_points = diff.size
            percent = (non_zero_diff / total_points) * 100
            
            ax.text(0.5, -0.15, f"被识别为异常值的点数: {non_zero_diff} ({percent:.2f}%)", 
                  transform=ax.transAxes, ha='center', fontsize=10)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 设置刻度
            x_ticks = np.linspace(0, diff.shape[1]-1, 5, dtype=int)
            y_ticks = np.linspace(0, diff.shape[0]-1, 4, dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            diff_path = os.path.join(output_dir, f"diff_heatmap_{lid}.png")
            plt.savefig(diff_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logging.info(f"差异热力图已保存到: {diff_path}")
        except Exception as e:
            logging.error(f"创建差异热力图失败: {str(e)}")
            logging.exception("详细错误信息")
    
    def _create_outlier_map(self, Z_original: np.ndarray, Z_filtered: np.ndarray,
                         lid: int, output_dir: str):
        """创建异常值分布图"""
        try:
            # 计算原始数据和过滤数据的差异
            diff = np.abs(Z_original - Z_filtered)
            
            # 创建二值掩码，标记异常点位置
            is_outlier = diff > 0
            
            # 计算各统计指标
            Z_mean = np.mean(Z_original)
            Z_median = np.median(Z_original)
            Z_std = np.std(Z_original)
            Z_min = np.min(Z_original)
            Z_max = np.max(Z_original)
            
            # 创建热力图，同时展示异常值分布和数据分布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
            
            # 左图：显示异常值分布
            im1 = ax1.imshow(is_outlier, cmap='hot_r', interpolation='none', aspect='auto')
            ax1.set_title(f"异常值分布图 (ID: {lid})", fontsize=12, pad=15)
            ax1.set_xlabel('X方向采样点 (共160点)', fontsize=10)
            ax1.set_ylabel('Y方向采样点 (共60点)', fontsize=10)
            
            # 添加网格线
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # 统计异常值信息并添加说明
            outlier_count = np.sum(is_outlier)
            total_points = Z_original.size
            outlier_percent = (outlier_count / total_points) * 100
            
            # 右图：显示Z分数分布
            # 计算每个点的Z分数，显示与平均值的离差程度
            z_scores = np.abs((Z_original - Z_mean) / (Z_std + 1e-10))
            im2 = ax2.imshow(z_scores, cmap='nipy_spectral', interpolation='bilinear', aspect='auto')
            ax2.set_title(f"Z分数分布图 (ID: {lid})", fontsize=12, pad=15)
            ax2.set_xlabel('X方向采样点 (共160点)', fontsize=10)
            ax2.set_ylabel('Y方向采样点 (共60点)', fontsize=10)
            
            # 添加颜色条
            cbar2 = fig.colorbar(im2, ax=ax2)
            cbar2.set_label('|Z分数| (与平均值的标准差倍数)', fontsize=10)
            
            # 添加网格线
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            # 设置刻度
            for ax in [ax1, ax2]:
                x_ticks = np.linspace(0, Z_original.shape[1]-1, 5, dtype=int)
                y_ticks = np.linspace(0, Z_original.shape[0]-1, 4, dtype=int)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
            
            # 添加总标题和统计信息
            fig.suptitle(f"异常值分析 (数据ID: {lid})", fontsize=14)
            info_text = (
                f"异常值数量: {outlier_count} ({outlier_percent:.2f}%)\n"
                f"平均值: {Z_mean:.2f}, 中位数: {Z_median:.2f}, 标准差: {Z_std:.2f}\n"
                f"最小值: {Z_min:.2f}, 最大值: {Z_max:.2f}, 范围: {Z_max-Z_min:.2f}"
            )
            fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='lightgray', alpha=0.5))
            
            # 调整布局
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])
            
            # 保存图表
            outlier_path = os.path.join(output_dir, f"outlier_map_{lid}.png")
            plt.savefig(outlier_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logging.info(f"异常值分布图已保存到: {outlier_path}")
        except Exception as e:
            logging.error(f"创建异常值分布图失败: {str(e)}")
            logging.exception("详细错误信息")
    
    def generate_summary_report(self, df_atm: pd.DataFrame, df_jpeg: pd.DataFrame,
                              df_ptm: pd.DataFrame, df_ld: pd.DataFrame):
        """生成数据汇总报告"""
        # 收集所有LD统计数据
        ld_stats_files = [f for f in os.listdir(self.dirs['json']) if f.startswith('ld_stats_')]
        ld_stats_list = []
        
        for stats_file in ld_stats_files:
            try:
                with open(os.path.join(self.dirs['json'], stats_file), 'r') as f:
                    stats_data = json.load(f)
                    ld_stats_list.append(stats_data)
            except Exception as e:
                logging.error(f"读取统计文件失败 {stats_file}: {str(e)}")
        
        # 计算激光雷达数据的汇总统计信息
        ld_summary = {}
        if ld_stats_list:
            # 初始化汇总数据结构
            ld_summary = {
                'original_data': {
                    'z_mean_avg': 0.0,
                    'z_std_avg': 0.0,
                    'z_median_avg': 0.0,
                    'z_min': float('inf'),
                    'z_max': float('-inf')
                },
                'filtered_data': {
                    'z_mean_avg': 0.0,
                    'z_std_avg': 0.0,
                    'z_median_avg': 0.0,
                    'z_min': float('inf'),
                    'z_max': float('-inf')
                },
                'outlier_info': {
                    'avg_outlier_percent': 0.0,
                    'max_outlier_percent': 0.0,
                    'min_outlier_percent': 100.0
                },
                'total_records': len(ld_stats_list)
            }
            
            # 累加统计值
            for stats in ld_stats_list:
                # 原始数据统计
                orig = stats.get('original_data', {})
                ld_summary['original_data']['z_mean_avg'] += orig.get('z_mean', 0)
                ld_summary['original_data']['z_std_avg'] += orig.get('z_std', 0)
                ld_summary['original_data']['z_median_avg'] += orig.get('z_median', 0)
                
                if 'z_range' in orig:
                    ld_summary['original_data']['z_min'] = min(ld_summary['original_data']['z_min'], orig['z_range'][0])
                    ld_summary['original_data']['z_max'] = max(ld_summary['original_data']['z_max'], orig['z_range'][1])
                
                # 过滤数据统计
                filt = stats.get('filtered_data', {})
                ld_summary['filtered_data']['z_mean_avg'] += filt.get('z_mean', 0)
                ld_summary['filtered_data']['z_std_avg'] += filt.get('z_std', 0)
                ld_summary['filtered_data']['z_median_avg'] += filt.get('z_median', 0)
                
                if 'z_range' in filt:
                    ld_summary['filtered_data']['z_min'] = min(ld_summary['filtered_data']['z_min'], filt['z_range'][0])
                    ld_summary['filtered_data']['z_max'] = max(ld_summary['filtered_data']['z_max'], filt['z_range'][1])
                
                # 异常值信息
                outlier = stats.get('outlier_info', {})
                outlier_percent = outlier.get('outlier_percent', 0)
                ld_summary['outlier_info']['avg_outlier_percent'] += outlier_percent
                ld_summary['outlier_info']['max_outlier_percent'] = max(
                    ld_summary['outlier_info']['max_outlier_percent'], outlier_percent
                )
                ld_summary['outlier_info']['min_outlier_percent'] = min(
                    ld_summary['outlier_info']['min_outlier_percent'], outlier_percent
                )
            
            # 计算平均值
            count = len(ld_stats_list)
            if count > 0:
                ld_summary['original_data']['z_mean_avg'] /= count
                ld_summary['original_data']['z_std_avg'] /= count
                ld_summary['original_data']['z_median_avg'] /= count
                ld_summary['filtered_data']['z_mean_avg'] /= count
                ld_summary['filtered_data']['z_std_avg'] /= count
                ld_summary['filtered_data']['z_median_avg'] /= count
                ld_summary['outlier_info']['avg_outlier_percent'] /= count
        
        # 基本报告信息
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_counts': {
                'atm_records': len(df_atm),
                'jpeg_records': len(df_jpeg),
                'ptm_records': len(df_ptm),
                'ld_records': len(df_ld)
            },
            'time_range': {
                'start': min(
                    df_atm['lct'].min() if not df_atm.empty else datetime.now(),
                    df_jpeg['lct'].min() if not df_jpeg.empty else datetime.now(),
                    df_ptm['lct'].min() if not df_ptm.empty else datetime.now(),
                    df_ld['lct'].min() if not df_ld.empty else datetime.now()
                ).strftime('%Y-%m-%d %H:%M:%S'),
                'end': max(
                    df_atm['lct'].max() if not df_atm.empty else datetime.now(),
                    df_jpeg['lct'].max() if not df_jpeg.empty else datetime.now(),
                    df_ptm['lct'].max() if not df_ptm.empty else datetime.now(),
                    df_ld['lct'].max() if not df_ld.empty else datetime.now()
                ).strftime('%Y-%m-%d %H:%M:%S')
            },
            'lidar_data_summary': ld_summary
        }
        
        # 保存报告
        report_path = os.path.join(self.output_base_dir, 'summary_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        logging.info(f"汇总报告已保存到: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建数据库管理器实例
    db_manager = PostgresManager(
        dbname="postgres",
        user="postgres",
        password="284490",
        host="localhost",
        port="5432"
    )
    
    try:
        # 创建数据处理器
        processor = DataProcessor(db_manager)
        
        # 处理所有数据
        processor.process_all_data()
        
        logging.info("数据处理完成!")
        
    except Exception as e:
        logging.error(f"数据处理失败: {str(e)}")
    finally:
        # 关闭数据库连接
        db_manager.close() 