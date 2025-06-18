import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QPushButton, QSizePolicy, QMessageBox,
    QButtonGroup, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem, QGraphicsRectItem, QFrame
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QIcon, QTransform
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QPointF
import numpy as np
import cv2
import json # For saving calibration
from typing import Optional, List # Import Optional and List
import logging

class ImageViewerWidget(QWidget):
    """用于显示图像、预览和进行校准的可视化窗口部件"""
    # Signals
    calibration_point_selected = pyqtSignal(str, int, int) # image_path, x, y
    calibration_reset_requested = pyqtSignal(str) # image_path
    calibration_save_requested = pyqtSignal(str, list) # image_path, points

    # 结果类型的映射关系
    RESULT_TYPE_INFO = {
        'analysis_image': {
            'order': 0,
            'display_name': '雷达分析结果',
            'description': '基于雷达数据的高度分析结果'
        },
        'traditional_default_analysis': {
            'order': 1,
            'display_name': '传统分析结果',
            'description': '基于传统图像处理的综合分析结果'
        },
        'original_debug_image': {
            'order': 2,
            'display_name': '原始图像',
            'description': '输入的原始图像'
        },
        'calibrated_debug_image': {
            'order': 3,
            'display_name': '校准后图像',
            'description': '经过透视变换校准后的图像'
        },
        'hsv_mask_debug_image': {
            'order': 4,
            'display_name': 'HSV分割掩码',
            'description': '使用HSV颜色空间进行分割的二值化结果'
        },
        'coverage_overlay_debug_image': {
            'order': 5,
            'display_name': '覆盖率叠加图',
            'description': '在原图上叠加显示草地覆盖区域'
        },
        'instance_mask_debug_image': {
            'order': 6,
            'display_name': '实例分割掩码',
            'description': '显示每个草地实例的标记结果'
        },
        'instance_overlay_debug_image': {
            'order': 7,
            'display_name': '实例叠加图',
            'description': '在原图上用不同颜色标记每个草地实例'
        },
        'density_overlay_debug_image': {
            'order': 8,
            'display_name': '密度叠加图',
            'description': '显示草地密度分布的可视化结果'
        }
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_image_path = None
        self.image_paths = []
        self.result_image_map = {} # Maps original_path -> {result_type: result_path}
        # result_type can be 'analysis_image', 'hsv_mask_debug_image', etc.
        self.loaded_calibration_points = {} # Store loaded/saved points for reuse
        self.pixmaps = {} # Cache loaded pixmaps for thumbnails
        self.current_scene = None
        self.current_pixmap_item = None
        self.calibration_points = []
        self.is_calibration_mode = False
        self.current_view_mode = 'original' # Modes: 'original', 'calibration', 'result'
        self.current_result_type_to_display = 'analysis_image' # Default result type
        self.calibration_save_dir = None # Directory to save calibration files
        
        # 添加结果数据存储
        self.analysis_data = {} # 存储每个图像的分析结果数据，格式: {image_path: {"盖度": value, "高度": value, "密度": value}}

        self._setup_ui()

    def _setup_ui(self):
        # --- 主布局 (改为 QHBoxLayout) ---
        main_layout = QHBoxLayout(self) # Layout for the whole widget
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # --- 中间: 主视图和控制 ---
        center_panel_widget = QWidget()
        center_layout = QVBoxLayout(center_panel_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)

        # --- View Mode Buttons --- #
        view_mode_layout = QHBoxLayout()
        self.view_mode_group = QButtonGroup(self)
        self.view_original_button = QPushButton("原始图像")
        self.view_original_button.setCheckable(True)
        self.view_original_button.setChecked(True)
        self.view_calibration_button = QPushButton("校准视图")
        self.view_calibration_button.setCheckable(True)
        self.view_calibration_button.setEnabled(False) # Initially disabled
        self.view_result_button = QPushButton("结果图像") # 这个按钮现在触发结果类型选择
        self.view_result_button.setCheckable(True)
        self.view_result_button.setEnabled(False) # Initially disabled

        self.view_mode_group.addButton(self.view_original_button, 0)
        self.view_mode_group.addButton(self.view_calibration_button, 1)
        self.view_mode_group.addButton(self.view_result_button, 2) # Add result button with ID 2
        self.view_mode_group.buttonClicked.connect(self._handle_button_click)

        view_mode_layout.addWidget(self.view_original_button)
        view_mode_layout.addWidget(self.view_calibration_button)
        view_mode_layout.addWidget(self.view_result_button)
        view_mode_layout.addStretch()
        center_layout.addLayout(view_mode_layout)

        # 主视图 (QGraphicsView 不变)
        self.graphics_view = QGraphicsView()
        # 设置渲染提示
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setStyleSheet("QGraphicsView { border: 1px solid #555; }" )
        
        # 设置拖动和缩放行为
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)  # 改为以视图中心为锚点
        self.graphics_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        # 设置滚动条策略
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 设置大小策略
        self.graphics_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.graphics_view)
        
        # 添加分隔线
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555555;")
        separator.setMaximumHeight(1)
        
        # 添加数据面板
        self.data_panel = QWidget()
        self.data_panel.setFixedHeight(23)  # 固定高度
        data_panel_layout = QHBoxLayout(self.data_panel)
        data_panel_layout.setContentsMargins(15, 0, 15, 0)  # 只保留左右边距
        data_panel_layout.setSpacing(30)  # 标签间距
        
        # 创建数据标签
        self.coverage_label = QLabel("盖度: N/A")
        self.height_label = QLabel("高度: N/A")
        self.density_label = QLabel("密度: N/A")
        
        # 设置标签样式
        label_style = """
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
        """
        
        for label in [self.coverage_label, self.height_label, self.density_label]:
            label.setStyleSheet(label_style)
        
        # 添加到面板布局
        data_panel_layout.addStretch(1)
        data_panel_layout.addWidget(self.coverage_label)
        data_panel_layout.addWidget(self.height_label)
        data_panel_layout.addWidget(self.density_label)
        data_panel_layout.addStretch(1)
        
        # 添加到中心布局
        center_layout.addWidget(separator)
        center_layout.addWidget(self.data_panel)

        # 校准控制按钮 (不变)
        self.calibration_controls_widget = QWidget()
        calibration_controls_layout = QHBoxLayout(self.calibration_controls_widget)
        calibration_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_button = QPushButton("重置点")
        self.reset_button.clicked.connect(self._reset_calibration_points)
        self.save_button = QPushButton("保存校准")
        self.save_button.clicked.connect(self._save_calibration)
        calibration_controls_layout.addWidget(self.reset_button)
        calibration_controls_layout.addWidget(self.save_button)
        center_layout.addWidget(self.calibration_controls_widget)
        self.calibration_controls_widget.setVisible(False)

        # --- 右侧: 预览列表 ---
        self.preview_list = QListWidget()
        preview_list_width = 120 # 稍微加宽一点
        icon_width = preview_list_width # 减去一些边距/填充
        icon_height = int(icon_width * 0.75) # 保持比例
        self.preview_list.setIconSize(QSize(icon_width, icon_height))
        self.preview_list.setFixedWidth(preview_list_width)
        # Add border-radius and padding for items
        self.preview_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #555;
                border-radius: 5px; /* Add border-radius */
            }
            QListWidget::item {
                padding: 5px 0px; /* Adjust padding (top/bottom 5px, left/right 0) */
                margin: 2px; /* Add small margin between items */
                text-align: center; /* Center text (might not affect icon) */
                /* Attempting icon centering - might require more advanced methods */
                background-color: transparent; /* Ensure background doesn't interfere */
            }
        """)
        self.preview_list.setViewMode(QListWidget.ViewMode.IconMode) # 改回 IconMode
        self.preview_list.setFlow(QListWidget.Flow.TopToBottom) # 垂直排列
        self.preview_list.setWrapping(False) # 禁止换行
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setMovement(QListWidget.Movement.Static)
        self.preview_list.currentItemChanged.connect(self._on_preview_selected)

        # --- 将 Center Panel 和 Preview List 添加到主布局 ---
        main_layout.addWidget(center_panel_widget, 1) # 主视图占大部分空间
        main_layout.addWidget(self.preview_list, 0) # 预览列表占较少空间

        # 设置鼠标事件过滤器
        self.graphics_view.viewport().installEventFilter(self)

    def load_images(self, paths: list[str]):
        """加载图像路径列表并生成预览图 (接收一个路径列表或单个路径字符串)"""
        if isinstance(paths, str):
            paths = [paths]
        elif not isinstance(paths, list):
             paths = []

        # 保存当前视图模式
        current_view_mode = self.current_view_mode
        
        self.image_paths = paths
        self.current_image_path = self.image_paths[0] if self.image_paths else None
        
        # 更新预览列表，但保持当前视图模式
        if current_view_mode == 'original' or current_view_mode == 'calibration':
            self._populate_preview_list('original')
        elif current_view_mode == 'result' and self.current_image_path:
            # 如果当前是结果视图模式，检查是否有结果数据
            result_path_dict = self.result_image_map.get(self.current_image_path, {})
            if result_path_dict:
                self._populate_preview_list('result', result_path_dict)
            else:
                # 如果没有结果数据，则切换回原始视图模式
                current_view_mode = 'original'
                self._populate_preview_list('original')

        if self.image_paths:
            self.display_image(self.image_paths[0])
            
            # 根据保存的视图模式设置对应按钮的选中状态
            if current_view_mode == 'original':
                self.view_original_button.setChecked(True)
            elif current_view_mode == 'calibration':
                self.view_calibration_button.setChecked(True)
            elif current_view_mode == 'result':
                self.view_result_button.setChecked(True)
        else:
             print("[Viewer] No valid images loaded, view cleared.")

    def display_image(self, image_path: str):
        """显示指定路径的图像，并自动从analyzed_files.json读取结果"""
        if not image_path or not os.path.exists(image_path):
            logging.error(f"[Viewer] 尝试显示不存在的图像: {image_path}")
            self._show_placeholder_scene("图像不存在或路径无效")
            return

        # 记录当前图像路径
        self.current_image_path = image_path

        # 加载并显示图像
        success = self._load_and_display_base_image(image_path)
        if not success:
            self._show_placeholder_scene(f"无法加载图像:\n{os.path.basename(image_path)}")
            return

        # 检查当前视图模式
        if self.current_view_mode == 'calibration':
            # 如果当前是校准模式，进入校准逻辑
            self._enter_calibration_logic(image_path)
        else:
            # 如果当前不是校准模式，确保校准模式已退出
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        
        # 新增：自动从analyzed_files.json读取结果
        try:
            from app.core.analyzed_files_tracker import AnalyzedFilesTracker
            tracker = AnalyzedFilesTracker()
            file_name = os.path.basename(image_path)
            analyzed = tracker.analyzed_files.get(file_name, {})
            results = analyzed.get('results', None)
            if results:
                self.set_result_data(image_path, results)
        except Exception as e:
            logging.error(f"[Viewer] 读取analyzed_files.json失败: {e}")

        # 更新数据面板显示
        self._update_data_panel()

    def _load_and_display_base_image(self, image_path: str) -> bool:
        """
        加载图像并在视图中显示
        
        参数:
            image_path: 图像路径
            
        返回:
            bool: 是否成功加载图像
        """
        if not os.path.exists(image_path):
            logging.error(f"[Viewer] 图像文件不存在: {image_path}")
            return False

        try:
            # 使用numpy和OpenCV读取图像，以支持非ASCII路径
            n = np.fromfile(image_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logging.error(f"[Viewer] 无法解码图像: {image_path}")
                return False
                
            # 转换为RGB（OpenCV默认为BGR）
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 创建QImage和QPixmap
            height, width, channel = img_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 创建场景和图元
            scene = QGraphicsScene()
            pixmap_item = scene.addPixmap(pixmap)
            pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            
            # 设置场景和视图
            self.graphics_view.setScene(scene)
            self.current_scene = scene
            self.current_pixmap_item = pixmap_item
            
            # 重置变换
            self.graphics_view.resetTransform()
            
            # 调整视图以适应图像
            self.graphics_view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            
            # 更新按钮状态
            self.view_calibration_button.setEnabled(True)
            result_dict = self.result_image_map.get(image_path, {})
            has_image_paths = any(isinstance(v, str) for k, v in result_dict.items() if k != 'calibration_points')
            self.view_result_button.setEnabled(has_image_paths)
            
            return True
        except Exception as e:
            logging.error(f"[Viewer] 加载图像时出错: {e}")
            return False

    def set_calibration_save_dir(self, directory: str):
        """设置保存校准文件的目录"""
        self.calibration_save_dir = directory

    def _enter_calibration_logic(self, image_path: str):
        """处理进入校准模式的逻辑"""
        # 确保图像已加载
        if not self.current_pixmap_item or self.current_image_path != image_path:
            # 直接加载图像，避免递归调用display_image
            success = self._load_and_display_base_image(image_path)
            if not success:
                logging.error(f"[Viewer] 无法加载校准图像: {image_path}")
                QMessageBox.critical(self, "错误", f"无法加载图像进行校准: {os.path.basename(image_path)}")
                return
            self.current_image_path = image_path

        # 设置校准模式状态
        self.is_calibration_mode = True
        self.calibration_controls_widget.setVisible(True)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)

        # 重置校准点
        self.calibration_points = []

        # 查找已保存的校准点
        if image_path in self.loaded_calibration_points:
            logging.info(f"[Viewer] 找到已保存的校准点: {self.loaded_calibration_points[image_path]}")
            self.calibration_points = self.loaded_calibration_points[image_path].copy()
            self._redraw_calibration_points()
        else:
            logging.info(f"[Viewer] 未找到已保存的校准点，开始新的校准")

        # 更新数据面板
        self._update_data_panel()

    def _on_preview_selected(self, current_item, previous_item):
        """处理预览列表项选择变化事件"""
        if not current_item:
            return

        # 获取选中项的数据
        selected_data = current_item.data(Qt.ItemDataRole.UserRole)
        if not selected_data:
            logging.warning("[Viewer _on_preview_selected] Selected item has no data.")
            return

        # 根据当前视图模式处理选择
        if self.current_view_mode in ['original', 'calibration']:
            if isinstance(selected_data, str):
                original_path = selected_data
                # --- FIX: Avoid reloading if already displayed --- #
                if original_path != self.current_image_path:
                    logging.info(f"[Viewer _on_preview_selected] Original/Calib mode: Selected new image: {os.path.basename(original_path)}")
                    self.display_image(original_path)
                    # 确保在切换图像时更新数据面板
                    self._update_data_panel()
                else:
                    logging.debug(f"[Viewer _on_preview_selected] Original/Calib mode: Selected image ({os.path.basename(original_path)}) is already displayed. Doing nothing.")
                # --- END FIX --- #
            else:
                 logging.warning(f"[Viewer] Error: Expected string path in original/calibration mode, got {type(selected_data)}")

        elif self.current_view_mode == 'result':
            if isinstance(selected_data, tuple) and len(selected_data) == 2:
                original_path, result_type = selected_data
                logging.info(f"[Viewer _on_preview_selected] Result mode: Selected {result_type}. Displaying...")
                self._display_result_image(original_path, result_type)
                # 显示结果时，数据面板也应该显示对应原始图像的分析结果
                self.current_image_path = original_path
                self._update_data_panel()
                # --- END FIX ---
            else:
                 logging.warning(f"[Viewer] Error: Expected tuple (type, path) in result mode, got {type(selected_data)}")

        else:
            logging.warning(f"[Viewer _on_preview_selected] Unknown view mode: {self.current_view_mode}")

    def _set_view_mode(self, mode_id: int, target_image_path: Optional[str] = None):
        """根据按钮点击设置视图模式。 0: original, 1: calibration, 2: result"""
        # Determine the effective image path for context (if needed)
        effective_image_path = target_image_path if target_image_path else self.current_image_path
        
        # 根据视图模式控制数据面板的显示
        self.data_panel.setVisible(mode_id in [0, 2])  # 只在原始图像(0)和结果图像(2)视图显示
        
        # 如果提供了目标图片路径，先确保它在图片列表中
        if target_image_path and target_image_path not in self.image_paths:
            logging.info(f"[Viewer] 添加目标图片到图片列表: {os.path.basename(target_image_path)}")
            self.image_paths.append(target_image_path)

        # Allow switching to original even if no image path exists
        if not effective_image_path and mode_id != 0:
            # If no image context, only allow switching back to (empty) original view
            if mode_id == 0:
                 logging.info("[Viewer _set_view_mode] No image loaded, switching to empty original view.")
                 self.current_view_mode = 'original'
                 self.is_calibration_mode = False
                 self.calibration_controls_widget.setVisible(False)
                 self.preview_list.clear()
                 self.graphics_view.setScene(None)
                 self.current_image_path = None # Ensure path is None
                 self.current_pixmap_item = None
                 self.view_original_button.setChecked(True)
                 self.view_calibration_button.setEnabled(False)
                 self.view_result_button.setEnabled(False)
            else:
                 # Prevent switching to calibration/result if no image context
                 logging.info(f"[Viewer _set_view_mode] Prevented switching to mode {mode_id} because no image context exists.")
                 # Re-check the button corresponding to the actual current mode
                 if self.current_view_mode == 'original': self.view_original_button.setChecked(True)
                 elif self.current_view_mode == 'calibration': self.view_calibration_button.setChecked(True)
                 elif self.current_view_mode == 'result': self.view_result_button.setChecked(True)
                 return # Stop processing the mode switch

        # --- Handle Mode Switching --- #
        if mode_id == 0: # Original
            logging.info("[Viewer _set_view_mode] Switching to Original mode.")

            # --- FIX 1 & 2: Final attempt at stable switching --- #
            # 1. Set mode state and UI elements first
            self.current_view_mode = 'original'
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self.view_original_button.setChecked(True)

            # 2. Ensure the target image data is loaded if path is valid
            image_loaded_successfully = False
            if effective_image_path:
                 # 直接调用_load_and_display_base_image，避免递归调用display_image
                 logging.info(f"[Viewer _set_view_mode] 加载原始图像: {os.path.basename(effective_image_path)}")
                 success = self._load_and_display_base_image(effective_image_path)
                 if success:
                     self.current_image_path = effective_image_path
                     image_loaded_successfully = True
                     # 更新数据面板
                     self._update_data_panel()
            else:
                 logging.info("[Viewer _set_view_mode] No effective image path for Original mode. Clearing graphics view.")
                 self.current_image_path = None
                 self.current_pixmap_item = None
                 self.graphics_view.setScene(None) # Explicitly clear scene

            # 3. Populate the preview list AFTER display attempt
            self._populate_preview_list('original')

            # 4. Select the correct item in the list (without blocking signals initially)
            selected_row = -1
            if self.current_image_path:
                for i in range(self.preview_list.count()):
                    item = self.preview_list.item(i)
                    if item and item.data(Qt.ItemDataRole.UserRole) == self.current_image_path:
                        logging.debug(f"[Viewer _set_view_mode] Found item for {os.path.basename(self.current_image_path)} at row {i}. Setting current row.")
                        self.preview_list.setCurrentRow(i) # Let signals fire
                        selected_row = i
                        break
                if selected_row == -1:
                    logging.warning("[Viewer _set_view_mode] Warning: Could not find list item for current image path after populating.")
            else:
                 self.preview_list.setCurrentRow(-1) # Ensure no selection if no image

            # 5. Final check: If image loading failed or path was invalid, show placeholder
            if not image_loaded_successfully and effective_image_path:
                self._show_placeholder_scene(f"无法加载原始图像:\n{os.path.basename(effective_image_path)}")
            elif not effective_image_path:
                self._show_placeholder_scene("无图像加载")
            # --- End FIX 1 & 2 ---

        elif mode_id == 1: # Calibration
            logging.info("[Viewer _set_view_mode] Switching to Calibration mode.")
            path_to_calibrate = target_image_path if target_image_path else self.current_image_path

            if not path_to_calibrate:
                logging.warning("[Viewer _set_view_mode] Cannot enter Calibration mode: No image specified or loaded.")
                # Switch back to original view
                self._set_view_mode(0)
                return

            # 确保目标图片存在
            if not os.path.exists(path_to_calibrate):
                logging.error(f"[Viewer] 校准目标图片不存在: {path_to_calibrate}")
                QMessageBox.critical(self, "错误", f"无法找到图片: {os.path.basename(path_to_calibrate)}")
                return

            # Set UI state for calibration *before* calling the logic function
            self.current_view_mode = 'calibration'
            self.view_calibration_button.setChecked(True)
            self._populate_preview_list('original') # Calibration view uses original previews

            # 确保当前图片在预览列表中被选中
            for i in range(self.preview_list.count()):
                item = self.preview_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == path_to_calibrate:
                    self.preview_list.setCurrentRow(i)
                    break

            # Call the function that handles the actual calibration entry logic
            self._enter_calibration_logic(path_to_calibrate)

        elif mode_id == 2: # Result
            logging.info("[Viewer _set_view_mode] Switching to Result mode.")
            if not self.current_image_path:
                logging.warning("[Viewer _set_view_mode] No current image path for Result mode.")
                return # 应该在前面的检查中处理

            result_path_dict = self.result_image_map.get(self.current_image_path, {})
            if not result_path_dict:
                logging.warning("[Viewer _set_view_mode] No results available for the current image.")
                QMessageBox.information(self, "无结果", "当前图像没有可用的分析结果。")
                # Switch back to original view if no results
                self._set_view_mode(0)
                return

            # If results exist, switch mode and update UI
            self.current_view_mode = 'result'
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

            # Populate preview list with result images
            self._populate_preview_list('result', result_path_dict)

            # 确保结果视图按钮被选中
            self.view_result_button.setChecked(True)
            
            # 如果预览列表中有项目，选择第一个并显示对应的结果图像
            if self.preview_list.count() > 0:
                first_item = self.preview_list.item(0)
                if first_item:
                    self.preview_list.setCurrentItem(first_item)
                    result_data = first_item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(result_data, tuple) and len(result_data) == 2:
                        original_path, result_type = result_data
                        self._display_result_image(original_path, result_type)
                        # 更新数据面板显示当前图像的分析结果
                        self._update_data_panel()

    def _populate_preview_list(self, mode: str, data: Optional[dict] = None):
        """Clears and populates the preview list based on the view mode."""
        self.preview_list.clear()
        self.pixmaps.clear() # Clear pixmap cache when changing modes

        fm = self.preview_list.fontMetrics()
        icon_width = self.preview_list.iconSize().width()
        available_width = icon_width - 10
        if available_width < 20:
             available_width = 20

        if mode == 'original':
            logging.debug("[Viewer _populate_preview_list] Populating with ORIGINAL images.")
            items_added = 0
            for path in self.image_paths:
                 pixmap = self._load_thumbnail(path)
                 if pixmap:
                     base_filename = os.path.basename(path)
                     text_width = fm.boundingRect(base_filename).width()
                     if text_width > available_width:
                         display_text = fm.elidedText(base_filename, Qt.TextElideMode.ElideRight, available_width)
                     else:
                         display_text = base_filename
                     item = QListWidgetItem(QIcon(pixmap), display_text)
                     item.setData(Qt.ItemDataRole.UserRole, path)
                     item.setToolTip(path)
                     self.preview_list.addItem(item)
                     items_added += 1

        elif mode == 'result':
            logging.debug("[Viewer _populate_preview_list] Populating with RESULT images.")
            if not data or not isinstance(data, dict):
                logging.warning("[Viewer _populate_preview_list] No result data provided.")
                return

            items_added = 0
            # 使用RESULT_TYPE_INFO中定义的顺序和显示名称
            sorted_results = []
            for result_type, result_path in data.items():
                if not isinstance(result_path, str) or result_type == 'calibration_points':
                    continue
                
                # 获取结果类型信息
                type_info = self.RESULT_TYPE_INFO.get(result_type, {
                    'order': 999,
                    'display_name': result_type,
                    'description': '未知结果类型'
                })
                
                sorted_results.append((
                    type_info['order'],
                    result_type,
                    type_info['display_name'],
                    type_info['description'],
                    result_path
                ))
            
            # 按order排序，确保结果图像按预定义顺序显示
            sorted_results.sort(key=lambda x: x[0])  # 按order排序

            for _, result_type, display_name, description, result_path in sorted_results:
                if not os.path.exists(result_path):
                    logging.debug(f"[Viewer] 跳过无效的结果路径 '{result_type}': {result_path}")
                    continue
                    
                pixmap = self._load_thumbnail(result_path)
                if pixmap:
                    # 使用友好的显示名称
                    text_width = fm.boundingRect(display_name).width()
                    if text_width > available_width:
                        display_text = fm.elidedText(display_name, Qt.TextElideMode.ElideRight, available_width)
                    else:
                        display_text = display_name
                        
                    item = QListWidgetItem(QIcon(pixmap), display_text)
                    # 存储原始图像路径和结果类型
                    item.setData(Qt.ItemDataRole.UserRole, (self.current_image_path, result_type))
                    # 设置完整的工具提示，包含描述
                    item.setToolTip(f"{display_name}\n{description}\n\n路径: {result_path}")
                    self.preview_list.addItem(item)
                    items_added += 1
                    logging.debug(f"[Viewer] 添加结果项: {display_name} ({result_type})")

            # 如果添加了项目，默认选择第一个
            if items_added > 0:
                logging.debug(f"[Viewer] 结果模式添加了 {items_added} 个项目，选择第一个")
                self.preview_list.setCurrentRow(0)
            else:
                logging.warning("[Viewer] 结果模式没有添加任何项目")
        else:
             logging.warning(f"[Viewer _populate_preview_list] Unknown mode: {mode}")

    def _load_thumbnail(self, path: str) -> Optional[QPixmap]:
        """Loads or retrieves cached thumbnail pixmap for a given path."""
        if path in self.pixmaps:
             return self.pixmaps[path]
        try:
            n = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img_bgr is None:
                 logging.warning(f"[Viewer] Failed to decode thumbnail image {path}")
                 return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.preview_list.iconSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.pixmaps[path] = scaled_pixmap # Cache it
            return scaled_pixmap
        except Exception as e:
            logging.error(f"[Viewer] Error loading thumbnail {path}: {e}")
            return None

    def _show_result_selector(self):
        """显示结果图像类型选择 (如果需要) - This function is now OBSOLETE and integrated into _set_view_mode(2)"""
        logging.info("[Viewer _show_result_selector] This function is obsolete.")
        # Kept for reference, but should not be called anymore
        pass
        # if not self.current_image_path or not self.result_image_map.get(self.current_image_path):
        # ... (rest of old logic) ...

    def eventFilter(self, source, event):
        """事件过滤器，用于捕获图形视图上的鼠标点击和移动以进行交互和活跃检测"""
        # 鼠标移动时，通知主窗口重置自动切换计时器
        if source == self.graphics_view.viewport() and event.type() == event.Type.MouseMove:
            # 尝试调用主窗口的reset_auto_switch_timer
            main_window = self.window()
            if hasattr(main_window, 'reset_auto_switch_timer'):
                main_window.reset_auto_switch_timer()
        # 保持原有点击逻辑
        if source == self.graphics_view.viewport() and event.type() == event.Type.MouseButtonPress:
            if self.is_calibration_mode and event.button() == Qt.MouseButton.LeftButton:
                # Allow adding points only if less than 4
                if len(self.calibration_points) < 4:
                    # 将视图坐标转换为场景坐标
                    scene_pos = self.graphics_view.mapToScene(event.pos())
                    # 确保点击在图像内部
                    if self.current_pixmap_item and self.current_pixmap_item.sceneBoundingRect().contains(scene_pos):
                        # 添加点（相对于 pixmap item 的坐标）
                        item_pos = self.current_pixmap_item.mapFromScene(scene_pos).toPoint()
                        self.calibration_points.append((item_pos.x(), item_pos.y()))
                        logging.info(f"[Viewer] Calibration point {len(self.calibration_points)} added: ({item_pos.x()}, {item_pos.y()})")
                        self._redraw_calibration_points()
                        # Ensure cursor stays as crosshair after adding point
                        self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)
                        # self.calibration_point_selected.emit(self.current_image_path, item_pos.x(), item_pos.y())
                        return True # 事件已处理
                else:
                    logging.info("[Viewer] Max 4 calibration points reached.")
                    QMessageBox.information(self, "提示", "已选择4个点。请点击 '保存校准' 或 '重置点'。")
                    return True # Prevent other actions while max points reached

        return super().eventFilter(source, event)

    def _redraw_calibration_points(self):
        """在场景中重新绘制校准点和连线"""
        # First, ensure the base image is visible if we are in calibration mode
        # REMOVED: Reloading block - Drawing depends on image being loaded beforehand.
 
        if not self.current_scene or not self.current_pixmap_item:
            # Clear scene if no pixmap item (e.g., image failed to load)
            if self.current_scene:
                 # Remove only calibration artifacts
                 for item in self.current_scene.items():
                            self.current_scene.removeItem(item)
            logging.debug("[Viewer _redraw_calibration_points] Cannot draw points: No scene or no pixmap item.")
            return
 
        # Remove old points and lines first
        items_to_remove = []
        for item in self.current_scene.items():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem)):
                # Keep the main pixmap item
                if item != self.current_pixmap_item:
                    items_to_remove.append(item)
        for item in items_to_remove:
             self.current_scene.removeItem(item)
 
        # Draw only if in calibration mode and pixmap exists and points exist
        if not self.is_calibration_mode or not self.calibration_points:
             # Clear points visually if not in calibration mode or no points exist
             logging.debug("[Viewer _redraw_calibration_points] Not in calibration mode or no points, clearing visual points.")
             # The removal of old points/lines above handles the clearing
             return
 
        # Check pixmap item again just before drawing
        if not self.current_pixmap_item or not self.current_pixmap_item.scene():
             logging.warning("[Viewer _redraw_calibration_points] Cannot draw points: pixmap item missing or not in scene before drawing.")
             return
 
        # 获取图像尺寸，用于计算自适应大小
        pixmap_rect = self.current_pixmap_item.boundingRect()
        image_width = pixmap_rect.width()
        image_height = pixmap_rect.height()
        
        # 计算图像对角线长度，用作比例参考
        image_diagonal = np.sqrt(image_width**2 + image_height**2)
        
        # 根据图像大小自适应计算点和线的大小
        # 对于小图像，保持最小尺寸；对于大图像，按比例增大但有上限
        base_point_radius = 3  # 基础点半径
        base_line_width = 2    # 基础线宽度
        
        # 计算自适应尺寸，使用对角线长度作为参考
        # 使用对数比例，避免大图像时尺寸过大
        size_factor = 0.8 + 0.7 * np.log10(max(1, image_diagonal / 500))  # 增加基础系数和比例系数
        size_factor = min(size_factor, 100.0)  # 增加最大缩放因子上限从3.0到10.0
        
        point_radius = max(8, int(base_point_radius * size_factor))
        line_width = max(3, int(base_line_width * size_factor))
        font_size = max(10, int(12 * size_factor))  # 增加字体基础大小从10到12
        
        # 根据点的大小调整标签偏移
        label_offset_x = point_radius + 2
        label_offset_y = -point_radius - 2
        
        # 绘制新的点和线
        point_pen = QPen(QColor(255, 0, 0), max(1, line_width/2))  # 红色点边框
        point_brush = QColor(255, 0, 0, 200)  # 半透明红色填充
        line_pen = QPen(QColor(255, 0, 0, 230), line_width)  # 红色线

        # Ensure points are within bounds (precautionary)

        scene_points = []
        for i, pt in enumerate(self.calibration_points):
            # Clamp point coordinates to be within the pixmap bounds
            # Check if pt is valid tuple/list
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                 logging.warning(f"[Viewer] Invalid point format skipped: {pt}")
                 continue
            clamped_x = max(0, min(pt[0], pixmap_rect.width()))
            clamped_y = max(0, min(pt[1], pixmap_rect.height()))
            item_point = QPointF(clamped_x, clamped_y)

            # 将点坐标从 item 坐标转换回场景坐标
            scene_pt = self.current_pixmap_item.mapToScene(item_point)
            scene_points.append(scene_pt) # Store scene points for lines

            # 绘制点
            ellipse = QGraphicsEllipseItem(scene_pt.x() - point_radius, scene_pt.y() - point_radius,
                                           point_radius * 2, point_radius * 2)
            ellipse.setPen(point_pen)
            ellipse.setBrush(point_brush)
            self.current_scene.addItem(ellipse)

            # 绘制序号
            text = QGraphicsSimpleTextItem(str(i + 1))
            text.setPos(scene_pt.x() + label_offset_x, scene_pt.y() + label_offset_y)
            text.setBrush(QColor("red"))
            font = text.font()
            font.setPointSize(font_size) # 使用自适应字体大小
            font.setBold(True)
            text.setFont(font)
            self.current_scene.addItem(text)

        # 绘制连线 (use stored scene_points)
        for i in range(len(scene_points)):
            if i > 0:
                line = QGraphicsLineItem(scene_points[i-1].x(), scene_points[i-1].y(), scene_points[i].x(), scene_points[i].y())
                line.setPen(line_pen)
                self.current_scene.addItem(line)

        # 绘制闭合线
        if len(scene_points) == 4:
            line = QGraphicsLineItem(scene_points[-1].x(), scene_points[-1].y(), scene_points[0].x(), scene_points[0].y())
            line.setPen(line_pen)
            self.current_scene.addItem(line)

        logging.debug(f"[Viewer _redraw_calibration_points] Drawing {len(self.calibration_points)} points with radius {point_radius} and line width {line_width}.")

    def _reset_calibration_points(self):
        """重置当前图像的校准点"""
        if self.is_calibration_mode:
            logging.info(f"[Viewer] Resetting calibration points for {os.path.basename(self.current_image_path)}")
            self.calibration_points = []
            # Also clear from loaded dict if resetting
            if self.current_image_path in self.loaded_calibration_points:
                del self.loaded_calibration_points[self.current_image_path]
            self._redraw_calibration_points()
            # self.calibration_reset_requested.emit(self.current_image_path)

    def _save_calibration(self):
        """保存当前校准点并发出信号"""
        if self.is_calibration_mode and len(self.calibration_points) == 4:
            logging.info(f"[Viewer] Saving calibration points for {os.path.basename(self.current_image_path)}: {self.calibration_points}")
            # Save to loaded points dict as well
            self.loaded_calibration_points[self.current_image_path] = self.calibration_points

            # 确保点是整数
            int_points = [[int(round(p[0])), int(round(p[1]))] for p in self.calibration_points]
            
            # 发出信号，传递图像路径和点列表
            self.calibration_save_requested.emit(self.current_image_path, int_points)
            
            # 不再在这里直接保存文件，由MainWindow统一处理
            # 校准模式的退出由MainWindow在处理完保存后调用exit_calibration_mode方法

        elif self.is_calibration_mode:
            logging.info("[Viewer] Error: Need exactly 4 points to save calibration.")
            QMessageBox.warning(self, "校准错误", "需要选择4个角点才能保存。")

    def exit_calibration_mode(self):
        """退出校准模式"""
        logging.info("[Viewer] Exiting calibration mode.")
        self.is_calibration_mode = False
        self.calibration_controls_widget.setVisible(False)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        
        # 不再调用_set_view_mode，避免递归调用
        # self._set_view_mode(0) # 这会导致递归调用

    def _display_result_image(self, original_image_path: str, result_type: str):
        """显示指定类型的结果图像"""
        if not original_image_path or not result_type:
            logging.error("[Viewer] 无法显示结果图像：缺少原始图像路径或结果类型")
            return

        result_path_dict = self.result_image_map.get(original_image_path, {})
        if not result_path_dict or result_type not in result_path_dict:
            logging.error(f"[Viewer] 无法找到结果图像：{result_type}")
            return

        result_path = result_path_dict[result_type]
        if not os.path.exists(result_path):
            logging.error(f"[Viewer] 结果图像文件不存在：{result_path}")
            return

        try:
            # 使用numpy和OpenCV读取图像，以支持非ASCII路径
            n = np.fromfile(result_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logging.error(f"[Viewer] 无法解码结果图像: {result_path}")
                return
                
            # 转换为RGB（OpenCV默认为BGR）
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 创建QImage和QPixmap
            height, width, channel = img_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 创建场景和图元
            scene = QGraphicsScene()
            pixmap_item = scene.addPixmap(pixmap)
            pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            
            # 设置场景和视图
            self.graphics_view.setScene(scene)
            self.current_scene = scene
            self.current_pixmap_item = pixmap_item
            
            # 重置变换
            self.graphics_view.resetTransform()
            
            # 调整视图以适应图像
            self.graphics_view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            
            # 更新当前图像路径，确保数据面板显示正确的分析结果
            self.current_image_path = original_image_path
            
            # 更新数据面板
            self._update_data_panel()
            
            logging.info(f"[Viewer] 已显示结果图像: {result_type}")
            
        except Exception as e:
            logging.error(f"[Viewer] 显示结果图像时出错: {e}")
            self._show_placeholder_scene(f"无法加载结果图像:\n{os.path.basename(result_path)}")

    def _show_placeholder_scene(self, message: str):
         """Displays a placeholder message in the graphics view."""
         temp_scene = QGraphicsScene()
         text_item = QGraphicsSimpleTextItem(message)
         font = text_item.font()
         font.setPointSize(14)
         text_item.setFont(font)
         text_item.setBrush(QColor("white"))
         rect = text_item.boundingRect()
         text_item.setPos(-rect.width()/2, -rect.height()/2)
         temp_scene.addItem(text_item)
         self.graphics_view.setScene(temp_scene)
         self.current_pixmap_item = None # Ensure no pixmap item is active

    def set_result_data(self, original_path: str, results: Optional[dict]):
        """设置指定原始图像的结果数据"""
        if not original_path or not results:
            logging.warning("[Viewer] 设置结果数据失败：路径或结果为空")
            return

        # 确保原始图像在列表中
        if original_path not in self.image_paths:
            self.image_paths.append(original_path)

        # 更新结果映射
        self.result_image_map[original_path] = results

        # 提取分析数据（盖度、密度、高度等）
        analysis_data = {}
        for key, value in results.items():
            if key in ['草地盖度', '草地密度', '草地高度']:
                # 处理盖度数据
                if key == '草地盖度':
                    if isinstance(value, str) and '%' in value:
                        try:
                            analysis_data['盖度'] = float(value.replace('%', ''))
                        except ValueError:
                            analysis_data['盖度'] = value
                    else:
                        analysis_data['盖度'] = value
                # 处理高度数据
                elif key == '草地高度':
                    # 只处理形如 'xx.xmm' 的字符串，直接提取数值，单位为mm
                    if isinstance(value, str) and value.endswith('mm'):
                        try:
                            height_mm = float(value.replace('mm', ''))
                            analysis_data['高度'] = height_mm
                        except ValueError:
                            analysis_data['高度'] = value
                    else:
                        analysis_data['高度'] = value
                # 处理密度数据
                elif key == '草地密度':
                    if isinstance(value, str) and '株/平方米' in value:
                        try:
                            analysis_data['密度'] = float(value.split(' ')[0])
                        except (ValueError, IndexError):
                            analysis_data['密度'] = value
                    else:
                        analysis_data['密度'] = value
            elif key.endswith('_image') or key.endswith('_path'):
                continue
            else:
                # 保留其他非图像路径数据
                analysis_data[key] = value

        # 更新分析数据
        if analysis_data:
            self.analysis_data[original_path] = analysis_data
            logging.info(f"[Viewer] 已更新分析数据: {analysis_data}")

        # 如果当前显示的是这个图像，更新数据面板
        if self.current_image_path == original_path:
            self._update_data_panel()

        # 启用结果查看按钮
        self.view_result_button.setEnabled(True)

        logging.info(f"[Viewer] 已设置结果数据: {os.path.basename(original_path)}")

    def clear_results(self, clear_images=True):
        """清除所有存储的结果映射"""
        self.result_image_map.clear()
        self.analysis_data.clear()  # 清除分析数据
        
        # 如果需要清除图像列表
        if clear_images:
            self.image_paths = []
            self.current_image_path = None
            self.preview_list.clear()
            
        # If an image is currently displayed, disable its result button
        if self.current_image_path:
            self.view_result_button.setEnabled(False)
            # 更新数据面板（清空数据）
        self._update_data_panel()
        logging.info("[Viewer] All result maps cleared.")
        
    def add_image_to_preview(self, image_path: str, max_history: int = 10):
        """
        将图像添加到预览列表，限制保留的历史图像数量
        
        参数:
            image_path: 图像路径
            max_history: 最大保留的历史图像数量
        """
        # 检查图像是否已在列表中
        for i in range(self.preview_list.count()):
            item = self.preview_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                # 如果已存在，将其移到列表顶部
                self.preview_list.takeItem(i)
                self.preview_list.insertItem(0, item)
                self.preview_list.setCurrentItem(item)
                return
                
        # 如果不在列表中，添加新项
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
            
        # 加载缩略图
        thumbnail = self._load_thumbnail(image_path)
        if thumbnail:
            item = QListWidgetItem()
            item.setIcon(QIcon(thumbnail))
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.preview_list.insertItem(0, item)
            self.preview_list.setCurrentItem(item)
            
            # 如果超过最大历史数量，移除最旧的项
            while self.preview_list.count() > max_history:
                self.preview_list.takeItem(self.preview_list.count() - 1)
                
    def has_result_for_image(self, image_path: str) -> bool:
        """
        检查指定图像是否有分析结果
        
        参数:
            image_path: 图像路径
            
        返回:
            bool: 是否有结果
        """
        return image_path in self.result_image_map and bool(self.result_image_map[image_path])

    def get_calibration_points(self) -> Optional[List[List[int]]]:
        """获取当前图像已验证(4点)并转换为整数的校准点"""
        current_points = None
        if self.is_calibration_mode and len(self.calibration_points) == 4:
            current_points = self.calibration_points
        elif not self.is_calibration_mode and self.current_image_path in self.loaded_calibration_points:
            # Allow retrieving previously saved/loaded points even if not in calibration mode
            loaded = self.loaded_calibration_points[self.current_image_path]
            if isinstance(loaded, list) and len(loaded) == 4:
                current_points = loaded

        if current_points:
            # Ensure points are valid format before conversion
            try:
                return [[int(round(p[0])), int(round(p[1]))] for p in current_points]
            except (TypeError, IndexError) as e:
                logging.error(f"[Viewer] Error converting calibration points to int: {e}, points: {current_points}")
                return None
        return None

    def _handle_button_click(self, button):
        """Receives the clicked button object and calls _set_view_mode with its ID."""
        mode_id = self.view_mode_group.id(button)
        logging.debug(f"[Viewer _handle_button_click] Button '{button.text()}' clicked, ID: {mode_id}")
        if mode_id != -1: # QButtonGroup returns -1 if button not found or has no ID
            self._set_view_mode(mode_id)
        else:
             logging.warning("[Viewer _handle_button_click] Warning: Clicked button has no valid ID in the group.")

    def wheelEvent(self, event):
        """处理鼠标滚轮事件，实现图像缩放和自动居中"""
        # 在校准模式下不允许缩放
        if self.is_calibration_mode:
            super().wheelEvent(event)
            return

        # 检查是否有图像加载
        if not self.current_pixmap_item or not self.current_scene:
            super().wheelEvent(event)
            return

        # 按住Ctrl键时才进行滚动，否则进行缩放
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return

        # 获取当前视图和场景的几何信息
        view_rect = self.graphics_view.viewport().rect()
        scene_rect = self.current_scene.sceneRect()

        # 缩放因子
        zoom_factor = 1.15  # 15% 缩放步长
        
        # 获取鼠标位置和对应的场景位置
        mouse_pos = event.position().toPoint()
        old_pos = self.graphics_view.mapToScene(mouse_pos)
        
        # 根据滚轮方向确定缩放方向
        angle = event.angleDelta().y()
        if angle > 0:
            scale_factor = zoom_factor  # 放大
        else:
            scale_factor = 1.0 / zoom_factor  # 缩小

        # 获取当前变换
        current_transform = self.graphics_view.transform()
        new_transform = current_transform.scale(scale_factor, scale_factor)
        
        # 计算缩放后的场景矩形
        mapped_rect = new_transform.mapRect(scene_rect)
        
        # 检查缩放是否在合理范围内（防止过分缩小或放大）
        view_width = view_rect.width()
        view_height = view_rect.height()
        
        # 限制最小缩放：图像至少占视图的20%
        min_scale = max(view_width * 0.2 / scene_rect.width(),
                       view_height * 0.2 / scene_rect.height())
        # 限制最大缩放：图像最大放大到原始尺寸的5倍
        max_scale = 5.0
        
        new_scale = new_transform.m11()  # 获取新的缩放比例
        
        # 如果缩放在合理范围内，应用缩放
        if min_scale <= new_scale <= max_scale:
            # 应用缩放
            self.graphics_view.setTransform(new_transform)
            
            # 获取缩放后的场景位置
            new_pos = self.graphics_view.mapToScene(mouse_pos)
            
            # 计算并应用位置调整
            delta = new_pos - old_pos
            self.graphics_view.translate(delta.x(), delta.y())
            
            # 如果缩放后图像小于视图大小，则居中显示
            if mapped_rect.width() < view_width or mapped_rect.height() < view_height:
                self.graphics_view.centerOn(scene_rect.center())
        
        # 阻止事件继续传播
        event.accept()
        # --- End Bug Fix 2 ---
        
    def _update_data_panel(self):
        """更新数据面板显示的分析结果"""
        if not self.current_image_path or self.current_image_path not in self.analysis_data:
            # 如果没有当前图像或没有分析数据，显示默认值
            self.coverage_label.setText("盖度: N/A")
            self.height_label.setText("高度: N/A")
            self.density_label.setText("密度: N/A")
            return
        
        # 获取当前图像的分析数据
        data = self.analysis_data[self.current_image_path]
        
        # 更新标签
        coverage = data.get('盖度', 'N/A')
        if coverage != 'N/A':
            coverage_str = f"{coverage:.1f}%" if isinstance(coverage, (int, float)) else str(coverage)
            self.coverage_label.setText(f"盖度: {coverage_str}")
        else:
            self.coverage_label.setText("盖度: N/A")
            
        height = data.get('高度', 'N/A')
        if height != 'N/A':
            height_str = f"{height:.2f}mm" if isinstance(height, (int, float)) else str(height)
            self.height_label.setText(f"高度: {height_str}")
        else:
            self.height_label.setText("高度: N/A")
            
        density = data.get('密度', 'N/A')
        if density != 'N/A':
            density_str = f"{density:.1f}/m²" if isinstance(density, (int, float)) else str(density)
            self.density_label.setText(f"密度: {density_str}")
        else:
            self.density_label.setText("密度: N/A")

# Example usage (for testing)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    # Need opencv-python for testing
    try:
        import cv2
        import numpy as np
        OPENCV_AVAILABLE = True
    except ImportError:
        OPENCV_AVAILABLE = False
        print("OpenCV not found, image loading test unavailable.")

    app = QApplication(sys.argv)
    viewer = ImageViewerWidget()

    # Create dummy images for testing if OpenCV is available
    test_image_paths = []
    if OPENCV_AVAILABLE:
        os.makedirs("temp_test_images", exist_ok=True)
        for i in range(5):
            img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
            path = os.path.join("temp_test_images", f"test_{i+1}.png")
            # Use cv2.imencode to handle potential path issues, although imwrite should work
            try:
                is_success, buffer = cv2.imencode(".png", img)
                if is_success:
                    with open(path, 'wb') as f:
                        f.write(buffer)
                    test_image_paths.append(path)
                else:
                    print(f"Failed to encode test image {i+1}")
            except Exception as e:
                 print(f"Error writing test image {i+1}: {e}")


        if test_image_paths:
             viewer.load_images(test_image_paths)
             viewer.set_calibration_save_dir("temp_test_images") # Set dir for saving test

    viewer.setWindowTitle("Image Viewer Test")
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec()) 