import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QPushButton, QSizePolicy, QMessageBox,
    QButtonGroup, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QIcon
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
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setStyleSheet("QGraphicsView { border: 1px solid #555; }" ) # Add border
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) # 允许拖动
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.graphics_view)

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

        self.image_paths = paths
        self.current_image_path = self.image_paths[0] if self.image_paths else None
        self._set_view_mode(0)

        if self.image_paths:
            self.display_image(self.image_paths[0])
        else:
             print("[Viewer] No valid images loaded, view cleared.")

    def display_image(self, image_path: str):
        """加载并显示指定路径的原始图像，重置状态。"""
        self.current_image_path = image_path
        self.calibration_points = []

        if self.current_view_mode != 'original':
             self.current_view_mode = 'original'
             self.is_calibration_mode = False
             self.calibration_controls_widget.setVisible(False)
             self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
             self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
             self.view_original_button.setChecked(True)

        self.is_calibration_mode = False
        self.calibration_controls_widget.setVisible(False)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

        success = self._load_and_display_base_image(image_path)

        if success:
            self.view_calibration_button.setEnabled(True)
            result_dict = self.result_image_map.get(image_path, {})
            has_image_paths = any(isinstance(v, str) for k, v in result_dict.items() if k != 'calibration_points')
            self.view_result_button.setEnabled(has_image_paths)
        else:
            self.current_image_path = None
            self.graphics_view.setScene(None)
            self.view_calibration_button.setEnabled(False)
            self.view_result_button.setEnabled(False)

    def _load_and_display_base_image(self, image_path: str) -> bool:
        """Helper to load the base QPixmap for the image and set the scene.
           Used for displaying both original and result images.
           Does NOT change the view mode state.
        """
        if not image_path:
            logging.error("[Viewer] Error: No image path provided for loading.")
            return False

        full_pixmap = None
        try:
            n = np.fromfile(image_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img_bgr is None: raise ValueError("imdecode returned None")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
            full_pixmap = QPixmap.fromImage(q_image)
        except Exception as e:
            logging.error(f"[Viewer] Error loading/reloading image '{os.path.basename(image_path)}': {e}")
            self.graphics_view.setScene(None)
            self.current_pixmap_item = None
            return False

        scene = self.graphics_view.scene()
        if scene:
             scene.clear()
             self.current_scene = scene
        else:
             self.current_scene = QGraphicsScene()
             self.graphics_view.setScene(self.current_scene)

        if full_pixmap:
            self.current_pixmap_item = self.current_scene.addPixmap(full_pixmap)
            self.graphics_view.fitInView(self.current_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_scene.update()
            self.graphics_view.viewport().update()
            return True
        else:
            logging.critical(f"[Viewer] CRITICAL ERROR: full_pixmap is None after loading {os.path.basename(image_path)}")
            self.current_pixmap_item = None
            self.graphics_view.setScene(None)
            return False

    def set_calibration_save_dir(self, directory: str):
        """设置保存校准文件的目录"""
        self.calibration_save_dir = directory

    def _enter_calibration_logic(self, image_path: str):
        """执行进入手动校准模式的逻辑 (加载图像、点、设置控件等)
           假定调用此函数时，视图模式已经是 'calibration'。
        """
        logging.info(f"[Viewer _enter_calibration_logic] Entering logic for: {os.path.basename(image_path)}")

        # Check if target image is the current context
        is_already_current = (image_path == self.current_image_path)
        load_success = False

        # 1. Ensure the base image is loaded and displayed
        if not is_already_current:
            # If it's a new image, load it (display_image handles setting self.current_image_path)
            logging.info(f"[Viewer _enter_calibration_logic] Calibration target '{os.path.basename(image_path)}' is not current. Loading original...")
            # Call display_image which uses _load_and_display_base_image
            self.display_image(image_path)
            # Check if display_image was successful (it calls _load_and_display_base_image)
            load_success = (self.current_image_path == image_path and self.current_pixmap_item is not None)

            if load_success:
                 # Select the item in the preview list without triggering selection signal again
                 for i in range(self.preview_list.count()):
                      item = self.preview_list.item(i)
                      if item and item.data(Qt.ItemDataRole.UserRole) == image_path:
                           self.preview_list.blockSignals(True)
                           self.preview_list.setCurrentItem(item)
                           self.preview_list.blockSignals(False)
                           break
            else:
                 logging.error(f"[Viewer] Failed to load image {image_path} for calibration.")
                 QMessageBox.warning(self, "加载错误", f"无法加载用于校准的图像:\n{os.path.basename(image_path)}")
                 return
        elif not self.current_pixmap_item or not self.current_pixmap_item.scene():
            # If it *was* the current image, but pixmap/scene is missing, try reloading.
            logging.warning(f"[Viewer _enter_calibration_logic] Calibration target '{os.path.basename(image_path)}' is current, but pixmap/scene missing. Reloading original...")
            load_success = self._load_and_display_base_image(image_path)
            if not load_success:
                 logging.error(f"[Viewer] Failed to reload current image {image_path} for calibration.")
                 QMessageBox.warning(self, "加载错误", f"无法重新加载当前图像以进行校准:\n{os.path.basename(image_path)}")
                 return
        else:
            # Image is current and pixmap/scene seems okay
            logging.debug(f"[Viewer _enter_calibration_logic] Calibration target '{os.path.basename(image_path)}' is current and seems loaded.")
            load_success = True # Assume success as it's already loaded

        # 2. Proceed only if image is loaded successfully
        if load_success and self.current_image_path == image_path and self.current_pixmap_item and self.current_pixmap_item.scene():
            logging.info(f"[Viewer _enter_calibration_logic] Entering calibration mode for: {os.path.basename(image_path)}")

            self.is_calibration_mode = True
            # self.current_view_mode = 'calibration' # Should be set by _set_view_mode(1)

            # Load existing points for this image or clear if none
            if self.current_image_path in self.loaded_calibration_points:
                self.calibration_points = self.loaded_calibration_points[self.current_image_path][:] # Use a copy
                logging.info(f"[Viewer] Loaded {len(self.calibration_points)} existing calibration points.")
            else:
                self.calibration_points = []
                logging.info("[Viewer _enter_calibration_logic] No existing calibration points found, starting fresh.")

            # Update UI elements for calibration mode
            self.calibration_controls_widget.setVisible(True)

            self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.graphics_view.viewport().setCursor(Qt.CursorShape.CrossCursor)

            # Draw points (this relies on the pixmap item being correctly set)
            self._redraw_calibration_points()
            logging.info(f"[Viewer _enter_calibration_logic] Calibration mode successfully entered for: {os.path.basename(image_path)}")

        else:
            # This case should ideally not be reached if the logic above is correct
            logging.critical(f"[Viewer] Critical Error: Failed to enter calibration mode. Image '{os.path.basename(image_path)}' loaded state inconsistent.")
            QMessageBox.critical(self, "内部错误", f"无法进入校准模式。\n图像加载状态异常，请重试或检查日志。")
            # Reset potentially inconsistent state?
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.graphics_view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            # Try to revert to original view
            self._set_view_mode(0)

    def _on_preview_selected(self, current_item, previous_item):
        """处理预览列表中的选择更改"""
        if not current_item:
            # No item selected, clear the main view? (Let's avoid clearing for now)
            # self.graphics_view.setScene(None)
            # self.current_pixmap_item = None
            logging.info("[Viewer _on_preview_selected] No item selected. View remains unchanged.")
            return

        selected_data = current_item.data(Qt.ItemDataRole.UserRole)

        if self.current_view_mode == 'original' or self.current_view_mode == 'calibration':
            if isinstance(selected_data, str):
                original_path = selected_data
                # --- FIX: Only display if path is different OR pixmap is missing --- #
                if original_path != self.current_image_path or not self.current_pixmap_item:
                    logging.info(f"[Viewer _on_preview_selected] Original/Calib mode: Selection changed or pixmap missing. Displaying {os.path.basename(original_path)}")
                    self.display_image(original_path) # Handles setting self.current_image_path and loading
                else:
                    logging.debug(f"[Viewer _on_preview_selected] Original/Calib mode: Selected image ({os.path.basename(original_path)}) is already displayed. Doing nothing.")
                # --- END FIX --- #
            else:
                 logging.warning(f"[Viewer] Error: Expected string path in original/calibration mode, got {type(selected_data)}")

        elif self.current_view_mode == 'result':
            if isinstance(selected_data, tuple) and len(selected_data) == 2:
                result_type, result_path = selected_data
                # --- FIX: Only display if result path is different OR pixmap is missing --- #
                # Check if the result image is already displayed (tricky without storing current result path explicitly)
                # Let's always reload the result image for simplicity for now, but avoid changing mode state.
                logging.info(f"[Viewer _on_preview_selected] Result mode: Selected {result_type}. Displaying {os.path.basename(result_path)}")
                self._load_and_display_base_image(result_path)
                # --- END FIX --- #
            else:
                 logging.warning(f"[Viewer] Error: Expected tuple (type, path) in result mode, got {type(selected_data)}")

        else:
            logging.warning(f"[Viewer _on_preview_selected] Unknown view mode: {self.current_view_mode}")

    def _set_view_mode(self, mode_id: int, target_image_path: Optional[str] = None):
        """根据按钮点击设置视图模式。 0: original, 1: calibration, 2: result"""
        # Determine the effective image path for context (if needed)
        effective_image_path = target_image_path if target_image_path else self.current_image_path

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
                 # Call display_image. It now only reloads if necessary.
                 logging.info(f"[Viewer _set_view_mode] Calling display_image for Original: {os.path.basename(effective_image_path)}")
                 self.display_image(effective_image_path)
                 # Check if display_image succeeded in setting the pixmap item
                 image_loaded_successfully = (self.current_image_path == effective_image_path and self.current_pixmap_item is not None)
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

            # Set UI state for calibration *before* calling the logic function
            self.current_view_mode = 'calibration'
            self.view_calibration_button.setChecked(True)
            self._populate_preview_list('original') # Calibration view uses original previews

            # Call the function that handles the actual calibration entry logic
            self._enter_calibration_logic(path_to_calibrate)

        elif mode_id == 2: # Result
            logging.info("[Viewer _set_view_mode] Switching to Result mode.")
            if not self.current_image_path: return # Should be handled above

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

            # Display the first result image by default (handled by _populate_preview_list selecting first item)
            # Ensure correct button is checked
            self.view_result_button.setChecked(True)

    def _populate_preview_list(self, mode: str, data: Optional[dict] = None):
        """Clears and populates the preview list based on the view mode."""
        self.preview_list.clear()
        self.pixmaps.clear() # Clear pixmap cache when changing modes

        fm = self.preview_list.fontMetrics()
        # --- Bug Fix 1: Use icon width for text eliding --- #
        icon_width = self.preview_list.iconSize().width()
        # Allow for some padding within the item under the icon
        available_width = icon_width - 10 # Subtract some padding
        if available_width < 20: # Ensure minimum width for ellipsis
             available_width = 20
        # --- End Bug Fix 1 ---

        if mode == 'original':
            logging.debug("[Viewer _populate_preview_list] Populating with ORIGINAL images.")
            items_added = 0
            for path in self.image_paths:
                 pixmap = self._load_thumbnail(path)
                 if pixmap:
                     base_filename = os.path.basename(path)
                     # --- Bug Fix 1: Optimization for ellipsis ---
                     text_width = fm.boundingRect(base_filename).width()
                     if text_width > available_width:
                         display_text = fm.elidedText(base_filename, Qt.TextElideMode.ElideRight, available_width)
                     else:
                         display_text = base_filename
                     # --- End Bug Fix 1 ---
                     item = QListWidgetItem(QIcon(pixmap), display_text) # Use potentially elided text
                     item.setData(Qt.ItemDataRole.UserRole, path) # Store full original path
                     item.setToolTip(path) # Tooltip 显示完整路径
                     self.preview_list.addItem(item)
                     items_added += 1
            # Select the current original image (logic moved to _set_view_mode)
            # if self.current_image_path and items_added > 0:
            #      # ... selection logic ...
            # elif items_added > 0:
            #      self.preview_list.setCurrentRow(0)

        elif mode == 'result':
            logging.debug("[Viewer _populate_preview_list] Populating with RESULT images.")
            if not data or not isinstance(data, dict):
                logging.warning("[Viewer _populate_preview_list] No result data provided.")
                return

            items_added = 0
            # Sort results for consistent order (e.g., analysis first)
            sorted_results = sorted(data.items(), key=lambda x: (0 if x[0] == 'analysis_image' else 1, x[0]))

            # Note: available_width calculation moved outside the loop

            for result_type, result_path in sorted_results:
                if not result_path or not isinstance(result_path, str) or not os.path.exists(result_path):
                    logging.debug(f"[Viewer] Skipping invalid result path for type '{result_type}': {result_path}")
                    continue
                pixmap = self._load_thumbnail(result_path)
                if pixmap:
                    # --- Bug Fix 1: Optimization for ellipsis ---
                    text_width = fm.boundingRect(result_type).width()
                    if text_width > available_width:
                        display_text = fm.elidedText(result_type, Qt.TextElideMode.ElideRight, available_width)
                    else:
                        display_text = result_type
                    # --- End Bug Fix 1 ---
                    item = QListWidgetItem(QIcon(pixmap), display_text) # Use potentially elided text
                    item.setData(Qt.ItemDataRole.UserRole, (result_type, result_path)) # Store tuple
                    item.setToolTip(result_path) # Tooltip 显示完整路径
                    self.preview_list.addItem(item)
                    items_added += 1
            # Select the first result item by default
            if items_added > 0:
                self.preview_list.setCurrentRow(0) # Select first result
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
        """事件过滤器，用于捕获图形视图上的鼠标点击以进行校准"""
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
 
        # 绘制新的点和线
        point_pen = QPen(QColor(255, 0, 0), 30) # 亮红色点轮廓
        point_brush = QColor(255, 0, 0, 200) # 半透明亮红色填充
        line_pen = QPen(QColor(255, 0, 0, 230), 30) # 亮红色粗线
        point_radius = 10 # 较大的点半径
        label_offset_x = 10
        label_offset_y = -20

        # Ensure points are within bounds (precautionary)
        pixmap_rect = self.current_pixmap_item.boundingRect()

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
            font.setPointSize(12) # Make number slightly larger
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

        logging.debug(f"[Viewer _redraw_calibration_points] Drawing {len(self.calibration_points)} points.")

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

            # 发出信号，传递图像路径和点列表
            self.calibration_save_requested.emit(self.current_image_path, self.calibration_points)

            # 同时将校准点保存到 JSON 文件
            save_path = None # Initialize save_path
            if self.calibration_save_dir:
                img_basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
                save_path = os.path.join(self.calibration_save_dir, f"{img_basename}.json")
                try:
                    # 确保点是整数
                    int_points = [[int(round(p[0])), int(round(p[1]))] for p in self.calibration_points]
                    # 顺时针排序 (可选但推荐)
                    # ordered_points = self._order_points_clockwise(int_points)
                    ordered_points = int_points # Assuming points are added somewhat clockwise

                    with open(save_path, 'w') as f:
                        json.dump(ordered_points, f, indent=4) # Use indent for readability
                    logging.info(f"校准文件已保存到: {save_path}")
                    save_successful = True
                except Exception as e:
                    logging.error(f"[Viewer] Error saving calibration file {save_path}: {e}")
                    QMessageBox.critical(self, "保存错误", f"无法保存校准文件到 {save_path}:\n{e}")
                    save_successful = False
            else:
                logging.error("[Viewer] Error: Calibration save directory not set.")
                QMessageBox.warning(self, "保存错误", "未设置校准文件保存目录。")
                save_successful = False

            # 只有在保存成功后才退出校准模式
            if save_successful:
                self.exit_calibration_mode() # Use the dedicated exit function
                QMessageBox.information(self, "校准已保存", f"{os.path.basename(self.current_image_path)} 的校准点已保存。")

        elif self.is_calibration_mode:
            logging.info("[Viewer] Error: Need exactly 4 points to save calibration.")
            QMessageBox.warning(self, "校准错误", "需要选择4个角点才能保存。")

    def exit_calibration_mode(self):
        """退出手动校准模式 (通常在保存后调用)"""
        logging.info("[Viewer] Exiting calibration mode.")
        # Set button state back to original
        self.view_original_button.setChecked(True)
        # Trigger view mode change if not already original
        self._set_view_mode(0) # This handles is_calibration_mode, controls visibility, redraw

    def _display_result_image(self, original_image_path: str, result_type: str):
        """显示指定类型的结果图像"""
        result_path_dict = self.result_image_map.get(original_image_path, {})
        result_path = result_path_dict.get(result_type)

        if not result_path:
            logging.warning(f"[Viewer] No result path found for type '{result_type}'. Showing placeholder.")
            self._show_placeholder_scene(f"结果图像 ({result_type}) 不可用")
            return

        # Load and display the specific result image
        success = self._load_and_display_base_image(result_path)
        if not success:
            self._show_placeholder_scene(f"无法加载结果图像 ({result_type})\n{result_path}")
        else:
             logging.info(f"[Viewer] Displaying result: {result_type} from {os.path.basename(result_path)}")
             self.current_view_mode = 'result' # Ensure mode is set
             # Optionally add overlay text indicating the result type being shown
             # self._add_overlay_text(f"显示: {result_type}")

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
        """Receives the results dictionary for a single original image."""
        if results and isinstance(results, dict):
            # --- FIX 2: More permissive filtering --- #
            keys_to_exclude = {'文件名', '分析模型', '状态', '错误信息', '详细错误', 'original_path'}
            filtered_results = {
                k: v for k, v in results.items()
                if k not in keys_to_exclude and v is not None # Keep all other non-None key-values
            }
            self.result_image_map[original_path] = filtered_results
            # --- END FIX 2 --- #

            num_items = len(self.result_image_map[original_path])
            logging.info(f"[Viewer] Stored {num_items} result items for {os.path.basename(original_path)}: {list(self.result_image_map[original_path].keys())}")

            # Update UI only if this is the currently displayed image
            if original_path == self.current_image_path:
                 # --- FIX: Check for any string value (assumed path) besides calibration_points --- #
                 # has_image_paths = any(isinstance(v, str) for k, v in self.result_image_map[original_path].items() if k != 'calibration_points')
                 has_image_paths = any(isinstance(v, str) for k, v in filtered_results.items() if k != 'calibration_points')
                 self.view_result_button.setEnabled(has_image_paths)
                 # --- END FIX --- #
                 if self.current_view_mode == 'result':
                     logging.info("[Viewer] Results updated for current image while in result view. Re-displaying results.")
                     self._set_view_mode(2) # Re-trigger result view display

        else:
             # Clear results for this path if results are None or invalid
             if original_path in self.result_image_map:
                 del self.result_image_map[original_path]
             if original_path == self.current_image_path:
                 self.view_result_button.setEnabled(False)
                 # If currently viewing results for this image, switch back to original
                 if self.current_view_mode == 'result':
                     self._set_view_mode(0)

    def clear_results(self):
        """清除所有存储的结果映射"""
        self.result_image_map.clear()
        # If an image is currently displayed, disable its result button
        if self.current_image_path:
            self.view_result_button.setEnabled(False)
        logging.info("[Viewer] All result maps cleared.")

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

    # --- New Slot to handle button click and get ID --- #
    def _handle_button_click(self, button):
        """Receives the clicked button object and calls _set_view_mode with its ID."""
        mode_id = self.view_mode_group.id(button)
        logging.debug(f"[Viewer _handle_button_click] Button '{button.text()}' clicked, ID: {mode_id}")
        if mode_id != -1: # QButtonGroup returns -1 if button not found or has no ID
            self._set_view_mode(mode_id)
        else:
             logging.warning("[Viewer _handle_button_click] Warning: Clicked button has no valid ID in the group.")
    # --- End New Slot --- #

    # --- Bug Fix 2: Add wheelEvent for zooming --- #
    def wheelEvent(self, event):
        # Allow zooming only in original or result view, not calibration
        if self.is_calibration_mode:
            super().wheelEvent(event) # Pass event up if in calibration
            return

        # Zoom Factor
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Check if an image is loaded
        if self.current_pixmap_item is None:
            super().wheelEvent(event)
            return

        # Set Anchors
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Zoom
        angle = event.angleDelta().y()
        if angle > 0:
            self.graphics_view.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.graphics_view.scale(zoom_out_factor, zoom_out_factor)

        # Don't pass the event up, we handled it
        # super().wheelEvent(event)
        # --- End Bug Fix 2 ---

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