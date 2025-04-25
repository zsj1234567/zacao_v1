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
        self.result_image_map = {} # Maps original_path -> result_path
        self.loaded_calibration_points = {} # Store loaded/saved points for reuse
        self.pixmaps = {} # Cache loaded pixmaps for thumbnails
        self.current_scene = None
        self.current_pixmap_item = None
        self.calibration_points = []
        self.is_calibration_mode = False
        self.current_view_mode = 'original' # Modes: 'original', 'calibration', 'result'
        self.calibration_save_dir = None # Directory to save calibration files

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # --- 左侧: 预览列表 ---
        self.preview_list = QListWidget()
        # Increase icon size to be wider, adjust height proportionally if needed
        self.preview_list.setIconSize(QSize(120, 90)) # 设置图标大小
        self.preview_list.setFixedWidth(140) # 固定宽度
        # Add border-radius and padding for items
        self.preview_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #555;
                border-radius: 5px; /* Add border-radius */
            }
            QListWidget::item {
                padding: 5px; /* Add some padding around items */
                margin: 2px; /* Add small margin between items */
            }
        """)
        self.preview_list.setViewMode(QListWidget.ViewMode.IconMode) # 图标模式
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setMovement(QListWidget.Movement.Static)
        # Set spacing between icons in IconMode
        self.preview_list.setSpacing(4) # Adjust spacing if needed
        self.preview_list.currentItemChanged.connect(self._on_preview_selected)
        main_layout.addWidget(self.preview_list)

        # --- 右侧: 主视图和控制 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        main_layout.addWidget(right_panel)

        # --- View Mode Buttons --- 
        view_mode_layout = QHBoxLayout()
        self.view_mode_group = QButtonGroup(self)
        self.view_original_button = QPushButton("原始图像")
        self.view_original_button.setCheckable(True)
        self.view_original_button.setChecked(True)
        self.view_calibration_button = QPushButton("校准视图")
        self.view_calibration_button.setCheckable(True)
        self.view_calibration_button.setEnabled(False) # Initially disabled
        self.view_result_button = QPushButton("结果图像")
        self.view_result_button.setCheckable(True)
        self.view_result_button.setEnabled(False) # Initially disabled

        self.view_mode_group.addButton(self.view_original_button, 0)
        self.view_mode_group.addButton(self.view_calibration_button, 1)
        self.view_mode_group.addButton(self.view_result_button, 2)
        self.view_mode_group.idClicked.connect(self._set_view_mode)

        view_mode_layout.addWidget(self.view_original_button)
        view_mode_layout.addWidget(self.view_calibration_button)
        view_mode_layout.addWidget(self.view_result_button)
        view_mode_layout.addStretch()
        right_layout.addLayout(view_mode_layout) # Add buttons above the view

        # 主视图 (使用 QGraphicsView)
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setStyleSheet("QGraphicsView { border: 1px solid #555; }" ) # Add border
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) # 允许拖动
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.graphics_view)

        # 校准控制按钮 (初始隐藏)
        self.calibration_controls_widget = QWidget()
        calibration_controls_layout = QHBoxLayout(self.calibration_controls_widget)
        calibration_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_button = QPushButton("重置点")
        self.reset_button.clicked.connect(self._reset_calibration_points)
        self.save_button = QPushButton("保存校准")
        self.save_button.clicked.connect(self._save_calibration)
        calibration_controls_layout.addWidget(self.reset_button)
        calibration_controls_layout.addWidget(self.save_button)
        right_layout.addWidget(self.calibration_controls_widget)
        self.calibration_controls_widget.setVisible(False)

        # 设置鼠标事件过滤器以捕获点击
        self.graphics_view.viewport().installEventFilter(self)

    def load_images(self, paths: list):
        """加载图像路径列表并生成预览图"""
        self.image_paths = paths
        self.preview_list.clear()
        self.pixmaps.clear()
        self.clear_results() # Clear previous results when loading new images
        self.loaded_calibration_points.clear() # Clear loaded points

        for path in self.image_paths:
            if path not in self.pixmaps:
                try:
                    # 使用numpy和imdecode加载，支持中文路径
                    n = np.fromfile(path, dtype=np.uint8)
                    img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                         print(f"[Viewer] Warning: Failed to decode image {path}")
                         continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.pixmaps[path] = pixmap.scaled(self.preview_list.iconSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                except Exception as e:
                    print(f"[Viewer] Error loading image {path}: {e}")
                    continue # Skip if loading fails

            if path in self.pixmaps:
                item = QListWidgetItem(QIcon(self.pixmaps[path]), os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path) # Store full path
                self.preview_list.addItem(item)

        # 默认显示第一张图片
        if self.image_paths:
            self.preview_list.setCurrentRow(0)
            # Trigger selection change to load the first image explicitly
            if self.preview_list.count() > 0:
                # Ensure signal connection happens if needed, but usually currentItemChanged handles it
                # If setCurrentRow doesn't trigger currentItemChanged when list was empty:
                # Also reload if the list content changed even if path is same
                if not self.current_image_path or self.current_image_path not in self.image_paths:
                     self._on_preview_selected(self.preview_list.item(0), None)
        else:
            # If no images loaded, clear the main view
            self.current_image_path = None
            self.graphics_view.setScene(None)
            self.view_calibration_button.setEnabled(False)
            self.view_result_button.setEnabled(False)

    def display_image(self, image_path: str):
        """加载并显示指定路径的新图像，重置状态。"""
        if image_path == self.current_image_path:
            # print("[Viewer] display_image called for the same image, skipping full reload.")
            return # Avoid redundant reloads if the image hasn't changed

        self.current_image_path = image_path

        success = self._load_and_display_base_image(image_path)

        if success:
            # Successfully loaded new image, reset state
            self.calibration_points = []
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self.view_original_button.setChecked(True) # Set button state
            self.current_view_mode = 'original' # Set internal state
            self.view_calibration_button.setEnabled(True)
            # Enable result button if result exists for this image
            if self.result_image_map.get(image_path):
                self.view_result_button.setEnabled(True)
            else:
                self.view_result_button.setEnabled(False)

            self._redraw_calibration_points() # Clear any old points from scene
        else:
            # Handle error (e.g., clear view, disable buttons)
            self.current_image_path = None
            self.graphics_view.setScene(None) # Clear the view
            self.view_calibration_button.setEnabled(False)
            self.view_result_button.setEnabled(False)

    def _load_and_display_base_image(self, image_path: str) -> bool:
        """Helper to load the base QPixmap for the image and set the scene."""
        if not image_path:
             return False

        full_pixmap = None
        # If preview pixmap exists, maybe load full res later, for now reload
        try:
            n = np.fromfile(image_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if img_bgr is None: raise ValueError("imdecode returned None")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
            full_pixmap = QPixmap.fromImage(q_image)
        except Exception as e:
             print(f"[Viewer] Error loading/reloading full image {image_path}: {e}")
             self.graphics_view.setScene(None)
             return False

        # 创建或更新场景
        if not self.current_scene:
            self.current_scene = QGraphicsScene()
            self.graphics_view.setScene(self.current_scene)
        else:
            # Important: Remove existing items *before* adding new ones,
            # especially the pixmap item to avoid keeping old references.
            # Clear might be okay, but managing items directly can be safer.
            items_to_remove = []
            for item in self.current_scene.items():
                 if item != self.current_pixmap_item: # Keep potential existing pixmap if reusing scene
                      items_to_remove.append(item)
            for item in items_to_remove:
                self.current_scene.removeItem(item)
            # self.current_scene.clear() # Alternative

        # If the pixmap item already exists (e.g., from a previous view mode),
        # update its pixmap instead of creating a new one.
        if self.current_pixmap_item and self.current_pixmap_item in self.current_scene.items():
             self.current_pixmap_item.setPixmap(full_pixmap)
        else:
             self.current_pixmap_item = self.current_scene.addPixmap(full_pixmap)

        self.graphics_view.fitInView(self.current_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio) # Fit view
        return True

    def set_calibration_save_dir(self, directory: str):
        """设置保存校准文件的目录"""
        self.calibration_save_dir = directory

    def set_calibration_mode(self, image_path: str):
        """切换到手动校准模式 (外部调用)"""
        # Ensure the correct image is displayed first
        if image_path != self.current_image_path:
            self.display_image(image_path)
            # Find and select the corresponding item in the list
            for i in range(self.preview_list.count()):
                 item = self.preview_list.item(i)
                 if item and item.data(Qt.ItemDataRole.UserRole) == image_path:
                     # Use blockSignals to prevent triggering _on_preview_selected during manual set
                     self.preview_list.blockSignals(True)
                     self.preview_list.setCurrentItem(item)
                     self.preview_list.blockSignals(False)
                     break
        elif not self.current_image_path and self.image_paths:
            # If called when no image is current, but paths exist, load the first one
             self.display_image(image_path)

        # Now, explicitly set the mode to calibration
        if self.current_image_path == image_path: # Check if display_image was successful
            print(f"[Viewer] Entering calibration mode for: {os.path.basename(image_path)}")
            if self._load_and_display_base_image(self.current_image_path): # Ensure base image is shown
                self.is_calibration_mode = True
                self.current_view_mode = 'calibration'
                self.calibration_points = [] # Clear points when *entering* calibration mode for an image
                self.calibration_controls_widget.setVisible(True)
                self.view_calibration_button.setChecked(True) # Reflect mode in buttons
                self._redraw_calibration_points()
            else:
                print(f"[Viewer] Error: Could not load image {image_path} to enter calibration mode.")
                QMessageBox.warning(self, "错误", f"无法加载图像 {os.path.basename(image_path)} 以进行校准。")
        else:
             print(f"[Viewer] Error: Cannot enter calibration mode, image {image_path} not loaded.")
             QMessageBox.warning(self, "错误", f"无法进入校准模式，图像 {os.path.basename(image_path)} 未加载。")

    def _on_preview_selected(self, current_item, previous_item):
        """当预览列表中的项目被选中时调用"""
        # Important: Prevent processing if selection is cleared or invalid
        if not current_item:
            self.current_image_path = None
            self.graphics_view.setScene(None)
            self.view_calibration_button.setEnabled(False)
            self.view_result_button.setEnabled(False)
            return

        image_path = current_item.data(Qt.ItemDataRole.UserRole)
        if image_path != self.current_image_path:
            # Only display if the path actually changed
            self.display_image(image_path)

    def _set_view_mode(self, button_id):
        """根据点击的按钮切换视图模式"""
        if not self.current_image_path: # No image loaded, do nothing
             print("[Viewer] No image loaded, cannot change view mode.")
             # Ensure buttons reflect this state if needed
             self.view_original_button.setChecked(True)
             return

        target_mode = 'original'
        if button_id == 1:
            target_mode = 'calibration'
        elif button_id == 2:
            target_mode = 'result'

        if target_mode == self.current_view_mode:
            # print(f"[Viewer] View mode already set to {target_mode}, skipping.")
            return # No change needed

        print(f"[Viewer] Switching view mode to: {target_mode}")
        previous_mode = self.current_view_mode
        self.current_view_mode = target_mode # Update internal state

        # Reset calibration state by default, enable only if in calibration mode
        self.is_calibration_mode = False
        self.calibration_controls_widget.setVisible(False)

        # Perform actions based on the target mode
        if target_mode == 'original':
            self._load_and_display_base_image(self.current_image_path)
            # No specific drawing needed here unless switching from calibration
        elif target_mode == 'calibration':
            if self._load_and_display_base_image(self.current_image_path):
                 self.is_calibration_mode = True # Enable calibration state
                 self.calibration_controls_widget.setVisible(True)
                 # Load existing points if available when switching to calibration view
                 if self.current_image_path in self.loaded_calibration_points:
                     self.calibration_points = self.loaded_calibration_points[self.current_image_path]
                 else:
                      self.calibration_points = [] # Start fresh if no loaded points
                 self._redraw_calibration_points() # Draw any existing points for this image
            else:
                 self.current_view_mode = previous_mode # Revert mode if load failed
                 # Maybe disable calibration button?
        elif target_mode == 'result':
            self._display_result_image(self.current_image_path) # Load and show result

        # Always redraw points after mode switch to ensure correct visibility/state
        # (e.g., clear points if switching away from calibration)
        if not self.is_calibration_mode:
             # Ensure points associated with calibration mode are visually cleared
             # but don't delete self.calibration_points data if we might switch back
             self._redraw_calibration_points() # This will clear if is_calibration_mode is False

        # If mode actually changed FROM calibration to something else,
        # keep the points in memory (self.calibration_points)
        # but they won't be drawn unless we switch back.

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
                        print(f"[Viewer] Calibration point {len(self.calibration_points)} added: ({item_pos.x()}, {item_pos.y()})")
                        self._redraw_calibration_points()
                        # self.calibration_point_selected.emit(self.current_image_path, item_pos.x(), item_pos.y())
                        return True # 事件已处理
                else:
                    print("[Viewer] Max 4 calibration points reached.")
                    QMessageBox.information(self, "提示", "已选择4个点。请点击 '保存校准' 或 '重置点'。")
                    return True # Prevent other actions while max points reached

        return super().eventFilter(source, event)

    def _redraw_calibration_points(self):
        """在场景中重新绘制校准点和连线"""
        # First, ensure the base image is visible if we are in calibration mode
        if self.is_calibration_mode and self.current_pixmap_item is None:
             print("[Viewer] Redraw requested in calibration mode but no base image item found. Reloading.")
             self._load_and_display_base_image(self.current_image_path)

        if not self.current_scene or not self.current_pixmap_item:
            # Clear scene if no pixmap item (e.g., image failed to load)
            if self.current_scene:
                 # Remove only calibration artifacts
                 for item in self.current_scene.items():
                     if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem)):
                         self.current_scene.removeItem(item)
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

        # Draw only if in calibration mode and pixmap exists
        if not self.is_calibration_mode or not self.calibration_points or not self.current_pixmap_item:
             return

        # 绘制新的点和线
        pen = QPen(QColor("red"), 5)
        line_pen = QPen(QColor("lime"), 2)
        point_radius = 4 # Controls the size of the red dot
        label_offset_x = 5
        label_offset_y = -15

        # Ensure points are within bounds (precautionary)
        pixmap_rect = self.current_pixmap_item.boundingRect()

        scene_points = []
        for i, pt in enumerate(self.calibration_points):
            # Clamp point coordinates to be within the pixmap bounds
            # Check if pt is valid tuple/list
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                 print(f"[Viewer] Invalid point format skipped: {pt}")
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
            ellipse.setPen(pen)
            ellipse.setBrush(QColor("red"))
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

    def _reset_calibration_points(self):
        """重置当前图像的校准点"""
        if self.is_calibration_mode:
            print(f"[Viewer] Resetting calibration points for {os.path.basename(self.current_image_path)}")
            self.calibration_points = []
            # Also clear from loaded dict if resetting
            if self.current_image_path in self.loaded_calibration_points:
                del self.loaded_calibration_points[self.current_image_path]
            self._redraw_calibration_points()
            # self.calibration_reset_requested.emit(self.current_image_path)

    def _save_calibration(self):
        """保存当前校准点并发出信号"""
        if self.is_calibration_mode and len(self.calibration_points) == 4:
            print(f"[Viewer] Saving calibration points for {os.path.basename(self.current_image_path)}: {self.calibration_points}")
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
                    print(f"校准文件已保存到: {save_path}")
                    save_successful = True
                except Exception as e:
                    print(f"[Viewer] Error saving calibration file {save_path}: {e}")
                    QMessageBox.critical(self, "保存错误", f"无法保存校准文件到 {save_path}:\n{e}")
                    save_successful = False
            else:
                print("[Viewer] Error: Calibration save directory not set.")
                QMessageBox.warning(self, "保存错误", "未设置校准文件保存目录。")
                save_successful = False

            # 只有在保存成功后才退出校准模式
            if save_successful:
                self.exit_calibration_mode() # Use the dedicated exit function
                QMessageBox.information(self, "校准已保存", f"{os.path.basename(self.current_image_path)} 的校准点已保存。")

        elif self.is_calibration_mode:
            print("[Viewer] Error: Need exactly 4 points to save calibration.")
            QMessageBox.warning(self, "校准错误", "需要选择4个角点才能保存。")

    def exit_calibration_mode(self):
        """退出手动校准模式 (通常在保存后调用)"""
        print("[Viewer] Exiting calibration mode.")
        # Set button state back to original
        self.view_original_button.setChecked(True)
        # Trigger view mode change if not already original
        self._set_view_mode(0) # This handles is_calibration_mode, controls visibility, redraw

    def _display_result_image(self, original_image_path):
        """加载并显示分析结果图像"""
        result_path = self.result_image_map.get(original_image_path)
        if result_path:
            print(f"[Viewer] Loading result image from: {result_path}")
            # Reuse the base loading logic for now, assuming result is an image file
            success = self._load_and_display_base_image(result_path)
            if not success:
                self._show_placeholder_scene(f"无法加载结果图像\n{os.path.basename(result_path)}")
        else:
            print(f"[Viewer] No result image path found for {os.path.basename(original_image_path)}")
            self._show_placeholder_scene(f"未找到结果图像\n{os.path.basename(original_image_path)}")

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

    def set_result_image_path(self, original_path: str, result_path: Optional[str]):
        """Stores the path to the result image for a given original image."""
        if result_path and os.path.exists(result_path):
            self.result_image_map[original_path] = result_path
            print(f"[Viewer] Result path set for {os.path.basename(original_path)}: {result_path}")
            # If the currently displayed image is this one, enable the button
            if self.current_image_path == original_path:
                self.view_result_button.setEnabled(True)
        else:
            if original_path in self.result_image_map:
                print(f"[Viewer] Clearing result path for {os.path.basename(original_path)}")
                del self.result_image_map[original_path]
            # If the currently displayed image is this one, disable the button
            if self.current_image_path == original_path:
                self.view_result_button.setEnabled(False)

    def clear_results(self):
        """Clears the stored result image paths."""
        self.result_image_map.clear()
        # Disable button generally only if no image is selected
        if self.current_image_path and self.current_image_path not in self.result_image_map:
             self.view_result_button.setEnabled(False)
        elif not self.current_image_path:
             self.view_result_button.setEnabled(False)
        print("[Viewer] Cleared all result image paths.")

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
                  print(f"[Viewer] Error converting calibration points to int: {e}, points: {current_points}")
                  return None
         return None

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