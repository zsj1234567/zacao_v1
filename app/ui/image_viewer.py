import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QPushButton, QSizePolicy, QMessageBox
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
        self.pixmaps = {} # Cache loaded pixmaps for thumbnails
        self.current_scene = None
        self.current_pixmap_item = None
        self.calibration_points = []
        self.is_calibration_mode = False
        self.calibration_save_dir = None # Directory to save calibration files

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # --- 左侧: 预览列表 ---
        self.preview_list = QListWidget()
        self.preview_list.setIconSize(QSize(100, 100)) # 设置图标大小
        self.preview_list.setFixedWidth(140) # 固定宽度
        self.preview_list.setViewMode(QListWidget.ViewMode.IconMode) # 图标模式
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setMovement(QListWidget.Movement.Static)
        self.preview_list.currentItemChanged.connect(self._on_preview_selected)
        main_layout.addWidget(self.preview_list)

        # --- 右侧: 主视图和控制 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        main_layout.addWidget(right_panel)

        # 主视图 (使用 QGraphicsView)
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
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
            # self.display_image(self.image_paths[0]) # _on_preview_selected handles this

    def display_image(self, image_path: str):
        """在主视图中显示指定路径的图像"""
        self.current_image_path = image_path
        self.is_calibration_mode = False # Default mode is viewing
        self.calibration_points = []
        self.calibration_controls_widget.setVisible(False)

        if image_path not in self.pixmaps:
            # If preview loading failed, try loading full image again
            try:
                n = np.fromfile(image_path, dtype=np.uint8)
                img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
                if img_bgr is None: raise ValueError("imdecode returned None")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
                full_pixmap = QPixmap.fromImage(q_image)
            except Exception as e:
                 print(f"[Viewer] Error loading full image {image_path}: {e}")
                 self.graphics_view.setScene(None)
                 return
        else:
            # Use cached full pixmap if available (needs caching mechanism)
            # For now, reload full image
            try:
                n = np.fromfile(image_path, dtype=np.uint8)
                img_bgr = cv2.imdecode(n, cv2.IMREAD_COLOR)
                if img_bgr is None: raise ValueError("imdecode returned None")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                q_image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
                full_pixmap = QPixmap.fromImage(q_image)
            except Exception as e:
                 print(f"[Viewer] Error reloading full image {image_path}: {e}")
                 self.graphics_view.setScene(None)
                 return


        # 创建新的场景并添加图像
        self.current_scene = QGraphicsScene()
        self.current_pixmap_item = self.current_scene.addPixmap(full_pixmap)
        self.graphics_view.setScene(self.current_scene)
        self.graphics_view.fitInView(self.current_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio) # Fit view

    def set_calibration_save_dir(self, directory: str):
        """设置保存校准文件的目录"""
        self.calibration_save_dir = directory

    def set_calibration_mode(self, image_path: str):
        """切换到手动校准模式"""
        if image_path != self.current_image_path:
            self.display_image(image_path)
            # Find and select the corresponding item in the list
            for i in range(self.preview_list.count()):
                 item = self.preview_list.item(i)
                 if item.data(Qt.ItemDataRole.UserRole) == image_path:
                     self.preview_list.setCurrentItem(item)
                     break

        print(f"[Viewer] Entering calibration mode for: {os.path.basename(image_path)}")
        self.is_calibration_mode = True
        self.calibration_points = []
        self.calibration_controls_widget.setVisible(True)
        self._redraw_calibration_points()

    def _on_preview_selected(self, current_item, previous_item):
        """当预览列表中的项目被选中时调用"""
        if current_item:
            image_path = current_item.data(Qt.ItemDataRole.UserRole)
            self.display_image(image_path)

    def eventFilter(self, source, event):
        """事件过滤器，用于捕获图形视图上的鼠标点击以进行校准"""
        if source == self.graphics_view.viewport() and event.type() == event.Type.MouseButtonPress:
            if self.is_calibration_mode and event.button() == Qt.MouseButton.LeftButton and len(self.calibration_points) < 4:
                # 将视图坐标转换为场景坐标
                scene_pos = self.graphics_view.mapToScene(event.pos())
                # 确保点击在图像内部
                if self.current_pixmap_item and self.current_pixmap_item.contains(scene_pos):
                    point = scene_pos.toPoint()
                    # 添加点（相对于 pixmap item 的坐标）
                    item_pos = self.current_pixmap_item.mapFromScene(scene_pos).toPoint()
                    self.calibration_points.append((item_pos.x(), item_pos.y()))
                    print(f"[Viewer] Calibration point {len(self.calibration_points)} added: ({item_pos.x()}, {item_pos.y()})")
                    self._redraw_calibration_points()
                    # self.calibration_point_selected.emit(self.current_image_path, item_pos.x(), item_pos.y())
                    return True # 事件已处理
        return super().eventFilter(source, event)

    def _redraw_calibration_points(self):
        """在场景中重新绘制校准点和连线"""
        if not self.current_scene or not self.current_pixmap_item:
            return

        # 移除旧的点和线 (需要存储它们的引用或者按类型查找)
        for item in self.current_scene.items():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem)):
                self.current_scene.removeItem(item)

        # 绘制新的点和线
        pen = QPen(QColor("red"), 5)
        line_pen = QPen(QColor("lime"), 2)

        for i, pt in enumerate(self.calibration_points):
            # 将点坐标从 item 坐标转换回场景坐标
            scene_pt = self.current_pixmap_item.mapToScene(QPointF(pt[0], pt[1]))
            # 绘制点
            ellipse = QGraphicsEllipseItem(scene_pt.x() - 4, scene_pt.y() - 4, 8, 8)
            ellipse.setPen(pen)
            ellipse.setBrush(QColor("red"))
            self.current_scene.addItem(ellipse)
            # 绘制序号
            text = QGraphicsSimpleTextItem(str(i + 1))
            text.setPos(scene_pt.x() + 5, scene_pt.y() - 15)
            text.setBrush(QColor("red"))
            self.current_scene.addItem(text)

            # 绘制连线
            if i > 0:
                prev_scene_pt = self.current_pixmap_item.mapToScene(QPointF(self.calibration_points[i-1][0], self.calibration_points[i-1][1]))
                line = QGraphicsLineItem(prev_scene_pt.x(), prev_scene_pt.y(), scene_pt.x(), scene_pt.y())
                line.setPen(line_pen)
                self.current_scene.addItem(line)

        # 绘制闭合线
        if len(self.calibration_points) == 4:
            first_scene_pt = self.current_pixmap_item.mapToScene(QPointF(self.calibration_points[0][0], self.calibration_points[0][1]))
            last_scene_pt = self.current_pixmap_item.mapToScene(QPointF(self.calibration_points[-1][0], self.calibration_points[-1][1]))
            line = QGraphicsLineItem(last_scene_pt.x(), last_scene_pt.y(), first_scene_pt.x(), first_scene_pt.y())
            line.setPen(line_pen)
            self.current_scene.addItem(line)

    def _reset_calibration_points(self):
        """重置当前图像的校准点"""
        if self.is_calibration_mode:
            print(f"[Viewer] Resetting calibration points for {os.path.basename(self.current_image_path)}")
            self.calibration_points = []
            self._redraw_calibration_points()
            # self.calibration_reset_requested.emit(self.current_image_path)

    def _save_calibration(self):
        """保存当前校准点并发出信号"""
        if self.is_calibration_mode and len(self.calibration_points) == 4:
            print(f"[Viewer] Saving calibration points for {os.path.basename(self.current_image_path)}: {self.calibration_points}")
            # 发出信号，传递图像路径和点列表
            self.calibration_save_requested.emit(self.current_image_path, self.calibration_points)
            # 同时将校准点保存到 JSON 文件
            if self.calibration_save_dir:
                img_basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
                save_path = os.path.join(self.calibration_save_dir, f"{img_basename}.json")
                try:
                    # 确保点是顺时针排序的 (可选但推荐)
                    # ordered_points = self._order_points_clockwise(self.calibration_points)
                    ordered_points = self.calibration_points # Assuming points are added somewhat clockwise
                    with open(save_path, 'w') as f:
                        json.dump(ordered_points, f)
                    print(f"校准文件已保存到: {save_path}")
                except Exception as e:
                    print(f"[Viewer] Error saving calibration file {save_path}: {e}")
                    QMessageBox.critical(self, "保存错误", f"无法保存校准文件到 {save_path}:\n{e}")
            else:
                print("[Viewer] Error: Calibration save directory not set.")
                QMessageBox.warning(self, "保存错误", "未设置校准文件保存目录。")
            # 退出校准模式（可选）
            self.is_calibration_mode = False
            self.calibration_controls_widget.setVisible(False)
            self._redraw_calibration_points() # Clear points from view
            QMessageBox.information(self, "校准已保存", f"{os.path.basename(self.current_image_path)} 的校准点已保存。")
        elif self.is_calibration_mode:
            print("[Viewer] Error: Need exactly 4 points to save calibration.")
            QMessageBox.warning(self, "校准错误", "需要选择4个角点才能保存。")

    def exit_calibration_mode(self):
        """退出手动校准模式"""
        print("[Viewer] Exiting calibration mode.")
        self.is_calibration_mode = False
        self.calibration_controls_widget.setVisible(False)
        self._redraw_calibration_points() # Clear points from view

    def get_calibration_points(self) -> Optional[List]:
         """获取当前图像的校准点"""
         if len(self.calibration_points) == 4:
             return self.calibration_points
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
            cv2.imwrite(path, img)
            test_image_paths.append(path)

        viewer.load_images(test_image_paths)

    viewer.setWindowTitle("Image Viewer Test")
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec()) 