import sys
import os
import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QProgressBar, QTextEdit, QComboBox,
    QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QSizePolicy, QStyleFactory, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont
import json # For calibration saving/loading

# 导入新的可视化窗口部件
try:
    from app.ui.image_viewer import ImageViewerWidget
except ImportError:
    # 创建一个临时的占位符，以防文件尚未创建
    class ImageViewerWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("可视化窗口占位符")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.setStyleSheet("border: 1px solid gray;")

# 动态添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.core.analysis_runner import AnalysisRunner
from app.utils.logging_handler import setup_logging

class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("杂草覆盖与密度分析工具")
        self.setGeometry(100, 100, 900, 750) # 调整窗口大小

        self.analysis_thread = None
        self.analysis_runner = None
        self.pending_calibrations = [] # List of image paths needing manual calibration
        self.loaded_calibration_points = {} # Dict to store loaded/saved points {img_path: points}
        self.last_run_config = None

        # 默认路径设置
        self.default_input_path = os.path.join(project_root, "input_data")
        self.default_output_path = os.path.join(project_root, "output_data")
        self.default_lidar_path = os.path.join(project_root, "lidar_data")
        
        # 确保默认文件夹存在
        os.makedirs(self.default_input_path, exist_ok=True)
        os.makedirs(self.default_output_path, exist_ok=True)
        os.makedirs(self.default_lidar_path, exist_ok=True)

        # 设置暗色主题 (可选)
        self.apply_stylesheet()

        # 初始化日志系统
        setup_logging(self.log_signal)

        # --- 主布局 ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Use a top-level layout for the main widget to hold the splitter
        top_level_layout = QHBoxLayout(self.main_widget)
        top_level_layout.setContentsMargins(5, 5, 5, 5) # Add some margin around the splitters
        top_level_layout.setSpacing(0)

        # --- Main Horizontal Splitter --- #
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(self.main_splitter)

        # --- 左侧: 配置面板 ---
        self.config_widget = QWidget()
        self.config_layout = QVBoxLayout(self.config_widget)
        self.config_layout.setSpacing(15)
        self.config_layout.setContentsMargins(0, 0, 0, 0) # 移除边距，由父布局控制
        # Add config widget to the main splitter
        self.main_splitter.addWidget(self.config_widget)

        # --- 顶部: 输入和输出选择 ---
        io_group = QGroupBox("输入与输出")
        io_layout = QVBoxLayout()
        io_group.setLayout(io_layout)
        self.config_layout.addWidget(io_group)

        # 输入路径
        input_layout = QHBoxLayout()
        self.input_label = QLabel("输入图片/文件夹:")
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("选择单个图片文件或包含图片的文件夹")
        self.input_path_edit.setText(self.default_input_path)  # 设置默认输入路径
        self.browse_input_button = QPushButton("浏览...")
        self.browse_input_button.clicked.connect(self.browse_input)
        self.browse_input_button.setObjectName("browse_button") # For styling
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_input_button)
        io_layout.addLayout(input_layout)

        # 输出路径
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出文件夹:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择保存结果的文件夹")
        self.output_path_edit.setText(self.default_output_path)  # 设置默认输出路径
        self.browse_output_button = QPushButton("浏览...")
        self.browse_output_button.clicked.connect(self.browse_output)
        self.browse_output_button.setObjectName("browse_button") # For styling
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.browse_output_button)
        io_layout.addLayout(output_layout)

        # --- 新增：校准文件/文件夹路径 ---
        calibration_layout = QHBoxLayout()
        self.calibration_label = QLabel("校准文件/目录:")
        self.calibration_path_edit = QLineEdit()
        self.calibration_path_edit.setPlaceholderText("可选：包含校准 .json 文件的目录或特定文件")
        self.browse_calibration_button = QPushButton("浏览...")
        self.browse_calibration_button.clicked.connect(self.browse_calibration)
        self.browse_calibration_button.setObjectName("browse_button") # For styling
        calibration_layout.addWidget(self.calibration_label)
        calibration_layout.addWidget(self.calibration_path_edit)
        calibration_layout.addWidget(self.browse_calibration_button)
        io_layout.addLayout(calibration_layout)

        # --- 新增：加载结果按钮 ---
        load_results_layout = QHBoxLayout()
        self.load_results_button = QPushButton("加载已有结果")
        self.load_results_button.clicked.connect(self.browse_and_load_results)
        self.load_results_button.setObjectName("load_results_button")
        self.load_results_button.setStyleSheet("background-color: #2a82da; color: white; font-weight: bold;")
        load_results_layout.addStretch()
        load_results_layout.addWidget(self.load_results_button)
        io_layout.addLayout(load_results_layout)

        # --- 调整标签宽度和对齐 (包括新的校准标签) ---
        input_fm = self.input_label.fontMetrics()
        output_fm = self.output_label.fontMetrics()
        calib_fm = self.calibration_label.fontMetrics()
        # Use boundingrect for more accurate width
        input_width = input_fm.boundingRect(self.input_label.text()).width()
        output_width = output_fm.boundingRect(self.output_label.text()).width()
        calib_width = calib_fm.boundingRect(self.calibration_label.text()).width()
        max_label_width = max(input_width, output_width, calib_width) + 10 # Add a small margin

        # 设置固定宽度和右对齐
        self.input_label.setFixedWidth(max_label_width)
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.output_label.setFixedWidth(max_label_width)
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.calibration_label.setFixedWidth(max_label_width)
        self.calibration_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # --- 中部: 分析参数配置 ---
        params_layout = QHBoxLayout()
        self.config_layout.addLayout(params_layout)

        # 左侧参数组
        left_params_group = QGroupBox("分析模型与方法")
        left_params_form_layout = QFormLayout()
        left_params_group.setLayout(left_params_form_layout)
        params_layout.addWidget(left_params_group, 1)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["传统方法", "深度学习"])
        self.model_type_combo.currentIndexChanged.connect(self.update_parameter_visibility)
        left_params_form_layout.addRow("分析模型:", self.model_type_combo)

        # 传统方法参数
        self.segment_method_combo = QComboBox()
        self.segment_method_combo.addItems(['HSV', 'LAB'])
        self.segment_method_label = QLabel("分割方法:")
        left_params_form_layout.addRow(self.segment_method_label, self.segment_method_combo)

        self.hsv_config_label = QLabel("HSV配置:")
        self.hsv_config_layout = QHBoxLayout()
        self.hsv_config_path_edit = QLineEdit()
        self.hsv_config_path_edit.setPlaceholderText("可选的 .json 配置文件")
        self.browse_hsv_config_button = QPushButton("浏览...")
        self.browse_hsv_config_button.clicked.connect(self.browse_hsv_config)
        self.browse_hsv_config_button.setObjectName("browse_button") # For styling
        self.hsv_config_layout.addWidget(self.hsv_config_path_edit)
        self.hsv_config_layout.addWidget(self.browse_hsv_config_button)
        left_params_form_layout.addRow(self.hsv_config_label, self.hsv_config_layout)

        # 深度学习参数
        self.dl_model_label = QLabel("模型路径:")
        self.dl_model_layout = QHBoxLayout()
        self.dl_model_path_edit = QLineEdit()
        self.dl_model_path_edit.setPlaceholderText("选择 .pt 或 .onnx 模型文件")
        self.browse_dl_model_button = QPushButton("浏览...")
        self.browse_dl_model_button.clicked.connect(self.browse_dl_model)
        self.browse_dl_model_button.setObjectName("browse_button") # For styling
        self.dl_model_layout.addWidget(self.dl_model_path_edit)
        self.dl_model_layout.addWidget(self.browse_dl_model_button)
        left_params_form_layout.addRow(self.dl_model_label, self.dl_model_layout)

        self.dl_device_label = QLabel("运行设备:")
        self.dl_device_combo = QComboBox()
        self.dl_device_combo.addItems(["cpu", "cuda" if self.check_cuda() else "cuda (不可用)"])
        self.dl_device_combo.setEnabled(self.check_cuda()) # 仅当 CUDA 可用时启用
        left_params_form_layout.addRow(self.dl_device_label, self.dl_device_combo)

        # 通用参数
        self.do_calibration_checkbox = QCheckBox("执行相机校准")
        self.do_calibration_checkbox.setChecked(True)
        left_params_form_layout.addRow(self.do_calibration_checkbox)

        self.save_debug_checkbox = QCheckBox("保存调试图像")
        self.save_debug_checkbox.setChecked(True)
        left_params_form_layout.addRow(self.save_debug_checkbox)

        self.calculate_density_checkbox = QCheckBox("计算覆盖密度")
        left_params_form_layout.addRow(self.calculate_density_checkbox)

        self.layout_label = QLabel("分析图布局:")
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["default (3x2)", "simple (1x3)"])
        self.layout_combo.setCurrentIndex(0) # Default to 'default'
        left_params_form_layout.addRow(self.layout_label, self.layout_combo)

        # 右侧参数组 (Lidar)
        # 使用一个容器布局来更好地控制 GroupBox 的位置和边距
        right_container_layout = QVBoxLayout()
        # 可以在这里为容器布局设置边距，确保 GroupBox 周围有足够空间
        # 例如，添加一点顶部边距来防止 GroupBox 标题被裁剪
        right_container_layout.setContentsMargins(0, 5, 0, 0) # left, top, right, bottom

        self.right_params_group = QGroupBox("Lidar 高度分析 (可选)")
        self.right_params_group.setCheckable(True) # 使整个组可选
        self.right_params_group.setChecked(False) # 默认不勾选
        self.right_params_group.toggled.connect(self.toggle_lidar_params)
        # Lidar group box indicator will inherit QCheckBox::indicator style

        right_params_layout = QFormLayout()
        # 可以选择性地为 QFormLayout 设置内部边距，调整标签/控件与 GroupBox 边框的距离
        self.right_params_group.setLayout(right_params_layout)

        # 将 GroupBox 添加到容器布局中
        right_container_layout.addWidget(self.right_params_group)
        # 如果不希望 Lidar 部分垂直拉伸填满空间，可以添加一个伸缩项
        # right_container_layout.addStretch(1)

        # 将右侧容器布局添加到主参数布局 (params_layout)
        # 假设 params_layout 是一个 QHBoxLayout，用于左右分割参数区域
        params_layout.addLayout(right_container_layout, 1) # 第三个参数是拉伸因子

        # --- Lidar 参数控件 (现在添加到 right_params_layout) ---
        self.lidar_dir_label = QLabel("Lidar数据文件夹:")
        self.lidar_dir_layout = QHBoxLayout()
        self.lidar_dir_edit = QLineEdit()
        self.lidar_dir_edit.setPlaceholderText("包含 .pcd 文件的文件夹")
        self.lidar_dir_edit.setText(self.default_lidar_path)  # 设置默认Lidar路径
        self.browse_lidar_button = QPushButton("浏览...")
        self.browse_lidar_button.clicked.connect(self.browse_lidar_dir)
        self.browse_lidar_button.setObjectName("browse_button") # For styling
        self.lidar_dir_layout.addWidget(self.lidar_dir_edit)
        self.lidar_dir_layout.addWidget(self.browse_lidar_button)
        right_params_layout.addRow(self.lidar_dir_label, self.lidar_dir_layout)

        self.dbscan_eps_label = QLabel("DBSCAN邻域半径(eps):")
        self.dbscan_eps_spinbox = QDoubleSpinBox()
        self.dbscan_eps_spinbox.setRange(0.01, 10.0)
        self.dbscan_eps_spinbox.setSingleStep(0.05)
        self.dbscan_eps_spinbox.setValue(0.3) # Default from main.py
        self.dbscan_eps_spinbox.setDecimals(3) # More precision if needed
        right_params_layout.addRow(self.dbscan_eps_label, self.dbscan_eps_spinbox)

        self.dbscan_min_samples_label = QLabel("DBSCAN最小样本数(min_samples):")
        self.dbscan_min_samples_spinbox = QSpinBox()
        self.dbscan_min_samples_spinbox.setRange(1, 100)
        self.dbscan_min_samples_spinbox.setValue(2) # Default from main.py
        right_params_layout.addRow(self.dbscan_min_samples_label, self.dbscan_min_samples_spinbox)

        # 初始时禁用 Lidar 参数
        self.toggle_lidar_params(False)
        # 根据模型类型更新参数可见性
        self.update_parameter_visibility()

        # --- 底部: 进度条、日志和控制按钮 ---
        bottom_layout = QVBoxLayout()
        self.config_layout.addLayout(bottom_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("进度: %p%")
        bottom_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 10)) # 使用等宽字体
        # 设置日志区域最小高度，并允许垂直扩展
        self.log_edit.setMinimumHeight(200)
        self.log_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bottom_layout.addWidget(self.log_edit)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(10) # Add some space between buttons

        self.start_button = QPushButton("开始分析")
        self.start_button.setObjectName("start_button") # Set object name for specific styling
        self.start_button.clicked.connect(self.start_analysis)

        self.stop_button = QPushButton("停止分析")
        self.stop_button.setObjectName("stop_button") # Set object name for specific styling
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)

        # Make buttons expand to fill the space equally
        control_layout.addWidget(self.start_button, 1) # Stretch factor 1
        control_layout.addWidget(self.stop_button, 1)  # Stretch factor 1

        bottom_layout.addLayout(control_layout)

        # --- 右侧: 可视化窗口 ---
        self.image_viewer = ImageViewerWidget()
        # Connect signals BEFORE adding to layout potentially
        self.image_viewer.calibration_save_requested.connect(self.on_calibration_saved)

        # --- 直接将 ImageViewer 添加到主 Splitter --- #
        self.main_splitter.addWidget(self.image_viewer) # Add the entire viewer widget

        # --- Set Initial Splitter Sizes --- #
        # Adjust main splitter sizes
        total_width = self.sizeHint().width() # Use sizeHint for initial estimate
        config_width = int(total_width * 0.30) # Config panel takes ~30%
        viewer_width = total_width - config_width

        self.main_splitter.setSizes([config_width, viewer_width])

        # Viewer internal splitter sizes are handled within ImageViewerWidget if it uses one,
        # or its internal layout stretch factors.
        # self.viewer_splitter.setSizes([preview_width, graphics_view_width]) # Removed

        # Optional: Set stretch factors for the main splitter
        self.main_splitter.setStretchFactor(0, 0) # Config panel doesn't stretch as much
        self.main_splitter.setStretchFactor(1, 1) # Viewer panel takes remaining space

        # Hide config panel if size becomes very small (optional)
        # self.main_splitter.splitterMoved.connect(self.handle_splitter_move)

        # 连接日志信号 (已移到更早的位置)
        # self.log_signal.connect(self.append_log)
        # 连接校准保存信号 (已移到更早的位置)
        # self.image_viewer.calibration_save_requested.connect(self.on_calibration_saved)

    def apply_stylesheet(self):
        """设置应用的样式表 (深色主题)"""
        # Fusion style is generally good with palettes
        QApplication.setStyle(QStyleFactory.create('Fusion'))

        # 应用调色板
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        # 设置禁用的文本颜色
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))

        QApplication.setPalette(dark_palette)

        # Custom Stylesheet Additions
        # (You can customize colors, padding, borders etc. here)
        self.setStyleSheet("""
            QWidget {
                font-size: 10pt; /* Slightly larger default font */
            }
            QGroupBox {
                border: 1px solid #666; /* Slightly lighter border */
                border-radius: 5px;
                margin-top: 1ex; /* leave space at the top for the title */
                padding: 10px 5px 5px 5px; /* top, right, bottom, left */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                left: 10px; /* Align title slightly from the left edge */
            }
            QPushButton {
                min-height: 16px; /* Ensure buttons are not too small */
                padding: 5px 15px; /* Increased padding */
                border-radius: 4px; /* Slightly more rounded */
                /* background-color gets set by palette */
            }
            QPushButton:checked {
                background-color: #6a6a6a; /* Darker when checked (for view buttons) */
                border: 1px solid #888;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox {
                min-height: 25px;
                border-radius: 3px;
                padding: 1px 5px;
            }
            QListWidget {
                border-radius: 0px;
            }
            QSplitter::handle {
                background-color: #444;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
            /* Less prominent Start/Stop buttons */
            QPushButton#start_button {
                background-color: #4682B4; /* Steel Blue */
                color: white;
                min-height: 24px; /* Further reduced height */
                padding: 5px 15px;
            }
            QPushButton#start_button:hover { background-color: #5A9BD5; }
            QPushButton#start_button:pressed { background-color: #3E70A0; }
            QPushButton#start_button:disabled { background-color: #5A5A5A; color: #999; }

            QPushButton#stop_button {
                background-color: #C85C5C; /* Soft Red */
                color: white;
                min-height: 28px; /* Further reduced height */
                padding: 5px 15px;
            }
            QPushButton#stop_button:hover { background-color: #DA7F7F; }
            QPushButton#stop_button:pressed { background-color: #B84B4B; }
            QPushButton#stop_button:disabled { background-color: #5A5A5A; color: #999; }

            /* Nicer CheckBox */
            QCheckBox {
                spacing: 8px; /* Space between indicator and text */
            }
            QCheckBox::indicator {
                width: 12px; /* Slightly smaller */
                height: 12px; /* Slightly smaller */
                border-radius: 4px;
                border: 1px solid #888;
                background-color: #444; /* Slightly lighter than base for visibility */
            }
            QCheckBox::indicator:unchecked:hover {
                border: 1px solid #aaa;
            }
            QCheckBox::indicator:checked {
                background-color: #2a82da; /* Check color */
                border: 1px solid #55aaff;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #55aaff;
                border: 1px solid #77ccff;
            }
            QGroupBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 4px;
                border: 1px solid #888;
                background-color: #444;
            }
            QGroupBox::indicator:unchecked:hover {
                border: 1px solid #aaa;
            }
            QGroupBox::indicator:checked {
                background-color: #2a82da;
                border: 1px solid #55aaff;
            }
            QGroupBox::indicator:checked:hover {
                background-color: #55aaff;
                border: 1px solid #77ccff;
            }
            QCheckBox:disabled {
                color: #888;
            }
            QCheckBox::indicator:disabled {
                border: 1px solid #555;
                background-color: #333;
            }
            QGroupBox::indicator:disabled {
                border: 1px solid #555;
                background-color: #333;
            }
            /* Style for Browse buttons */
            QPushButton#browse_button {
                border: 1px solid #666;
                padding: 3px 8px; /* Smaller padding */
                min-height: 16px; /* Match other inputs */
                background-color: #555;
            }
            QPushButton#browse_button:hover { background-color: #666; }
            QPushButton#browse_button:pressed { background-color: #777; }

            /* Preview List selection style */
            QListWidget::item:selected {
                 background-color: rgba(66, 135, 245, 100); /* Semi-transparent light blue */
                 color: white; /* Ensure text stays white */
                 border: 1px solid #55aaff; /* Add a border for clarity */
                 border-radius: 3px;
            }
            QListWidget::item:selected:!active {
                 background-color: rgba(66, 135, 245, 80); /* Slightly less prominent when not active */
            }
        """)

    def _handle_input_path_selection(self, selected_path_or_files):
        """Handles logic after input path/files are selected via browse."""
        input_paths = []
        display_text = ""

        if isinstance(selected_path_or_files, list): # Multiple files selected
             input_paths = [p for p in selected_path_or_files if os.path.isfile(p)]
             display_text = ";".join(input_paths)
        elif isinstance(selected_path_or_files, str): # Single path (file or dir)
             path = selected_path_or_files
             display_text = path
             if os.path.isdir(path):
                 supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                 try:
                     input_paths = [os.path.join(path, fname) for fname in os.listdir(path) if fname.lower().endswith(supported_exts)]
                 except OSError as e:
                     QMessageBox.critical(self, "文件错误", f"无法读取文件夹 '{path}': {e}")
                     input_paths = [] # Reset paths on error
             elif os.path.isfile(path):
                 supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                 if path.lower().endswith(supported_exts):
                      input_paths = [path]
                 else:
                      QMessageBox.warning(self, "文件类型错误", "选择的文件不是支持的图像格式。")
                      input_paths = []
             else:
                 QMessageBox.warning(self, "路径无效", "选择的路径不是有效的文件或文件夹。")
                 input_paths = []
        else:
             print("[UI] Invalid selection type received.")
             input_paths = []

        # Update text edit regardless of image loading success
        self.input_path_edit.setText(display_text)

        # Normalize paths before loading
        try:
            normalized_paths = [os.path.normpath(p) for p in input_paths]
        except Exception as e:
             QMessageBox.critical(self, "路径错误", f"规范化输入路径时出错: {e}")
             normalized_paths = []

        # Load images into viewer
        if normalized_paths:
             print(f"[UI] Loading {len(normalized_paths)} images into viewer.")
             self.image_viewer.load_images(normalized_paths)
             # Set calibration save dir based on input (if it's a directory)
             if isinstance(selected_path_or_files, str) and os.path.isdir(selected_path_or_files):
                  self.image_viewer.set_calibration_save_dir(selected_path_or_files)
             elif input_paths: # If files selected, maybe use parent dir of first file?
                  self.image_viewer.set_calibration_save_dir(os.path.dirname(input_paths[0]))
        else:
             print("[UI] No valid images found or error occurred, clearing viewer.")
             self.image_viewer.load_images([]) # Clear viewer

    def browse_input(self):
        dialog = QFileDialog(self)
        # Start by allowing selection of files or a directory
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        # 以默认输入路径为起始目录
        starting_dir = self.input_path_edit.text() if os.path.exists(self.input_path_edit.text()) else self.default_input_path

        # Try getting a directory first
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", starting_dir, options=QFileDialog.Option.ShowDirsOnly)
        if dir_path:
            self._handle_input_path_selection(dir_path)
            return

        # If no directory was selected, try getting file(s)
        files, _ = QFileDialog.getOpenFileNames(self, "选择一个或多个图片文件", starting_dir,
                                            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)")
        if files:
            # Handle single or multiple file selection
            self._handle_input_path_selection(files)

    def browse_output(self):
        starting_dir = self.output_path_edit.text() if os.path.exists(self.output_path_edit.text()) else self.default_output_path
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", starting_dir, QFileDialog.Option.ShowDirsOnly)
        if path:
            self.output_path_edit.setText(path)

    def browse_hsv_config(self):
        starting_dir = os.path.dirname(self.hsv_config_path_edit.text()) if os.path.exists(os.path.dirname(self.hsv_config_path_edit.text())) else self.default_input_path
        path, _ = QFileDialog.getOpenFileName(self, "选择 HSV 配置文件", starting_dir, "JSON 文件 (*.json)")
        if path:
            self.hsv_config_path_edit.setText(path)

    def browse_dl_model(self):
        starting_dir = os.path.dirname(self.dl_model_path_edit.text()) if os.path.exists(os.path.dirname(self.dl_model_path_edit.text())) else self.default_input_path
        path, _ = QFileDialog.getOpenFileName(self, "选择深度学习模型文件", starting_dir,
                                            "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)")
        if path:
            self.dl_model_path_edit.setText(path)

    def browse_lidar_dir(self):
        starting_dir = self.lidar_dir_edit.text() if os.path.exists(self.lidar_dir_edit.text()) else self.default_lidar_path
        path = QFileDialog.getExistingDirectory(self, "选择 Lidar 数据文件夹", starting_dir, QFileDialog.Option.ShowDirsOnly)
        if path:
            self.lidar_dir_edit.setText(path)

    def browse_calibration(self):
        """浏览校准文件或目录"""
        dialog = QFileDialog(self)
        # 定义起始目录
        starting_dir = self.calibration_path_edit.text() if os.path.exists(self.calibration_path_edit.text()) else self.default_input_path
        # 允许选择文件或目录
        path = QFileDialog.getExistingDirectory(self, "选择校准目录", starting_dir, QFileDialog.Option.ShowDirsOnly)
        if path:
            self.calibration_path_edit.setText(path)
            return
        # 如果没选目录，尝试选文件
        file, _ = QFileDialog.getOpenFileName(self, "选择校准文件", starting_dir, "JSON 文件 (*.json)")
        if file:
             self.calibration_path_edit.setText(file)

    def update_parameter_visibility(self):
        is_dl_model = self.model_type_combo.currentText() == "深度学习"

        # 传统方法参数
        self.segment_method_label.setVisible(not is_dl_model)
        self.segment_method_combo.setVisible(not is_dl_model)
        self.hsv_config_label.setVisible(not is_dl_model)
        self.hsv_config_layout.itemAt(0).widget().setVisible(not is_dl_model)
        self.hsv_config_layout.itemAt(1).widget().setVisible(not is_dl_model)

        # 深度学习参数
        self.dl_model_label.setVisible(is_dl_model)
        self.dl_model_layout.itemAt(0).widget().setVisible(is_dl_model)
        self.dl_model_layout.itemAt(1).widget().setVisible(is_dl_model)
        self.dl_device_label.setVisible(is_dl_model)
        self.dl_device_combo.setVisible(is_dl_model)

    def toggle_lidar_params(self, checked):
        """启用或禁用 Lidar 参数输入"""
        self.lidar_dir_label.setEnabled(checked)
        self.lidar_dir_layout.itemAt(0).widget().setEnabled(checked)
        self.lidar_dir_layout.itemAt(1).widget().setEnabled(checked)
        self.dbscan_eps_label.setEnabled(checked)
        self.dbscan_eps_spinbox.setEnabled(checked)
        self.dbscan_min_samples_label.setEnabled(checked)
        self.dbscan_min_samples_spinbox.setEnabled(checked)

    def check_cuda(self):
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_config(self) -> Optional[dict]:
        """收集所有配置参数."""
        input_path = self.input_path_edit.text().strip()
        output_dir = self.output_path_edit.text().strip()

        if not input_path or not output_dir:
            QMessageBox.warning(self, "缺少输入", "请输入有效的输入图片/文件夹和输出文件夹路径。")
            return None

        # 检查输入路径是否存在
        if not os.path.exists(input_path):
             QMessageBox.warning(self, "路径无效", f"输入路径不存在: {input_path}")
             return None

        # 尝试创建输出目录
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "创建目录失败", f"无法创建输出目录: {output_dir}\n错误: {e}")
            return None

        # 获取模型类型
        model_type_text = self.model_type_combo.currentText()
        model_type = 'dl' if '深度学习' in model_type_text else 'traditional'

        # 获取绘图布局
        layout_text = self.layout_combo.currentText() # 使用新的下拉框
        plot_layout = 'simple' if 'simple' in layout_text else 'default'

        config = {
            "input_path": input_path,
            "output_dir": output_dir,
            "calibration_path": self.calibration_path_edit.text().strip() or None,
            "model_type": model_type,
            "do_calibration": self.do_calibration_checkbox.isChecked(),
            "save_debug_images": self.save_debug_checkbox.isChecked(),
            "calculate_density": self.calculate_density_checkbox.isChecked(),
            "plot_layout": plot_layout, # 添加新的布局字符串
            "perform_lidar_analysis": self.right_params_group.isChecked(),
            # Lidar specific parameters (only relevant if perform_lidar_analysis is True)
            "lidar_dir": self.lidar_dir_edit.text().strip() or None,
            "dbscan_eps": self.dbscan_eps_spinbox.value(), # 添加DBSCAN eps
            "dbscan_min_samples": self.dbscan_min_samples_spinbox.value(), # 添加DBSCAN min_samples
        }

        # 添加模型特定参数
        if model_type == 'traditional':
            config["segment_method"] = self.segment_method_combo.currentText().lower()
            config["hsv_config_path"] = self.hsv_config_path_edit.text().strip() or None
        else: # dl model
            config["dl_model_path"] = self.dl_model_path_edit.text().strip() or None
            config["dl_device"] = self.dl_device_combo.currentText().split(" ")[0] # Get 'cpu' or 'cuda'

        # 验证 Lidar 目录（如果启用）
        if config["perform_lidar_analysis"] and not config["lidar_dir"]:
             QMessageBox.warning(self, "缺少 Lidar 目录", "请在启用 Lidar 分析时提供 Lidar 数据文件夹路径。")
             return None
        if config["perform_lidar_analysis"] and not os.path.isdir(config["lidar_dir"]):
             QMessageBox.warning(self, "无效 Lidar 目录", f"Lidar 数据文件夹不存在或不是有效目录: {config['lidar_dir']}")
             return None

        logging.info(f"获取到的配置: {config}")
        return config

    def get_calibration_file_path(self, image_path):
        """
        获取给定图像对应的校准文件路径
        将校准文件保存在图片所在目录下的 calibrations 文件夹中
        """
        # 获取图片所在目录
        image_dir = os.path.dirname(image_path)
        # 在图片目录下创建 calibrations 文件夹
        calib_dir = os.path.join(image_dir, 'calibrations')
        os.makedirs(calib_dir, exist_ok=True)
            
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(calib_dir, f"{img_basename}.json")

    def start_analysis(self):
        config = self.get_config()
        if config is None:
            return # 配置无效

        # -- Clear previous state --
        self.pending_calibrations = []
        self.loaded_calibration_points = {}
        self.log_edit.clear() # Clear previous logs
        self.progress_bar.setValue(0)

        if self.analysis_thread and self.analysis_thread.isRunning():
            QMessageBox.warning(self, "正在运行", "分析任务已在运行中。")
            return

        logging.info("获取配置。")

        # --- Calibration Handling ---
        do_calib = config.get('do_calibration', False)
        calib_path_input = config.get('calibration_path')

        # --- 确保输入路径列表的处理 --- #
        potential_paths = config['input_path']
        actual_image_paths_list = []
        if isinstance(potential_paths, str):
            if os.path.isfile(potential_paths):
                actual_image_paths_list = [potential_paths]
            elif os.path.isdir(potential_paths):
                supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                try:
                    all_files = os.listdir(potential_paths)
                    actual_image_paths_list = sorted([
                        os.path.join(potential_paths, fname) 
                        for fname in all_files 
                        if fname.lower().endswith(supported_exts)
                    ])
                except OSError as e:
                    QMessageBox.critical(self, "文件错误", f"无法读取输入文件夹 '{potential_paths}' 进行校准检查: {e}")
            else:
                QMessageBox.warning(self, "路径无效", f"输入框中的路径无效: {potential_paths}")
        elif isinstance(potential_paths, list):
            actual_image_paths_list = sorted(potential_paths)
        else:
            logging.error(f"配置中的 input_path 类型未知: {type(potential_paths)}。无法检查校准。")

        if not actual_image_paths_list:
            QMessageBox.warning(self, "无有效图像", "未找到有效的输入图像文件。")
            self.set_inputs_enabled(True)
            return

        config['input_paths'] = actual_image_paths_list
        if 'input_path' in config:
            del config['input_path']

        self.last_run_config = config.copy()

        # --- 优化校准文件处理 ---
        if do_calib:
            logging.info("检查校准文件...")
            
            # 检查每个图像的校准状态
            for img_path in actual_image_paths_list:
                calib_file = self.get_calibration_file_path(img_path)
                if os.path.exists(calib_file):
                    try:
                        with open(calib_file, 'r') as f:
                            points = json.load(f)
                            if isinstance(points, list) and len(points) == 4:
                                self.loaded_calibration_points[img_path] = points
                                logging.info(f"已加载校准点 ({os.path.basename(img_path)}): {points}")
                            else:
                                logging.warning(f"校准文件 {os.path.basename(calib_file)} 格式无效")
                                self.pending_calibrations.append(img_path)
                    except Exception as e:
                        logging.error(f"加载校准文件 {os.path.basename(calib_file)} 出错: {e}")
                        self.pending_calibrations.append(img_path)
                else:
                    logging.info(f"图像 {os.path.basename(img_path)} 需要校准")
                    self.pending_calibrations.append(img_path)

            if self.pending_calibrations:
                self.start_button.setEnabled(False)
                self.set_inputs_enabled(False)
                
                # 加载所有图像到查看器
                self.image_viewer.load_images(actual_image_paths_list)
                
                total_pending = len(self.pending_calibrations)
                QMessageBox.information(
                    self, 
                    "需要校准",
                    f"需要为 {total_pending} 张图像进行校准。\n"
                    f"请在右侧窗口为每张图像选择1平方米区域的4个角点。\n"
                    f'完成后点击"保存校准"按钮。'
                )
                
                # 开始第一张图片的校准
                first_image = self.pending_calibrations[0]
                # 设置校准保存目录为图片所在目录
                self.image_viewer.set_calibration_save_dir(os.path.dirname(first_image))
                self.image_viewer._set_view_mode(1, target_image_path=first_image)
                return

        # 如果不需要校准或所有校准已完成
        self.image_viewer.load_images(actual_image_paths_list)
        self._proceed_with_analysis(self.last_run_config)

    def _proceed_with_analysis(self, config):
        """实际启动后台分析线程"""
        logging.info("准备开始后台分析...")
        logging.info(f"配置详情: {config}")
        logging.info(f"使用的校准点: {self.loaded_calibration_points}")

        # Add loaded calibration points to the config for the runner
        config['calibration_data'] = self.loaded_calibration_points.copy()

        # Store config used for this run to find results later
        # self.last_run_config = config.copy() # This is now done in start_analysis

        # 创建分析器和线程
        self.analysis_runner = AnalysisRunner(config)
        self.analysis_thread = QThread()
        self.analysis_runner.moveToThread(self.analysis_thread)

        # 连接信号槽
        self.analysis_runner.progress_updated.connect(self.update_progress)
        self.analysis_runner.log_message.connect(self.append_log)
        self.analysis_runner.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.started.connect(self.analysis_runner.run)
        # 清理线程
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        self.analysis_runner.analysis_complete.connect(self.analysis_thread.quit)

        # 禁用开始按钮，启用停止按钮
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_inputs_enabled(False)

        # 启动线程
        self.analysis_thread.start()

    def stop_analysis(self):
        if self.analysis_runner:
            logging.info("正在请求停止分析...")
            self.analysis_runner.stop() # 请求 AnalysisRunner 停止
        # 不立即启用/禁用按钮，等待 analysis_complete 信号
        self.stop_button.setEnabled(False)
        self.stop_button.setText("正在停止...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_log(self, message):
        self.log_edit.append(message.strip()) # 追加文本并自动滚动
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())
        QApplication.processEvents() # 强制 UI 刷新以显示日志

    def load_results_from_summary(self, output_dir):
        """从analysis_summary.json文件中加载所有结果图像"""
        try:
            # 规范化输出目录路径
            output_dir = os.path.normpath(output_dir)
            summary_file = "analysis_summary.json"
            summary_file_path = os.path.join(output_dir, summary_file)
            
            # 读取摘要文件
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # 获取所有图像路径
            image_results = summary_data.get("image_results", [])
            if not image_results:
                self.append_log("分析摘要中未找到图像结果")
                return False
            
            self.append_log(f"找到 {len(image_results)} 个结果文件")
            
            # 按图片序号分组
            image_groups = {}  # 序号 -> {原始图片路径, 结果图片字典}
            
            # 第一遍：识别所有原始图像
            for path in image_results:
                # 确保路径是绝对路径
                if not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(output_dir, path))
                else:
                    path = os.path.normpath(path)
                
                if not os.path.exists(path):
                    self.append_log(f"警告: 图像文件不存在: {path}")
                    continue
                
                filename = os.path.basename(path)
                
                # 检查是否是原始图像 (没有特殊后缀的图像)
                is_original = True
                for suffix in ['_debug.png', '_analysis.png', 'traditional_default_analysis.png']:
                    if suffix in filename:
                        is_original = False
                        break
                
                if is_original:
                    # 提取序号 (可能是 "01.jpeg" 或 "01_xxx.jpg" 格式)
                    file_base = os.path.splitext(filename)[0]  # 去掉扩展名
                    prefix = file_base.split('_')[0]  # 取第一部分作为序号
                    
                    # 检查是否是数字
                    if prefix.isdigit():
                        if prefix not in image_groups:
                            image_groups[prefix] = {'original': None, 'results': {}}
                        
                        image_groups[prefix]['original'] = path
                        self.append_log(f"添加原始图像: {filename} (序号: {prefix})")
            
            # 第二遍：处理所有结果图像
            for path in image_results:
                # 确保路径是绝对路径
                if not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(output_dir, path))
                else:
                    path = os.path.normpath(path)
                
                if not os.path.exists(path):
                    continue
                
                filename = os.path.basename(path)
                
                # 跳过原始图像
                is_result = False
                for suffix in ['_debug.png', '_analysis.png', 'traditional_default_analysis.png']:
                    if suffix in filename:
                        is_result = True
                        break
                
                if not is_result:
                    continue
                
                # 提取序号
                prefix = filename.split('_')[0]
                if not prefix.isdigit():
                    self.append_log(f"警告: 无法从结果图像名称提取序号: {filename}")
                    continue
                
                # 确保序号组存在
                if prefix not in image_groups:
                    image_groups[prefix] = {'original': None, 'results': {}}
                
                # 根据文件名确定图片类型
                if 'traditional_default_analysis.png' in filename:
                    image_groups[prefix]['results']['traditional_default_analysis'] = path
                    self.append_log(f"添加传统分析结果: {filename}")
                elif '_analysis.png' in filename:
                    image_groups[prefix]['results']['analysis_image'] = path
                    self.append_log(f"添加雷达分析结果: {filename}")
                elif '_debug.png' in filename:
                    # 提取debug类型（例如：从 xxx_hsv_mask_debug.png 提取 hsv_mask）
                    debug_parts = filename.split('_')
                    if len(debug_parts) >= 3:
                        debug_type = '_'.join(debug_parts[1:-1])
                        result_type = f'{debug_type}_debug_image'
                        image_groups[prefix]['results'][result_type] = path
                        self.append_log(f"添加调试图像: {filename} -> {result_type}")
            
            # 准备加载的原始图片列表
            original_images = []
            valid_groups = {}
            
            # 第三遍：验证和整理数据
            for prefix in sorted(image_groups.keys()):  # 按序号排序
                group = image_groups[prefix]
                if group['original'] and os.path.exists(group['original']):
                    original_images.append(group['original'])
                    if group['results']:  # 只有当有结果图片时才添加到有效组
                        valid_groups[group['original']] = group['results']
                        self.append_log(f"有效组 {prefix}: 原始图像 {os.path.basename(group['original'])}, "
                                     f"结果类型: {list(group['results'].keys())}")
            
            # 加载原始图像
            if original_images:
                self.append_log(f"开始加载 {len(original_images)} 个原始图像")
                self.image_viewer.load_images(original_images)
                
                # 为每个原始图像设置结果数据
                for orig_path, results in valid_groups.items():
                    self.append_log(f"设置结果数据: {os.path.basename(orig_path)} -> {list(results.keys())}")
                    self.image_viewer.set_result_data(orig_path, results)
                
                return True
            else:
                self.append_log("未找到任何原始图像")
                return False
                
        except Exception as e:
            self.append_log(f"加载分析摘要时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def on_analysis_complete(self, success, message):
        # logging.info(f"分析完成: success={success}, message='{message}'") # 移除原始详细日志
        completion_status = "成功完成" if success else "失败"
        self.progress_bar.setValue(100 if success else self.progress_bar.value())
        # --- 修改：只记录状态，不记录完整 message（JSON） ---
        self.append_log(f"\n--- 分析{completion_status} --- ")
        # --- 结束修改 ---

        # 恢复按钮状态和输入
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("停止分析")
        self.set_inputs_enabled(True)

        # 清理引用
        self.analysis_thread = None
        self.analysis_runner = None

        # After analysis, regardless of success/fail, re-enable inputs
        self.set_inputs_enabled(True) # Make sure inputs are re-enabled

        self.image_viewer.clear_results() # Clear previous results first

        if success:
            # 加载分析摘要文件中的所有结果图像
            if self.last_run_config and 'output_dir' in self.last_run_config:
                output_dir = self.last_run_config['output_dir']
                self.load_results_from_summary(output_dir)
            
            # QMessageBox.information(self, "完成", message) # 改为最后显示摘要信息
            results_list = []
            try:
                # --- 修改：解析来自 AnalysisRunner 的 message ---
                results_list = json.loads(message)
                if not isinstance(results_list, list):
                    logging.error(f"分析结果格式错误：期望列表，但收到 {type(results_list)}。Message: {message}")
                    results_list = [] # 重置为空列表以避免后续错误
            except json.JSONDecodeError:
                logging.error(f"无法解析来自 AnalysisRunner 的分析结果 JSON: {message}")
                QMessageBox.critical(self, "错误", "分析完成，但无法解析结果摘要。")
                results_list = [] # 重置为空列表
            except Exception as e:
                 logging.error(f"处理分析结果时发生意外错误: {e}\\nMessage: {message}")
                 QMessageBox.critical(self, "错误", f"处理分析结果时发生意外错误: {e}")
                 results_list = [] # 重置为空列表

            # --- 使用解析后的 results_list ---
            if results_list:
                logging.info(f"从 AnalysisRunner 加载 {len(results_list)} 个结果摘要...")
                found_any_results = False
                summary_message = "分析完成:\\n" # 用于最终弹窗的消息

                for i, image_result in enumerate(results_list):
                    # --- 修改：传递整个 image_result 字典 --- #
                    original_path = image_result.get('original_path') # 确保 AnalysisRunner 添加了这个键
                    if not original_path or not os.path.exists(original_path):
                         logging.warning(f"结果条目缺少有效的 original_path 或文件不存在: {original_path}. 跳过...")
                         continue

                    # 构建简单的摘要文本
                    summary_message += f"\\n图像 {i+1}: {os.path.basename(original_path)}\\n"
                    summary_message += f"  盖度: {image_result.get('草地盖度', 'N/A')}\\n"
                    summary_message += f"  密度: {image_result.get('草地密度', 'N/A')}\\n"
                    summary_message += f"  高度: {image_result.get('草地高度', 'N/A')}\\n"
                    result_img_path = image_result.get('结果图路径') # 键名来自 AnalysisRunner
                    if result_img_path and os.path.exists(result_img_path):
                        summary_message += f"  结果图: {os.path.basename(result_img_path)}\\n"
                    else:
                        summary_message += f"  结果图: 未生成或未找到\\n"


                    self.image_viewer.set_result_data(original_path, image_result)
                    found_any_results = True # 标记至少找到一个条目
                    # --- 结束修改 ---\

                if found_any_results:
                     QMessageBox.information(self, "完成", summary_message)
                else:
                     logging.warning("分析成功，但结果摘要中未找到包含有效 original_path 的条目。")
                     QMessageBox.warning(self, "警告", "分析成功，但无法加载任何结果进行显示。")
            else:
                # 如果 results_list 为空但 success 为 True (理论上不应发生，除非 Runner 返回空的 '[]')
                logging.warning("分析成功，但未收到任何结果摘要。")
                QMessageBox.information(self, "完成", "分析已成功完成，但没有可显示的结果。")
        else:
            QMessageBox.critical(self, "错误", message)

        # --- BEGIN Bug Fix 1: Refresh viewer after analysis (保持不变) ---
        # Ensure the correct view mode button is checked
        self.image_viewer.view_original_button.setChecked(True)

        # Reload the currently selected image in the preview list
        current_item = self.image_viewer.preview_list.currentItem()
        if current_item:
            current_path = current_item.data(Qt.ItemDataRole.UserRole)
            if current_path:
                logging.info(f"分析完成，重新加载当前图像: {os.path.basename(current_path)}")
                # Force display, potentially resetting view mode handled internally
                self.image_viewer.display_image(current_path)
            else:
                logging.warning("分析完成，但无法从当前预览项获取路径。")
        else:
            logging.info("分析完成，预览列表中没有选定项。")
        # --- END Bug Fix 1 ---

        # Ensure viewer widget itself is enabled
        self.image_viewer.setEnabled(True)

    def set_inputs_enabled(self, enabled):
        """启用或禁用所有输入控件"""
        # 输入输出
        self.input_path_edit.setEnabled(enabled)
        self.browse_input_button.setEnabled(enabled)
        self.output_path_edit.setEnabled(True)
        self.browse_output_button.setEnabled(True)
        # 参数
        self.model_type_combo.setEnabled(enabled)
        self.segment_method_combo.setEnabled(enabled)
        self.hsv_config_path_edit.setEnabled(enabled)
        self.browse_hsv_config_button.setEnabled(enabled)
        self.dl_model_path_edit.setEnabled(enabled)
        self.browse_dl_model_button.setEnabled(enabled)
        self.dl_device_combo.setEnabled(enabled and self.check_cuda())
        self.do_calibration_checkbox.setEnabled(enabled)
        self.save_debug_checkbox.setEnabled(enabled)
        self.calculate_density_checkbox.setEnabled(enabled)
        self.layout_combo.setEnabled(enabled)
        self.right_params_group.setEnabled(enabled)
        # Lidar (只有在 group 启用且主开关启用时才启用)
        lidar_enabled = enabled and self.right_params_group.isChecked()
        self.toggle_lidar_params(lidar_enabled)
        # 确保 groupbox 本身可以切换
        if enabled: # 只有在分析结束时才恢复勾选框本身的可编辑性
            self.right_params_group.setEnabled(True)
        self.calibration_path_edit.setEnabled(enabled) # 控制校准路径输入框
        self.browse_calibration_button.setEnabled(enabled)

    def on_calibration_saved(self, image_path, points):
        """处理校准点保存事件"""
        if not image_path or image_path not in self.pending_calibrations:
            logging.warning(f"收到未知图像的校准保存请求: {image_path}")
            return

        try:
            # 保存校准点到文件
            calib_file = self.get_calibration_file_path(image_path)
            os.makedirs(os.path.dirname(calib_file), exist_ok=True)
            with open(calib_file, 'w') as f:
                json.dump(points, f)
            logging.info(f"已保存校准点到: {os.path.basename(calib_file)}")
            
            # 更新内存中的校准点
            self.loaded_calibration_points[image_path] = points
            self.pending_calibrations.remove(image_path)
            
            # 通知用户校准已保存
            QMessageBox.information(self, "校准已保存", f"{os.path.basename(image_path)} 的校准点已保存。")
            
            if not self.pending_calibrations:
                # 所有校准完成
                logging.info("所有图像校准完成")
                QMessageBox.information(self, "校准完成", "所有图像校准已完成，即将开始分析。")
                self.image_viewer.exit_calibration_mode()
                
                if self.last_run_config:
                    self._proceed_with_analysis(self.last_run_config)
                else:
                    QMessageBox.critical(self, "错误", "无法在校准后找到原始配置，分析取消。")
                    logging.error("校准后配置丢失")
                    self.set_inputs_enabled(True)
                    self.start_button.setEnabled(True)
            else:
                # 继续下一张图片的校准
                next_image = self.pending_calibrations[0]
                remaining = len(self.pending_calibrations)
                total = len(self.loaded_calibration_points) + remaining
                
                # 更新校准保存目录为下一张图片所在目录
                self.image_viewer.set_calibration_save_dir(os.path.dirname(next_image))
                
                QMessageBox.information(
                    self, 
                    "继续校准", 
                    f"已完成 {total - remaining}/{total} 张图像的校准\n"
                    f"请继续校准: {os.path.basename(next_image)}"
                )
                
                # 确保下一张图片被正确加载
                self.image_viewer._set_view_mode(1, target_image_path=next_image)
                
        except Exception as e:
            logging.error(f"保存校准点时出错: {e}")
            QMessageBox.critical(self, "保存失败", f"保存校准点时出错: {e}")

    def closeEvent(self, event):
        """关闭窗口前检查是否有正在运行的分析"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(self,
                                       "确认退出",
                                       "分析任务仍在运行中，确定要退出吗？",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_analysis() # 尝试停止
                # 可以考虑等待一小段时间让线程结束，或者直接退出
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _set_view_mode(self, mode_id: int, target_image_path: Optional[str] = None):
        """
        设置当前视图模式
        mode_id: 0=原始, 1=校准, 2=结果
        target_image_path: 可选，指定要显示的图像路径
        """
        if mode_id == 1 and target_image_path:
            # 进入校准模式并指定了目标图片
            self.image_viewer._set_view_mode(mode_id, target_image_path=target_image_path)
        else:
            # 其他模式切换
            self.image_viewer._set_view_mode(mode_id)

    def browse_and_load_results(self):
        """浏览并加载已有的分析结果"""
        try:
            # 从当前输出路径或默认输出路径开始浏览
            starting_dir = self.output_path_edit.text() if os.path.exists(self.output_path_edit.text()) else self.default_output_path
            results_dir = QFileDialog.getExistingDirectory(
                self, "选择结果文件夹", starting_dir, QFileDialog.Option.ShowDirsOnly
            )
            
            if not results_dir:
                return
            
            # 规范化路径
            results_dir = os.path.normpath(results_dir)
            self.append_log(f"选择了结果文件夹: {results_dir}")
            
            # 检查是否存在analysis_summary.json文件
            summary_file_path = os.path.join(results_dir, "analysis_summary.json")
            if not os.path.exists(summary_file_path):
                error_msg = f"所选文件夹中未找到分析摘要文件:\n{summary_file_path}"
                self.append_log(error_msg)
                QMessageBox.warning(self, "文件不存在", error_msg)
                return
            
            try:
                # 先尝试读取摘要文件确保格式正确
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                if not summary_data.get("image_results"):
                    error_msg = "分析摘要文件中未找到有效的图像结果"
                    self.append_log(error_msg)
                    QMessageBox.warning(self, "无效数据", error_msg)
                    return
                
                # 清除当前状态
                self.append_log("清除当前图像查看器状态...")
                self.image_viewer.preview_list.clear()
                self.image_viewer.clear_results()
                
                # 加载结果
                self.append_log(f"开始加载结果...")
                success = self.load_results_from_summary(results_dir)
                
                if success:
                    self.append_log(f"成功加载结果: {results_dir}")
                    QMessageBox.information(self, "加载成功", 
                                          f"已加载分析结果: {os.path.basename(results_dir)}")
                    
                    # 确保切换到原始视图模式
                    self.image_viewer.view_original_button.setChecked(True)
                    self.image_viewer._set_view_mode(0)
                    
                    # 选择第一个预览项
                    if self.image_viewer.preview_list.count() > 0:
                        self.image_viewer.preview_list.setCurrentRow(0)
                else:
                    error_msg = f"加载结果失败，请检查日志了解详细信息。"
                    self.append_log(error_msg)
                    QMessageBox.warning(self, "加载失败", error_msg)
            
            except json.JSONDecodeError as e:
                error_msg = f"分析摘要文件格式错误: {str(e)}"
                self.append_log(error_msg)
                QMessageBox.critical(self, "文件错误", error_msg)
            except Exception as e:
                error_msg = f"加载结果时发生错误: {str(e)}"
                self.append_log(error_msg)
                logging.error(traceback.format_exc())
                QMessageBox.critical(self, "错误", error_msg)
        
        except Exception as e:
            error_msg = f"浏览结果时发生错误: {str(e)}"
            self.append_log(error_msg)
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)

# # --- 用于测试 --- #
# if __name__ == '__main__':
#     # 需要确保在项目根目录运行，或正确设置 PYTHONPATH
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec()) 