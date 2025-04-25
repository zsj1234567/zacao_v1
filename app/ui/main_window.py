import sys
import os
import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QProgressBar, QTextEdit, QComboBox,
    QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QSizePolicy
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

        # 设置暗色主题 (可选)
        self.setup_dark_theme()

        # 初始化日志系统
        setup_logging(self.log_signal)

        # --- 主布局 ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_h_layout = QHBoxLayout(self.main_widget)

        # --- 左侧: 配置面板 ---
        self.config_widget = QWidget()
        self.config_layout = QVBoxLayout(self.config_widget)
        self.config_layout.setSpacing(15)
        self.config_layout.setContentsMargins(0, 0, 0, 0) # 移除边距，由父布局控制
        self.main_h_layout.addWidget(self.config_widget, 1) # 配置面板占1份

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
        self.browse_input_button = QPushButton("浏览...")
        self.browse_input_button.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_input_button)
        io_layout.addLayout(input_layout)

        # 输出路径
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出文件夹:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择保存结果的文件夹")
        self.browse_output_button = QPushButton("浏览...")
        self.browse_output_button.clicked.connect(self.browse_output)
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
        calibration_layout.addWidget(self.calibration_label)
        calibration_layout.addWidget(self.calibration_path_edit)
        calibration_layout.addWidget(self.browse_calibration_button)
        io_layout.addLayout(calibration_layout)

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
        left_params_form_layout.addRow(self.do_calibration_checkbox)

        self.save_debug_checkbox = QCheckBox("保存调试图像")
        self.save_debug_checkbox.setChecked(True)
        left_params_form_layout.addRow(self.save_debug_checkbox)

        self.calculate_density_checkbox = QCheckBox("计算覆盖密度")
        self.calculate_density_checkbox.setChecked(True)
        left_params_form_layout.addRow(self.calculate_density_checkbox)

        plot_layout_label = QLabel("绘图布局 (行, 列):")
        self.plot_layout_rows_spin = QSpinBox()
        self.plot_layout_rows_spin.setRange(1, 10)
        self.plot_layout_rows_spin.setValue(2)
        self.plot_layout_cols_spin = QSpinBox()
        self.plot_layout_cols_spin.setRange(1, 10)
        self.plot_layout_cols_spin.setValue(2)
        plot_layout_hbox = QHBoxLayout()
        plot_layout_hbox.addWidget(self.plot_layout_rows_spin)
        plot_layout_hbox.addWidget(QLabel("x"))
        plot_layout_hbox.addWidget(self.plot_layout_cols_spin)
        left_params_form_layout.addRow(plot_layout_label, plot_layout_hbox)

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
        # 使用样式表调整 QGroupBox 标题栏复选框的大小
        # 注意：具体数值可能需要根据视觉效果微调
        self.right_params_group.setStyleSheet("""
            QGroupBox::indicator {
                width: 13px;
                height: 13px;
            }
        """)

        right_params_layout = QFormLayout()
        # 可以选择性地为 QFormLayout 设置内部边距，调整标签/控件与 GroupBox 边框的距离
        # right_params_layout.setContentsMargins(10, 10, 10, 10)
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
        self.browse_lidar_button = QPushButton("浏览...")
        self.browse_lidar_button.clicked.connect(self.browse_lidar_dir)
        self.lidar_dir_layout.addWidget(self.lidar_dir_edit)
        self.lidar_dir_layout.addWidget(self.browse_lidar_button)
        right_params_layout.addRow(self.lidar_dir_label, self.lidar_dir_layout)

        self.dbscan_eps_label = QLabel("DBSCAN ε (邻域半径):")
        self.dbscan_eps_spinbox = QDoubleSpinBox()
        self.dbscan_eps_spinbox.setRange(0.01, 10.0)
        self.dbscan_eps_spinbox.setSingleStep(0.01)
        self.dbscan_eps_spinbox.setValue(0.1)
        right_params_layout.addRow(self.dbscan_eps_label, self.dbscan_eps_spinbox)

        self.dbscan_min_samples_label = QLabel("DBSCAN MinPts (最小点数):")
        self.dbscan_min_samples_spinbox = QSpinBox()
        self.dbscan_min_samples_spinbox.setRange(1, 100)
        self.dbscan_min_samples_spinbox.setValue(5)
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
        self.main_h_layout.setStretchFactor(bottom_layout, 1) # 让日志区域占据更多垂直空间

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始分析")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button = QPushButton("停止分析")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        bottom_layout.addLayout(control_layout)

        # --- 右侧: 可视化窗口 ---
        self.image_viewer = ImageViewerWidget()
        self.main_h_layout.addWidget(self.image_viewer, 2) # 可视化窗口占2份

        # 连接日志信号
        self.log_signal.connect(self.append_log)
        # 连接校准保存信号
        self.image_viewer.calibration_save_requested.connect(self.on_calibration_saved)

    def setup_dark_theme(self):
        """应用一个简单的暗色主题"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        # 设置禁用的文本颜色
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))

        QApplication.instance().setPalette(palette)

        # 样式表调整 (可选)
        self.setStyleSheet("""
            QWidget {
                font-size: 10pt;
            }
            QPushButton {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #777;
            }
            QPushButton:disabled {
                background-color: #444;
                border-color: #444;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                background-color: #424242;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png); /* 需要一个白色下拉箭头图标 */
                width: 12px;
                height: 12px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px; /* 为标题留出空间 */
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                left: 10px; /* 调整标题位置 */
            }
            QGroupBox:checked {
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                 /* TODO: Add unchecked box image */
                 border: 1px solid #666;
                 background-color: #424242;
            }
            QCheckBox::indicator:checked {
                 /* TODO: Add checked box image */
                 background-color: #2a82da;
                 border: 1px solid #2a82da;
            }
             QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #424242;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50; /* 进度条颜色 */
                border-radius: 4px;
            }
        """)

    def browse_input(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles) # 初始允许选择多个文件
        # 提供选择文件夹或文件的选项
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "", options=options)
        if path:
            self.input_path_edit.setText(path)
            return

        # 如果没有选择文件夹，尝试选择文件
        files, _ = QFileDialog.getOpenFileNames(self, "选择一个或多个图片文件", "",
                                            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)")
        if files:
            self.input_path_edit.setText(";".join(files))

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "", QFileDialog.Option.ShowDirsOnly)
        if path:
            self.output_path_edit.setText(path)

    def browse_hsv_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 HSV 配置文件", "", "JSON 文件 (*.json)")
        if path:
            self.hsv_config_path_edit.setText(path)

    def browse_dl_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择深度学习模型文件", "",
                                            "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)")
        if path:
            self.dl_model_path_edit.setText(path)

    def browse_lidar_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择 Lidar 数据文件夹", "", QFileDialog.Option.ShowDirsOnly)
        if path:
            self.lidar_dir_edit.setText(path)

    def browse_calibration(self):
        """浏览校准文件或目录"""
        dialog = QFileDialog(self)
        # 允许选择文件或目录
        path = QFileDialog.getExistingDirectory(self, "选择校准目录", "", QFileDialog.Option.ShowDirsOnly)
        if path:
            self.calibration_path_edit.setText(path)
            return
        # 如果没选目录，尝试选文件
        file, _ = QFileDialog.getOpenFileName(self, "选择校准文件", "", "JSON 文件 (*.json)")
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
        """收集 UI 配置并进行基本验证"""
        config = {}
        input_path = self.input_path_edit.text().strip()
        output_dir = self.output_path_edit.text().strip()
        calibration_path = self.calibration_path_edit.text().strip() # 获取校准路径

        if not input_path:
            QMessageBox.warning(self, "输入错误", "请选择输入图片或文件夹。")
            return None
        if not output_dir:
            QMessageBox.warning(self, "输入错误", "请选择输出文件夹。")
            return None

        # 处理输入路径 (单个文件、多个文件用分号分隔、文件夹)
        input_paths = []
        if os.path.isdir(input_path):
            supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            try:
                for fname in os.listdir(input_path):
                    if fname.lower().endswith(supported_exts):
                        input_paths.append(os.path.join(input_path, fname))
                if not input_paths:
                    QMessageBox.warning(self, "输入错误", f"文件夹 '{input_path}' 中未找到支持的图片文件。")
                    return None
            except OSError as e:
                QMessageBox.critical(self, "文件错误", f"无法读取文件夹 '{input_path}': {e}")
                return None
        elif os.path.isfile(input_path):
             input_paths = [input_path]
        elif ';' in input_path: # 多个文件
             input_paths = [p.strip() for p in input_path.split(';') if p.strip() and os.path.isfile(p.strip())]
             if not input_paths:
                 QMessageBox.warning(self, "输入错误", "提供的多个文件路径无效或文件不存在。")
                 return None
        else:
             QMessageBox.warning(self, "输入错误", f"输入路径 '{input_path}' 不是有效的文件或文件夹。")
             return None

        # Normalize all collected paths
        try:
            normalized_input_paths = [os.path.normpath(p) for p in input_paths]
        except Exception as e:
            QMessageBox.critical(self, "路径错误", f"规范化输入路径时出错: {e}")
            return None

        config['input_paths'] = normalized_input_paths
        config['output_dir'] = output_dir
        config['calibration_path'] = calibration_path if calibration_path else None # 添加到配置

        # 分析模型和方法
        is_dl_model = self.model_type_combo.currentText() == "深度学习"
        config['model_type'] = 'dl' if is_dl_model else 'traditional'

        if is_dl_model:
            config['dl_model_path'] = self.dl_model_path_edit.text().strip()
            config['dl_device'] = self.dl_device_combo.currentText().split(" ")[0] # 取 'cpu' 或 'cuda'
            if not config['dl_model_path'] or not os.path.isfile(config['dl_model_path']):
                QMessageBox.warning(self, "配置错误", "请选择有效的深度学习模型文件。")
                return None
        else:
            # 获取选中的分割方法枚举成员
            config['segment_method'] = self.segment_method_combo.currentText().lower()

            hsv_config = self.hsv_config_path_edit.text().strip()
            if hsv_config and not os.path.isfile(hsv_config):
                QMessageBox.warning(self, "配置错误", f"HSV 配置文件 '{hsv_config}' 不存在。")
                return None
            config['hsv_config_path'] = hsv_config if hsv_config else None

        # 通用参数
        config['do_calibration'] = self.do_calibration_checkbox.isChecked()
        config['save_debug_images'] = self.save_debug_checkbox.isChecked()
        config['calculate_density'] = self.calculate_density_checkbox.isChecked()
        config['plot_layout'] = (
            self.plot_layout_rows_spin.value(),
            self.plot_layout_cols_spin.value()
        )

        # Lidar 参数
        config['perform_lidar_analysis'] = self.right_params_group.isChecked()
        if config['perform_lidar_analysis']:
            lidar_dir = self.lidar_dir_edit.text().strip()
            if not lidar_dir or not os.path.isdir(lidar_dir):
                QMessageBox.warning(self, "配置错误", "请选择有效的 Lidar 数据文件夹。")
                return None
            config['lidar_dir'] = lidar_dir
            config['dbscan_eps'] = self.dbscan_eps_spinbox.value()
            config['dbscan_min_samples'] = self.dbscan_min_samples_spinbox.value()
        else:
             config['lidar_dir'] = None
             config['dbscan_eps'] = 0.1 # Default even if disabled
             config['dbscan_min_samples'] = 5 # Default even if disabled

        return config

    def start_analysis(self):
        config = self.get_config()
        if config is None:
            return # 配置无效

        # -- Clear previous state --
        self.pending_calibrations = []
        self.loaded_calibration_points = {}
        self.log_edit.clear() # Clear previous logs
        self.progress_bar.setValue(0)

        # -- Populate Image Viewer --
        # Do this early so user sees images even if calibration is needed
        self.image_viewer.load_images(config['input_paths'])

        if self.analysis_thread and self.analysis_thread.isRunning():
            QMessageBox.warning(self, "正在运行", "分析任务已在运行中。")
            return

        logging.info("获取配置。")

        # --- Calibration Handling --- 
        do_calib = config.get('do_calibration', False)
        calib_path_input = config.get('calibration_path')
        input_image_paths = config['input_paths']

        if do_calib:
            logging.info("检查校准文件...")
            # Determine the directory to search/save calibrations
            calib_dir_to_use = None
            if calib_path_input:
                if os.path.isdir(calib_path_input):
                    calib_dir_to_use = calib_path_input
                elif os.path.isfile(calib_path_input):
                    # If a file is given, use its directory
                    calib_dir_to_use = os.path.dirname(calib_path_input)
                else:
                     logging.warning(f"提供的校准路径无效: {calib_path_input}")

            if not calib_dir_to_use:
                 calib_dir_to_use = os.path.join(os.path.dirname(config['output_dir']), 'calibrations') # Default near output
                 logging.info(f"未提供有效校准目录，将使用默认位置: {calib_dir_to_use}")
                 os.makedirs(calib_dir_to_use, exist_ok=True)

            self.effective_calibration_dir = calib_dir_to_use # Store for saving later
            self.image_viewer.set_calibration_save_dir(self.effective_calibration_dir)

            for img_path in input_image_paths:
                 img_basename = os.path.splitext(os.path.basename(img_path))[0]
                 expected_json_path = os.path.join(self.effective_calibration_dir, f"{img_basename}.json")

                 if os.path.exists(expected_json_path):
                     try:
                         with open(expected_json_path, 'r') as f:
                             points = json.load(f)
                             if isinstance(points, list) and len(points) == 4:
                                 self.loaded_calibration_points[img_path] = points
                                 logging.info(f"为 {os.path.basename(img_path)} 加载校准点: {points}")
                             else:
                                 logging.warning(f"校准文件 {expected_json_path} 格式无效，需要手动校准。")
                                 self.pending_calibrations.append(img_path)
                     except Exception as e:
                         logging.error(f"加载校准文件 {expected_json_path} 时出错: {e}，需要手动校准。")
                         self.pending_calibrations.append(img_path)
                 else:
                      logging.info(f"未找到校准文件 {expected_json_path}，需要手动校准。")
                      self.pending_calibrations.append(img_path)

            # --- Trigger Manual Calibration if Needed ---
            if self.pending_calibrations:
                logging.info(f"需要手动校准 {len(self.pending_calibrations)} 张图像。")
                self.start_button.setEnabled(False) # Keep start disabled
                self.set_inputs_enabled(False) # Keep inputs disabled
                QMessageBox.information(self, "需要校准",
                                        f"请在右侧窗口为 {len(self.pending_calibrations)} 张图像选择1平方米区域的4个角点。\n" +
                                        '完成后点击"保存校准"按钮。')
                # Start calibration for the first image
                first_image_to_calibrate = self.pending_calibrations[0]
                self.image_viewer.set_calibration_mode(first_image_to_calibrate)
                return # Stop here, wait for on_calibration_saved signal
            else:
                 logging.info("所有图像均找到有效校准文件或不需要校准。")

        else: # Not doing calibration
             logging.info("跳过校准步骤。")

        # --- If we reach here, either calibration is done/skipped --- 
        self._proceed_with_analysis(config)

    def _proceed_with_analysis(self, config):
        """实际启动后台分析线程"""
        logging.info("准备开始后台分析...")
        logging.info(f"配置详情: {config}")
        logging.info(f"使用的校准点: {self.loaded_calibration_points}")

        # Add loaded calibration points to the config for the runner
        config['calibration_data'] = self.loaded_calibration_points.copy()

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

    def on_analysis_complete(self, success, message):
        logging.info(f"分析完成: success={success}, message='{message}'")
        self.progress_bar.setValue(100 if success else self.progress_bar.value())
        self.append_log(f"\n--- {message} ---")

        # 恢复按钮状态和输入
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("停止分析")
        self.set_inputs_enabled(True)

        # 清理引用
        self.analysis_thread = None
        self.analysis_runner = None

        # Re-enable UI for next run (or maybe keep disabled until explicitly cleared?)
        # For now, re-enable
        self.set_inputs_enabled(True)
        self.image_viewer.setEnabled(True)

        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "错误", message)

    def set_inputs_enabled(self, enabled):
        """启用或禁用所有输入控件"""
        # 输入输出
        self.input_path_edit.setEnabled(enabled)
        self.browse_input_button.setEnabled(enabled)
        self.output_path_edit.setEnabled(enabled)
        self.browse_output_button.setEnabled(enabled)
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
        self.plot_layout_rows_spin.setEnabled(enabled)
        self.plot_layout_cols_spin.setEnabled(enabled)
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
        """当用户在 ImageViewer 中保存校准点时调用"""
        logging.info(f"收到来自查看器的校准点: {image_path} -> {points}")
        if image_path in self.pending_calibrations:
            self.loaded_calibration_points[image_path] = points
            self.pending_calibrations.remove(image_path)

            if not self.pending_calibrations:
                # All calibrations are done!
                logging.info("所有手动校准完成，准备开始分析...")
                QMessageBox.information(self, "校准完成", "所有必需的校准已完成，即将开始分析。")
                self.image_viewer.exit_calibration_mode() # Add this method to viewer
                # Retrieve the original config again to start analysis
                config = self.get_config()
                if config:
                    self._proceed_with_analysis(config)
                else:
                     QMessageBox.critical(self, "错误", "无法在校准后重新获取配置，分析取消。")
                     self.set_inputs_enabled(True) # Re-enable inputs
                     self.start_button.setEnabled(True)
            else:
                # Trigger calibration for the next image
                next_image = self.pending_calibrations[0]
                logging.info(f"下一个需要校准的图像: {os.path.basename(next_image)}")
                QMessageBox.information(self, "继续校准", f"请为下一个图像校准： {os.path.basename(next_image)}")
                self.image_viewer.set_calibration_mode(next_image)
        else:
             logging.warning(f"收到未知图像的校准保存请求: {image_path}")

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

# # --- 用于测试 --- #
# if __name__ == '__main__':
#     # 需要确保在项目根目录运行，或正确设置 PYTHONPATH
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec()) 