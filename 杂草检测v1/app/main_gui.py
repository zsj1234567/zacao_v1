import sys
import os
from PyQt6.QtWidgets import QApplication
import datetime

# 将项目根目录添加到 sys.path，以便导入 scripts 模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # 项目根目录路径处理
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ui.main_window import MainWindow   #主窗口模块导入
from utils.db_connection import get_db_connection

try:
    db_conn = get_db_connection()
    print("数据库连接成功")
except Exception as e:
    print("数据库连接失败:", e)
    db_conn = None

if __name__ == '__main__':
    # 日志目录和文件初始化
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime('runlog_%Y%m%d_%H%M%S.log')
    log_file_path = os.path.join(logs_dir, log_filename)

    app = QApplication(sys.argv)
    main_window = MainWindow(log_file_path=log_file_path)
    main_window.show()
    sys.exit(app.exec()) 