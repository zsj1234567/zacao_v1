import sys
import os
from PyQt6.QtWidgets import QApplication

# 将项目根目录添加到 sys.path，以便导入 scripts 模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ui.main_window import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec()) 