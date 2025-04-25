import logging
from PyQt6.QtCore import QObject, pyqtSignal

class QtLogHandler(logging.Handler):
    """将日志记录发送到 PyQt 信号的处理器"""
    def __init__(self, log_signal: pyqtSignal):
        super().__init__()
        self.log_signal = log_signal
        # 设置一个更详细的格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.setFormatter(formatter)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_signal.emit(msg + "\n") # PyQt 文本区域需要显式换行符
        except Exception:
            self.handleError(record)

def setup_logging(log_signal: pyqtSignal, level=logging.INFO):
    """配置 Python 的 logging 系统以使用 QtLogHandler"""
    # 获取根记录器
    logger = logging.getLogger()
    logger.setLevel(level)

    # 移除可能存在的旧处理器，避免重复输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建并添加 Qt 处理器
    qt_handler = QtLogHandler(log_signal)
    logger.addHandler(qt_handler)

    # (可选) 如果希望控制台也输出日志，可以添加一个 StreamHandler
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(stream_handler)

    # 覆盖 print 函数 (可选，但可以捕获脚本中直接的 print 输出)
    # 注意：这会影响全局的 print，仅在需要时使用
    # import builtins
    # original_print = builtins.print
    # def gui_print(*args, **kwargs):
    #     message = " ".join(map(str, args))
    #     log_signal.emit(f"[PRINT] {message}\n")
    #     # original_print(*args, **kwargs) # 如果还想在控制台打印
    # builtins.print = gui_print

    # 测试日志
    logging.info("日志系统已初始化，将输出到 GUI。")

# --- 如果需要恢复 print ---
# def restore_print():
#     import builtins
#     # Assuming original_print was stored somewhere accessible
#     # builtins.print = original_print
#     pass 