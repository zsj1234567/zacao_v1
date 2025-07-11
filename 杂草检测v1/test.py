import os
import ctypes

# 修改为你的PostgreSQL bin目录路径
postgres_bin = r"C:\Program Files\PostgreSQL\15\bin"

# 添加路径到环境变量
os.environ["PATH"] = postgres_bin + ";" + os.environ["PATH"]

try:
    # 尝试加载libpq.dll
    libpq = ctypes.CDLL("libpq.dll")
    print("libpq.dll加载成功!")
except Exception as e:
    print(f"加载失败: {e}")

    # 显示详细的搜索路径
    print("\nDLL搜索路径:")
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(path) and "postgres" in path.lower():
            print(f"- {path}")