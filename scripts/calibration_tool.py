import cv2
import numpy as np
import os
import json

# 全局变量
points = []
image = None
original_image = None
window_name = "Calibration Tool - Select 4 corners of 1 square meter area"
window_closed = False
display_scale = 1.0  # 显示比例


def on_window_close(event, x, y, flags, param):
    global window_closed
    if event == cv2.EVENT_LBUTTONDOWN:
        window_closed = True


def resize_image_keep_aspect_ratio(img, max_width=1280, max_height=720):
    """调整图像大小，保持纵横比，不产生形变"""
    height, width = img.shape[:2]

    # 计算缩放比例，确保宽度和高度都不超过最大值
    scale_width = min(max_width / width, 1.0)
    scale_height = min(max_height / height, 1.0)
    scale = min(scale_width, scale_height)

    # 如果图像已经足够小，不需要缩放
    if scale >= 1.0:
        return img.copy(), 1.0

    # 缩放图像
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img, scale


def create_info_panel(width, text):
    """创建信息面板"""
    # 创建固定高度的信息栏，使用半透明效果
    info_height = 30  # 减小高度
    info_img = np.ones((info_height, width, 3), dtype=np.uint8) * 245  # 更浅的背景色

    # 添加底部边框
    cv2.line(info_img, (0, info_height-1),
             (width, info_height-1), (200, 200, 200), 1)

    # 使用更小更美观的字体
    font_size = 0.5  # 减小字号
    font_thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX  # 使用更美观的字体

    # 计算文本大小以居中显示
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_size, font_thickness)
    text_x = (width - text_width) // 2
    text_y = (info_height + text_height) // 2 - 2  # 微调垂直位置

    # 绘制文本
    cv2.putText(info_img, text, (text_x, text_y),
                font, font_size, (50, 50, 50), font_thickness)  # 深灰色文字

    return info_img


def display_image():
    """显示图像和信息"""
    global image, display_scale
    if image is None:
        return

    h, w = image.shape[:2]

    # 创建信息文本
    info_text = f"Selected points: {len(points)}/4  |  Press 'r' to reset, 's' to save, 'q' to quit"

    # 创建顶部信息面板
    top_info = create_info_panel(w, info_text)

    # 将图像和信息面板垂直堆叠
    display_img = np.vstack([top_info, image])

    # 显示图像
    cv2.imshow(window_name, display_img)


def order_points_clockwise(pts):
    """
    将四个点按顺时针排序，从左上角开始
    这样无论用户点击的顺序如何，都能得到一致的透视变换结果
    """
    if len(pts) != 4:
        return pts

    # 转换为numpy数组
    pts = np.array(pts)

    # 计算质心
    center = np.mean(pts, axis=0)

    # 计算每个点相对于质心的角度
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    # 按角度排序
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    # 找到左上角点（距离原点最近的点）
    distances = np.sum((sorted_pts - np.array([0, 0]))**2, axis=1)
    start_idx = np.argmin(distances)

    # 重新排序，从左上角开始
    sorted_pts = np.roll(sorted_pts, -start_idx, axis=0)

    return sorted_pts.tolist()


def click_event(event, x, y, flags, param):
    """鼠标点击事件处理函数"""
    global points, image, original_image, display_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            # 获取信息面板高度
            info_height = 30

            # 确保点击在图像区域内
            if y <= info_height:
                return  # 点击在信息栏上，忽略

            # 计算点在原始图像中的位置
            original_x = int(x / display_scale)
            original_y = int((y - info_height) / display_scale)

            # 添加点
            points.append((original_x, original_y))

            # 让点和线的大小与显示比例无关，始终保持视觉一致
            fixed_point_radius = 5  # 你可以根据需要调整
            fixed_border_radius = 7
            fixed_line_width = 2
            point_radius = max(1, int(round(fixed_point_radius / display_scale)))
            border_radius = max(point_radius + 1, int(round(fixed_border_radius / display_scale)))
            line_width = max(1, int(round(fixed_line_width / display_scale)))

            point_pos = (int(x), int(y - info_height))
            # 绘制白色边框
            cv2.circle(image, point_pos, border_radius, (255, 255, 255), -1)
            # 绘制红色填充
            cv2.circle(image, point_pos, point_radius, (0, 0, 255), -1)

            # 显示点的序号，使用更小的字体和位置
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1
            text = str(len(points))
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = point_pos[0] - text_width // 2
            text_y = point_pos[1] - point_radius - 4
            # 绘制白色文本背景
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness + 1)
            # 绘制黑色文本
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            # 如果已经有多个点，绘制连线
            if len(points) > 1:
                for i in range(len(points)-1):
                    pt1 = (int(points[i][0] * display_scale),
                           int(points[i][1] * display_scale))
                    pt2 = (int(points[i+1][0] * display_scale),
                           int(points[i+1][1] * display_scale))
                    # 线宽也根据显示比例修正
                    cv2.line(image, pt1, pt2, (0, 255, 0), line_width, lineType=cv2.LINE_AA)

            # 如果选择了4个点，绘制闭合线
            if len(points) == 4:
                pt1 = (int(points[-1][0] * display_scale),
                       int(points[-1][1] * display_scale))
                pt2 = (int(points[0][0] * display_scale),
                       int(points[0][1] * display_scale))
                cv2.line(image, pt1, pt2, (0, 255, 0), line_width, lineType=cv2.LINE_AA)

            display_image()


def calibrate_image(image_path, output_dir='calibrations'):
    """
    交互式校准工具

    参数:
        image_path: 图像文件路径
        output_dir: 校准文件保存目录，默认为 'calibrations'

    返回:
        校准点坐标列表或 None（如果校准被取消）
    """
    global points, image, original_image, window_name, window_closed, display_scale

    # 重置全局变量
    points = []
    window_closed = False

    # 确保校准目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建校准文件路径，保持与图像相同的目录结构
    rel_path = os.path.dirname(image_path)
    if rel_path == '':  # 如果图像在当前目录
        rel_path = '.'

    # 在calibrations目录下创建对应的子目录
    calibration_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(calibration_subdir, exist_ok=True)

    # 构建校准文件路径，去除图片后缀
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    calibration_file = os.path.join(calibration_subdir, f"{image_name}.json")

    # 检查是否存在校准文件
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                points = json.load(f)
            return points
        except Exception as e:
            print(f"加载校准文件时出错: {e}")
            print("将重新进行校准...")

    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 获取屏幕分辨率
    screen_width = 1920  # 默认值，实际应根据系统获取
    screen_height = 1080
    try:
        # 尝试获取实际屏幕分辨率
        from screeninfo import get_monitors
        for m in get_monitors():
            if m.is_primary:
                screen_width = m.width
                screen_height = m.height
                break
    except:
        pass  # 如果无法获取屏幕信息，使用默认值

    # 计算可用显示区域（减去窗口标题栏和边框的估计高度）
    available_height = screen_height - 100
    available_width = screen_width - 100

    # 调整图像大小，保持纵横比
    image, display_scale = resize_image_keep_aspect_ratio(
        original_image, available_width, available_height)

    # 创建窗口并设置鼠标回调
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    # 设置窗口大小为图像大小加上信息栏高度
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0] + 30)

    # 显示初始图像
    display_image()

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 检查窗口是否被关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            points = None
            break

        if key == ord('r'):
            points = []
            # 重新加载图像，保持原始比例
            image, display_scale = resize_image_keep_aspect_ratio(
                original_image, available_width, available_height)
            display_image()

        elif key == ord('s'):
            if len(points) == 4:
                # 对点进行顺时针排序，确保透视变换结果与原图方向一致
                ordered_points = order_points_clockwise(points)

                # 保存校准点到JSON文件
                with open(calibration_file, 'w') as f:
                    json.dump(ordered_points, f)
                print(f"校准点已保存到 {calibration_file}")
                break
            else:
                print(f"请选择4个点 (当前已选择 {len(points)} 个点)")
                display_image()

        elif key == ord('q') or key == 27:  # q 或 ESC
            points = None
            break

    cv2.destroyAllWindows()
    return points


def main():
    # 获取数据集中的所有图像
    image_dir = 'datasets'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if not image_files:
        print("未找到图像文件")
        return

    # 处理第一张图像作为示例
    image_path = os.path.join(image_dir, image_files[0])
    print(f"校准图像: {image_path}")

    # 运行校准工具
    calibration_points = calibrate_image(image_path)

    if calibration_points:
        print("校准完成！")
        print(f"校准点: {calibration_points}")
        print("现在可以运行 grass_analysis.py 进行草地分析")


if __name__ == "__main__":
    main()
