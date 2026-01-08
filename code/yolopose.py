from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------- 1. 加载模型并推理（延续你的原有代码）----------------------
model = YOLO("yolo11x-pose.pt")  # 加载官方姿态模型
img_path = "/data/ljy23/data/output_frames/frame_0005.jpg"
# 推理（添加save=False避免自动保存默认结果，我们自定义绘图）
results = model(img_path, save=False)

# ---------------------- 2. 定义关键配置（骨架连接、颜色等）----------------------
# （1）YOLO11-Pose 17个关节点的顺序（对应keypoints的索引）
keypoint_labels = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左手腕",
    "右手腕", "左髋", "右髋", "左膝", "右膝",
    "左脚踝", "右脚踝"
]

# （2）骨架连接关系（每个元素是两个关节点的索引，对应上述顺序）
# 确保连接逻辑合理：躯干→四肢→头部
skeleton_connections = [
    (5, 6),  # 左肩 ↔ 右肩
    (5, 11), (6, 12),  # 左肩 ↔ 左髋，右肩 ↔ 右髋
    (11, 12),  # 左髋 ↔ 右髋
    (5, 7), (7, 9),  # 左肩 ↔ 左肘 ↔ 左手腕
    (6, 8), (8, 10),  # 右肩 ↔ 右肘 ↔ 右手腕
    (11, 13), (13, 15),  # 左髋 ↔ 左膝 ↔ 左脚踝
    (12, 14), (14, 16),  # 右髋 ↔ 右膝 ↔ 右脚踝
    (0, 1), (0, 2),  # 鼻子 ↔ 左眼、鼻子 ↔ 右眼
    (1, 3), (2, 4)   # 左眼 ↔ 左耳、右眼 ↔ 右耳
]

# （3）绘图颜色配置（关节点、骨架线段）
kp_color = (0, 255, 0)  # 关节点：绿色
skeleton_color = (255, 0, 0)  # 骨架线段：蓝色
kp_radius = 5  # 关节点半径
line_thickness = 2  # 骨架线段粗细

# ---------------------- 3. 加载原图并进行骨架绘制 ----------------------
# 读取原图（OpenCV格式）
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法读取图片：{img_path}")

# 遍历推理结果（支持单图多人体）
for result in results:
    # 提取关节点数据：(人体数量, 17个关节点, 3) → (x, y, visibility)
    # visibility：置信度（0-1，越接近1表示关节点越可靠）
    keypoints = result.keypoints.data.cpu().numpy()  # 转换为numpy数组（脱离GPU/张量）
    
    # 遍历每个人体的关节点
    for person_kpts in keypoints:
        # ---------------------- 绘制骨架线段（先画线段，再画关节点，避免点被线段遮挡）----------------------
        for (idx1, idx2) in skeleton_connections:
            # 获取两个关节点的坐标和置信度
            x1, y1, vis1 = person_kpts[idx1]
            x2, y2, vis2 = person_kpts[idx2]
            
            # 过滤低置信度关节点（仅绘制置信度>0.5的连接，避免无效绘图）
            if vis1 > 0.5 and vis2 > 0.5:
                # 转换为整数坐标（OpenCV绘图要求整数）
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                # 绘制线段
                cv2.line(img, pt1, pt2, skeleton_color, line_thickness)
        
        # ---------------------- 绘制关节点（圆形）----------------------
        for (x, y, vis) in person_kpts:
            if vis > 0.5:  # 过滤低置信度关节点
                pt = (int(round(x)), int(round(y)))
                cv2.circle(img, pt, kp_radius, kp_color, -1)  # -1表示填充圆形

# ---------------------- 4. 保存或显示结果 ----------------------
# （1）保存绘制后的图片（服务器无图形界面优先选择，推荐）
output_img_path = ".frame_0005_with_skeleton.jpg"
cv2.imwrite(output_img_path, img)
print(f"骨架绘制完成，结果保存至：{output_img_path}")

# （2）显示图片（仅适用于有图形界面的环境，服务器端注释此行）
# cv2.imshow("YOLO11-Pose Skeleton Result", img)
# cv2.waitKey(0)  # 按下任意键关闭窗口
# cv2.destroyAllWindows()