# YOLO 人脸检测模块 —— 负责从摄像头画面中找出人脸的位置

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class FaceDetector:
    def __init__(self,model_path: str='yolov8-face.pt'):
        """
        初始化YOLO模型
        """
        self.model = YOLO(model_path)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的所有人脸
        :param image: 摄像头读取的 BGR 格式图像 (numpy 数组)
        :return: 人脸框列表，每个框为 (x1, y1, x2, y2) 左上角和右下角坐标
        知识点：【类型提示】、【List与Tuple】
        """
        results = self.model(image)
        faces = []

        for result in results:
            for box in result.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                faces.append((x1, y1, x2, y2))

        return faces
