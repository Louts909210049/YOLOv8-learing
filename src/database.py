# SQLite 数据库操作
import sqlite3
import numpy as np
import os
from typing import Tuple

class FaceDatabase:
    def __init__(self, db_path='face_database.db'):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                            CREATE TABLE IF NOT EXISTS faces (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT NOT NULL,
                                embedding BLOB NOT NULL
                            )
                        ''')
            conn.commit()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """将numpy数组序列化为bytes"""
        return embedding.tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """将bytes反序列化为numpy数组"""
        return np.frombuffer(blob, dtype=np.float32)

    def add_face(self, name: str, embedding: np.ndarray) -> bool:
        """
        添加人脸到数据库
        :param name: 人名
        :param embedding: 人脸嵌入向量
        :return: 是否添加成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO faces (name, embedding) VALUES (?, ?)',
                    (name, self._serialize_embedding(embedding))
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"添加人脸失败: {e}")
            return False

    def find_closest_face(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        """
        查找最匹配的人脸
        :param query_embedding: 查询的人脸嵌入向量
        :return: (name, distance) 最匹配的人名和距离
        """
        min_distance = float('inf')
        best_name = "Unknown"

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, embedding FROM faces')
                rows = cursor.fetchall()

                for name, blob in rows:
                    embedding = self._deserialize_embedding(blob)
                    distance = np.linalg.norm(query_embedding - embedding)

                    if distance < min_distance:
                        min_distance = distance
                        best_name = name

                if min_distance > 1.0:  # 设置一个阈值
                    best_name = "Unknown"

        except Exception as e:
            print(f"查找人脸失败: {e}")

        return best_name, min_distance

