# db_logger.py

import sqlite3
import numpy as np
import threading
from datetime import datetime


class TrainingLogger:
    def __init__(self, db_path='negotiation_training.db'):
        """
        初始化数据库日志记录器。
        每个线程将使用自己的数据库连接。
        :param db_path: SQLite数据库文件的路径。
        """
        self.db_path = db_path
        self._local = threading.local()  # 使用线程局部存储来管理连接
        self.init_db()

    def get_conn(self):
        """为当前线程获取或创建一个新的数据库连接。"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn

    def init_db(self):
        """
        初始化数据库并创建 'training_log' 表（如果不存在）。
        """
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            # 创建表的SQL语句
            # BLOB 类型用于存储二进制数据，如序列化后的Numpy数组
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                state BLOB NOT NULL,
                discrete_action INTEGER NOT NULL,
                action_parameters BLOB NOT NULL,
                reward REAL NOT NULL,
                next_state BLOB NOT NULL,
                next_discrete_action INTEGER NOT NULL,
                next_action_parameters BLOB,
                done INTEGER NOT NULL
            )
            ''')
            conn.commit()
            print(f"数据库 '{self.db_path}' 初始化成功。")
        except sqlite3.Error as e:
            print(f"数据库初始化失败: {e}")

    def log_step(self, state, action, reward, next_state, next_action, done):
        """
        记录一步的训练数据到数据库中。

        :param state: 当前状态 (numpy.ndarray)
        :param action: 执行的动作 (元组: int, numpy.ndarray)
        :param reward: 奖励 (float)
        :param next_state: 下一个状态 (numpy.ndarray)
        :param next_action: 下一个动作 (元组: int, numpy.ndarray)
        :param done: 是否结束 (bool)
        """
        try:
            conn = self.get_conn()
            cursor = conn.cursor()

            discrete_act_idx, all_actual_params_vector = action
            next_discrete_act_idx, next_all_actual_params_vector = next_action

            # 将Numpy数组转换为二进制格式(BLOB)
            # 使用 numpy.tobytes() 进行序列化
            state_blob = state.tobytes()
            params_blob = all_actual_params_vector.tobytes()
            next_state_blob = next_state.tobytes()
            next_params_blob = next_all_actual_params_vector.tobytes()

            # 将布尔值转换为整数 (0 or 1)
            done_int = 1 if done else 0

            # 获取当前时间戳
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # 插入数据的SQL语句
            cursor.execute('''
            INSERT INTO training_log (
                timestamp, state, discrete_action, action_parameters, reward, 
                next_state, next_discrete_action, next_action_parameters, done
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp_str,
                sqlite3.Binary(state_blob),
                discrete_act_idx,
                sqlite3.Binary(params_blob),
                reward,
                sqlite3.Binary(next_state_blob),
                next_discrete_act_idx,
                sqlite3.Binary(next_params_blob),
                done_int
            ))

            conn.commit()
        except sqlite3.Error as e:
            print(f"写入数据库失败: {e}")
            # 在多线程环境中，最好不要在这里关闭连接
            # conn.rollback()

    def close(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn

    @staticmethod
    def read_log(db_path='negotiation_training.db', start_time=None, end_time=None):
        """
        从数据库读取指定时间范围内的所有记录（不包括timestamp和id字段）

        :param db_path: 数据库文件路径
        :param start_time: 开始时间字符串 (格式: "%Y-%m-%d %H:%M:%S.%f")
        :param end_time: 结束时间字符串 (格式: "%Y-%m-%d %H:%M:%S.%f")
        :return: 包含所有记录的列表，每条记录为字典
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 构建查询条件 - 不选择id字段
        query = "SELECT state, discrete_action, action_parameters, reward, next_state, next_discrete_action, next_action_parameters, done FROM training_log"
        params = ()

        if start_time and end_time:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params = (start_time, end_time)
        elif start_time:
            query += " WHERE timestamp >= ?"
            params = (start_time,)
        elif end_time:
            query += " WHERE timestamp <= ?"
            params = (end_time,)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # 列名列表（不包括timestamp和id）
        columns = [
            'state', 'discrete_action', 'action_parameters',
            'reward', 'next_state', 'next_discrete_action',
            'next_action_parameters', 'done'
        ]

        results = []
        for row in rows:
            record = dict(zip(columns, row))

            # 反序列化二进制字段
            binary_fields = ['state', 'action_parameters', 'next_state', 'next_action_parameters']
            for field in binary_fields:
                if record[field]:
                    # 注意：实际使用时需要根据原始数据类型调整dtype
                    record[field] = np.frombuffer(record[field], dtype=np.float32)

            # 转换done字段为布尔值
            record['done'] = bool(record['done'])

            results.append(record)

        return results

if __name__ == '__main__':
    results=TrainingLogger.read_log('../output/training_data.db')