import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import numpy as np


class Logger(object):
    def __init__(self, stdout, filename="log.txt"):
        self.terminal = stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()  # 缓冲区的内容及时更新到log文件中

    def flush(self):
        pass

    def close(self):
        self.log.close()
