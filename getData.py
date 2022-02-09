import time
import timeit
from tqdm import tqdm


class FrequencyLimitation:  # Tushare 调取数据频率上限：1min 500次 每次5000条
    def __init__(self, time_interval=0.12):
        self.dTime = 0.0
        self.timeStamp = 0.0
        self.timeInterval = time_interval  # 单位 ： 1s
        pass

    def __call__(self):  # 在请求前或后使用
        # self.dTime = timeit.default_timer() - self.timeStamp
        self.timeStamp += self.timeInterval
        # print(self.timeStamp, self.dTime)
        if timeit.default_timer() < self.timeStamp:
            time.sleep(self.timeStamp - timeit.default_timer())
        self.timeStamp = timeit.default_timer()
        pass


print("getData.py has been imported")

if __name__ == "__main__":

    wait = FrequencyLimitation()
    for i in tqdm(range(1000000000)):
        wait()
        # print("TimeStamp      : ", wait.dTime)
        # print("Time  interval : ", wait.dTime)
        # print(i)
