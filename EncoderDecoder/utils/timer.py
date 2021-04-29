import datetime
import time


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.reset()
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()

    def elapsed(self):
        if self.start_time and self.stop_time:
            return self.stop_time - self.start_time
        else:
            return -1 # return invalid time

    @staticmethod
    def timeString():
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return st

    @staticmethod
    def timeFilenameString():
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d[%H_%M_%S]')
        return st
