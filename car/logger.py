import csv
import time

from car.config import LOG_FILE


class StateLogger:
    def __init__(self, path=LOG_FILE):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "state", "command", "lane_center", "steering"])

    def log(self, state, command, lane_center, steering):
        self._writer.writerow([time.time(), state, command, lane_center, steering])
        self._file.flush()

    def close(self):
        self._file.close()
