import sys


class Saver(object):
    def __init__(self, filename='output.log', stream=sys.stdout):
        # 初始化
        self.terminal = stream
        self.filename = filename

    def write(self, message):
        # 重写write方法，将输出同时写入终端和日志文件
        self.terminal.write(message)  # 将输出写入终端
        # 将输出写入日志文件,print时自动调用
        with open(self.filename, 'a') as log:
            log.write(message)

    def flush(self):
        # 重写flush方法，不做任何操作
        pass
# 使用示例
# sys.stdout = Logger(a.log, sys.stdout) # 重定向标准输出
# sys.stderr = Logger(a.log_file, sys.stderr) # 重定向标准错误输出（如果需要）
