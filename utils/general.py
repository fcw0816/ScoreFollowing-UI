import math

def make_divisible(x, divisor):
    """taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py"""
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor