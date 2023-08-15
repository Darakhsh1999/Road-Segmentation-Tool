import cv2
from tool import SegTool
from config import Config


config = Config()
source_path = r".\Videos\ScenicDrive.mp4"
SegTool(source_path, config)