from config import Config
from tool import SegTool


config = Config()
source_path = r".\Videos\ScenicDrive.mp4"
tool = SegTool(source_path, config)
tool.start_window()
