from pydantic import BaseModel
from typing import List, Tuple, Dict


class VideoOutput:
    def __init__(self, video_path: str):
        self.video_path = video_path


class Сlass1:
    def __init__(self, class1_arg):
        self.class1_arg = class1_arg
