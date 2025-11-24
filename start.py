from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("API key not found!")

rf = Roboflow(api_key=api_key)
project = rf.workspace("classroom-lzi7f").project("emotion-57ymy")
version = project.version(2)
dataset = version.download("yolov12")
                