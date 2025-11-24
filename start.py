from roboflow import Roboflow

rf = Roboflow(api_key="#")   # Необходимо изменать на ваш roboflow Private API Key с сайта roboflow.com
project = rf.workspace("classroom-lzi7f").project("emotion-57ymy")
version = project.version(2)
dataset = version.download("yolov12")
                
