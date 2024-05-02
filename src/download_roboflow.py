from roboflow import Roboflow
rf = Roboflow(api_key="cHcjYZ7fHmj1slaM8q6M")
project = rf.workspace("training-s7fhi").project("peru-license-plate")
version = project.version(3)
dataset = version.download("yolov8")

dataset.save("datasets")
