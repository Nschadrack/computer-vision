from ultralytics import YOLO
from tracking_car import process_video


if __name__ == "__main__":
    class_names_path = "D:\\PERSONAL\\AI\\objectTracking\\yolo_files\\classes.txt"
    path = "D:\\PERSONAL\\AI\\objectTracking\\videos\\los_angeles.mp4"
    # path = "D:\\PERSONAL\\AI\\objectTracking\\videos\\test_video_22_cars.mp4"
    class_names = []
    with open(class_names_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_names.append(line)
    
    # for index, value in enumerate(class_names):
    #     print(f"{index} -> {value}")

    class_id = [2, 5, 7]  # 2 -> car, 5 -> bus, 7 -> truck
    model = YOLO("yolov8n.pt", task="track")

    process_video(video_source=path, model=model, class_names=class_names, class_id=2)