import sys
from ultralytics import YOLO
from tracking_car import process_video


if __name__ == "__main__":
    args = sys.argv

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

    class_id = [2, 5, 6, 7]  # 2 -> car, 5 -> bus, 6-> train, 7 -> truck
    model = YOLO("yolov8n.pt", task="track")


    if len(args) == 3: # provided tracking line coord and tracking region
        tracking_line_coord = tuple(int(i)for i in args[1].strip("(").strip(")").split(","))
        tracking_region = int(args[2])
    if len(args) == 2:
        tracking_line_coord = tuple(int(i) for i in args[1].strip("(").strip(")").split(","))
        tracking_region = -20
    else:
        tracking_line_coord = None
        tracking_region = -20

    # print(args)
    # print(tracking_line_coord)
    process_video(video_source=path, 
                  model=model, class_names=class_names, 
                  class_id=2, tracking_line_coord=tracking_line_coord, 
                  tracking_region=tracking_region)