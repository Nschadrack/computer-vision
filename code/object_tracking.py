import os
import sys
import argparse
import cv2
from ultralytics import YOLO, solutions



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object tracking, cars.")
    parser.add_argument("-s", "--source", help="Source of the video. Path to the video(absolute path preferrable) and camera id")
    parser.add_argument("-lt", "--hlefttop", help="Distance from the middle height going up(negative value) left side")
    parser.add_argument("-lb", "--hleftbottom", help="Distance from the middle height going down(positive value) left side")
    parser.add_argument("-rt", "--hrighttop", help="Distance from the middle height going up(negative value) right side")
    parser.add_argument("-rb", "--hrightbottom", help="Distance from the middle height going down(positive value) right side")
    parser.add_argument("-l", "--lposition", help="Starting position from the left(positive value)")
    parser.add_argument("-r", "--rposition", help="Starting position from the right(negative value)")
    parser.add_argument("-d", "--destination", help="Destination file path if you want to save the output video with extension.mp4")

    args = parser.parse_args()

    source = args.source
    left_top = args.hlefttop
    left_bottom = args.hleftbottom
    right_top = args.hrighttop
    right_bottom = args.hrightbottom
    left_position = args.lposition
    right_position = args.rposition
    destination = args.destination

    if source is None or (source is not None and not os.path.isfile(source) and (source is not None and not source.isnumeric())):
        print("\n\nYou should provide the video source\n\n")
        sys.exit(1)

    
    if destination is not None and not os.path.isdir(os.path.dirname(destination)) or (destination is not None and not destination.endswith(".mp4")):
        print("\n\nYou should provide a valid output video file\n\n")
        sys.exit(1)
    
    if left_top is None:
        left_top = -40
    else:
        left_top = int(left_top)
    
    if left_bottom is None:
        left_bottom = 40
    else:
        left_bottom = int(left_bottom)
    
    if right_top is None:
        right_top = -40
    else:
        right_top = int(right_top)
    
    if right_bottom is None:
        right_bottom = 40
    else:
        right_bottom = int(right_bottom)
    
    if left_position is None:
        left_position = 340
    else:
        left_position = int(left_position)
    
    if right_position is None:
        right_position = 90
    else:
        right_position = int(right_position)

    if source:
        # Load the YOLO model
        model = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        # Get video properties
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        if destination is not None:
            video_writer = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        else:
            video_writer = None

        # Define region points for counting (optional, based on your application)
        region_points = [(left_position, (h//2) + left_top), (w + right_position, (h//2) + right_top), (w + right_position, (h//2) + right_bottom), (left_position, (h//2) + left_bottom)]

        # Initialize Object Counter (optional)
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=region_points,
            names=model.names,
            draw_tracks=True,
            line_thickness=2
        )
        # Class names to track (e.g., cars, buses, trucks)
        target_classes = [2, 3, 5, 6, 7]  # Modify as needed based on your specific classes


        while cv2.waitKey(1) != 27:
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False, classes=target_classes)
            im0 = counter.start_counting(im0, tracks)

            if video_writer is not None:
                video_writer.write(im0)

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        try:
            os.system("cls")
        except:
            os.system("clear")

        print(f"\n\nTotal cars: {len(counter.count_ids)}\n\n")
    else:
        print("\n\nYou must at least provide the video source")
