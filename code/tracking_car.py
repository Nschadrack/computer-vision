import cv2
from helpers import read_video, merge_boxes
from deep_sort_realtime.deepsort_tracker import DeepSort

# logging.getLogger().setLevel(logging.WARNING)

def track_car(model, frame, tracker, class_id):
    """
    Function to perform object tracking using YOLO and Deep SORT
    params:
        model: the YOLO model for tracking
        frame: the current frame of the video
        tracker: the Deep SORT tracker instance
        class_id: the target class id(integer)
    returns:
        results: the results of the tracking
        tracks: the tracked objects with IDs
    """
    if not isinstance(class_id, list):
        class_id = [class_id]

    pred_class_ids = []
    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            pred_class_id = int(box.cls[0])
            if pred_class_id in class_id: # Filter class based on class id
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # converting coordinates to integer
                width = x2 - x1
                height = y2 - y1
                confidence = float(box.conf[0]) # Convert confidence to float

                if confidence >= 0.65:
                    detections.append(([x1, y1, width, height], confidence, pred_class_id))
                    pred_class_ids.append(pred_class_id)
    
    # Merge overlapping boxes
    detections = merge_boxes(detections)
    # Update the tracker with new detections
    tracks = tracker.update_tracks(raw_detections=detections, frame=frame)

    return results, tracks



def process_video(video_source, model, class_names, class_id, output_path=None):
    """
    Function for processing the video
    params:
        video_source: he source  of the video either from file or from camera(real time) -> 0 is default camera where the script is running
        output_path: the path where the output video will be saved in casse you need to save it
    """
    if not isinstance(class_id, list):
        class_id = [class_id]

    cap = read_video(video_source=video_source) 
    window = "Video"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Initialize the Deep SORT tracker
    tracker =  DeepSort(max_age=20, n_init=3, max_iou_distance=0.5, max_cosine_distance=0.5)

    # Initialize the set for keeping unique car IDs
    unique_car_ids = set()

    while cv2.waitKey(1) != 27:
        has_frame, frame = cap.read()

        if not has_frame:
            print("Reading video completed!")
            break

        # perform object tracking
        results, tracks = track_car(model, frame, tracker, class_id)
        frame_height, frame_width = frame.shape[:2]
        coord = (0, (frame_height//2) - 40, frame_width, frame_height//2)
        cv2.line(frame, (coord[0], coord[1]-20), (coord[2], coord[3]-20), (0, 0, 255), thickness=3)

        # # Filter results to focus on required class name
        # if len(results) > 0:
        #     annotated_frame = results[0].plot()  # Using .plot() for visualization
        # else:
        #     annotated_frame = frame  # No detections, show the original frame
        annotated_frame = frame.copy()


        # Draw bounding boxes around detected cars
        for track in tracks:
            bbox = track.to_tlbr() # Get bounding box in (x1, y1, x2, y2) format
            track_id = track.track_id

            x1, y1, x2, y2 = map(int, bbox)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if x1 < cx < x2 and  y1 - 5 < cy < y2 + 5: # count car if it passes through line region
                # Add track ID to unique_car_ids
                unique_car_ids.add(track_id)  # add track id to the set

            cv2.putText(annotated_frame, f"Car counts: {len(unique_car_ids)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.putText(annotated_frame, f"Tracking ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        cv2.imshow(window, annotated_frame)
        print(f"Car IDs = {unique_car_ids}\n\nTotal cars: {len(unique_car_ids)}\n")
    cap.release()
    cv2.destroyWindow(window)

