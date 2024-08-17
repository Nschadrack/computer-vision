import cv2


def read_video(video_source):
    """
    Function to open the video source
    params:
        video_source: the source of the video either from file or from camera (0 is default camera where the script is running)
    returns:
        cap: video capture object
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError("Cannot open video source")
    return cap


# Define IoU and box merging functions
def iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection coordinates
    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2, y2_2)
    
    # Calculate intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate areas of both boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou_value = inter_area / (box1_area + box2_area - inter_area)
    return iou_value


def merge_boxes(boxes, threshold=0.3):
    """Merge overlapping bounding boxes."""
    merged_boxes = []
    while boxes:
        # Take the first box and remove from list
        box = boxes.pop(0)
        x1, y1, x2, y2 = box[0]

        # Initialize with the first box
        merged_box = [x1, y1, x2, y2]
        confidence, pred_class_id = box[1], box[2]

        # Compare with all other boxes
        for other_box in boxes[:]:
            if iou(merged_box, other_box[0]) > threshold:
                # Merge boxes by extending the merged box
                x1 = min(merged_box[0], other_box[0][0])
                y1 = min(merged_box[1], other_box[0][1])
                x2 = max(merged_box[2], other_box[0][2])
                y2 = max(merged_box[3], other_box[0][3])
                merged_box = [x1, y1, x2, y2]
                confidence, pred_class_id
                boxes.remove(other_box)
        
        merged_boxes.append((merged_box, confidence, pred_class_id ))
    return merged_boxes