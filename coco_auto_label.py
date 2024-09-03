import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
import json

yolo_model_path = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\yolov8x-pose.pt'
video_source = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\kesim.mp4"

OUTPUT_DIR = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\yolov8pose_output\coco_pose3'
LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

model = YOLO(yolo_model_path)

model_to_coco_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def process_video(video_path, model, frame_skip=1):
    base_folder, video_file = os.path.split(video_path)
    video_folder_name, _ = os.path.splitext(video_file)
    unique_id = str(uuid.uuid4())
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılamıyor"
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        img_height, img_width = frame.shape[:2]
        
        if frame_count % frame_skip == 0:
            results = model([frame], device="0", conf=0.42)
            
            for result in results:
                if len(result.keypoints.data) > 0:
                    file_base = f"{unique_id}_{frame_count}"

                    # Save the frame as an image
                    save_frame(frame, file_base, IMAGES_DIR)

                    # Write COCO format for each frame as a separate JSON file
                    coco_format = create_coco_format(result.boxes, result.keypoints.data, file_base, img_width, img_height)
                    save_coco_json(coco_format, file_base, LABELS_DIR)
                    
        frame_count += 1

    cap.release()

def create_coco_format(boxes, keypoints, file_base, img_width, img_height):
    coco_format = {
        "info": {
            "year": "2024",
            "version": "2",
            "description": "Pose estimation dataset",
            "contributor": "",
            "url": "",
            "date_created": "2024-08-27T14:15:32+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0"
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "pose",
               "keypoints": [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16"
            ]

            }
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 0
    image_info = {
        "id": image_id,
        "license": 1,
        "file_name": f"{file_base}.jpg",
        "height": img_height,
        "width": img_width,
        "date_captured": "2024-08-27T14:15:32+00:00"
    }
    coco_format["images"].append(image_info)

    annotation_id = 0
    for i in range(len(boxes)):
        box = boxes.xywhn[i].tolist()
        x_center, y_center, width, height = box
        x_min = max(0, (x_center - width / 2) * img_width)
        y_min = max(0, (y_center - height / 2) * img_height)
        x_max = min(img_width, (x_center + width / 2) * img_width)
        y_max = min(img_height, (y_center + height / 2) * img_height)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        keypoints_coco = []
        keypoints_person = keypoints[i].data.tolist()

        keypoints_person_sorted = [keypoints_person[idx] for idx in model_to_coco_map]
        
        num_valid_keypoints = 0  
        for kp in keypoints_person_sorted:
            px = min(max(0, kp[0]), img_width)
            py = min(max(0, kp[1]), img_height)

            visibility = kp[2]

            if visibility < 0.5:
                visibility = 0
            else:
                visibility = 2

            if px <= 0 and py <= 0:
                visibility = 0 

            keypoints_coco += [px, py, int(visibility)]
            if visibility > 0:  
                num_valid_keypoints += 1

        if num_valid_keypoints > 0: 
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "keypoints": keypoints_coco,
                "num_keypoints": num_valid_keypoints  
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1
    
    return coco_format

def save_coco_json(coco_format, file_base, folder):
    json_file_name = f"{file_base}.json"
    json_output_path = os.path.join(folder, json_file_name)
    
    with open(json_output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

def save_frame(frame, file_base, folder):
    file_name = f"{file_base}.jpg"
    output_path = os.path.join(folder, file_name)
    cv2.imwrite(output_path, frame)

process_video(video_source, model, frame_skip=25)

