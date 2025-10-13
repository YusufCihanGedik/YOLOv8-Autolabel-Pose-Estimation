import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
import json

# ------------------ Konfig ------------------
yolo_model_path = r'/home/zbook/emre/local-working/models/yolov8x-pose.pt'
video_source = r"/home/zbook/emre/local-working/tofas-videos/D43_20250909085959.mp4"

OUTPUT_DIR = r'/home/zbook/emre/local-working/preprocess-dataset/multiclass/D43_5959'
LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

model = YOLO(yolo_model_path)

# YOLOv8 pose -> COCO keypoint sırası (aynı sıra, kimlik map'i)
model_to_coco_map = list(range(17))  # [0..16]

# ------------------ COCO Kategorisi ------------------
# ### DEĞİŞİKLİK: Roboflow'un keypoint'leri tanıması için 'keypoints' ve 'skeleton' eklendi
COCO_KEYPOINTS_17 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# COCO 17 için yaygın skeleton (1-based index!)
COCO_SKELETON_17 = [
    [16,14],[14,12],[17,15],[15,13],[12,13],
    [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],
    [2,3],[1,2],[1,3],[2,4],[3,5]
]

categories = [
    {
        "id": 1,
        "name": "person",
        "supercategory": "pose",
        # ### DEĞİŞİKLİK:
        "keypoints": COCO_KEYPOINTS_17,
        "skeleton": COCO_SKELETON_17
    }
]

def process_video(video_path, model, frame_skip=50, images_per_folder=100000):
    base_folder, video_file = os.path.split(video_path)
    video_folder_name, _ = os.path.splitext(video_file)
    unique_id = str(uuid.uuid4())
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılamıyor"
    
    frame_count = 0
    saved_images = 0
    folder_count = 1
    
    current_image_dir = os.path.join(IMAGES_DIR, f"folder_{folder_count}")
    current_label_dir = os.path.join(LABELS_DIR, f"folder_{folder_count}")
    
    os.makedirs(current_image_dir, exist_ok=True)
    os.makedirs(current_label_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        
        if frame_count % frame_skip == 0:
            # ### DEĞİŞİKLİK: conf değeri bırakıldı; device arg'ı int verilebilir
            results = model([frame], device=0, conf=0.42)
            
            for result in results:
                # result.keypoints.data: (num_det, 17, 3) -> (x, y, conf)
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    file_base = f"{unique_id}_{frame_count}"

                    # Görseli kaydet
                    save_frame(frame, file_base, current_image_dir)

                    # JSON formatında kaydet
                    coco_format = create_coco_format(
                        result.boxes,
                        result.keypoints.data,
                        file_base,
                        img_width,
                        img_height
                    )
                    save_coco_json(coco_format, file_base, current_label_dir)
                    
                    saved_images += 1
                    
                    # images_per_folder adet görüntüden sonra yeni klasör
                    if saved_images % images_per_folder == 0:
                        folder_count += 1
                        current_image_dir = os.path.join(IMAGES_DIR, f"folder_{folder_count}")
                        current_label_dir = os.path.join(LABELS_DIR, f"folder_{folder_count}")
                        
                        os.makedirs(current_image_dir, exist_ok=True)
                        os.makedirs(current_label_dir, exist_ok=True)
                    
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
        # ### DEĞİŞİKLİK: Kategoriler keypoints+skeleton ile burada
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    # Bu JSON tek bir imaj içeriyor; o yüzden image_id=0 kalabilir
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
    num_dets = len(boxes)
    for i in range(num_dets):
        # ### DEĞİŞİKLİK: Tensörleri CPU'ya al, sonra listele
        box_whn = boxes.xywhn[i].detach().cpu().tolist()  # [xc, yc, w, h] normalized
        x_center, y_center, width, height = box_whn

        # Normalize -> piksel bbox (xywh)
        x_min = max(0.0, (x_center - width / 2.0) * img_width)
        y_min = max(0.0, (y_center - height / 2.0) * img_height)
        x_max = min(float(img_width),  (x_center + width / 2.0) * img_width)
        y_max = min(float(img_height), (y_center + height / 2.0) * img_height)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        # --- Keypoints ---
        keypoints_coco = []

        # ### DEĞİŞİKLİK: Güvenli dönüşüm (CUDA -> CPU) ve direkt tensörden alın
        # keypoints: (num_det, 17, 3) ; i: (17,3)
        kps_i = keypoints[i].detach().cpu().tolist()  # [[x,y,conf], ...] len=17

        # YOLOv8 sırası COCO ile aynı; yine de map üzerinden sıralıyoruz
        keypoints_person_sorted = [kps_i[idx] for idx in model_to_coco_map]
        
        num_valid_keypoints = 0
        for kp in keypoints_person_sorted:
            px = float(np.clip(kp[0], 0, img_width))
            py = float(np.clip(kp[1], 0, img_height))

            conf = float(kp[2])  # [0..1]
            # ### DEĞİŞİKLİK: COCO v (0/1/2) üretimi
            # conf<0.5 -> v=0, aksi halde v=2 (görünür)
            visibility = 2 if conf >= 0.5 else 0

            # (opsiyonel) eğer nokta sınır dışıysa v=0
            if (px <= 0.0 and py <= 0.0) or (px >= img_width and py >= img_height):
                visibility = 0

            keypoints_coco += [px, py, int(visibility)]
            if visibility > 0:
                num_valid_keypoints += 1

        if num_valid_keypoints > 0:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # person
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "keypoints": keypoints_coco,  # [x1,y1,v1, ..., x17,y17,v17]
                "num_keypoints": num_valid_keypoints
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1
    
    return coco_format

def save_coco_json(coco_format, file_base, folder):
    json_file_name = f"{file_base}.json"
    json_output_path = os.path.join(folder, json_file_name)
    
    with open(json_output_path, 'w') as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=2)

def save_frame(frame, file_base, folder):
    file_name = f"{file_base}.jpg"
    output_path = os.path.join(folder, file_name)
    cv2.imwrite(output_path, frame)

process_video(video_source, model, frame_skip=50)
