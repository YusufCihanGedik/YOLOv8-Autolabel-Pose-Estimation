
# import cv2
# from ultralytics import YOLO
# import os 


# # Define the directory where you want to save the images
# OUTPUT_DIR = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\yolov8pose_images'

# # Create the output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Path to the trained model file
# model_file = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\yolov8m-pose.pt"

# # Load the trained model
# model = YOLO(model_file)

# # List all files in the folder
# # video_files = os.listdir(TESTING_DIR)


# # for video_file in video_files:
# while True:
#     # cap = cv2.VideoCapture(os.path.join(TESTING_DIR, video_file))
#     cap = cv2.VideoCapture(r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\11.mp4")

#     # Get the dimensions and frame rate of the video
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#     # Create the output video writer
#     out = cv2.VideoWriter(f'{OUTPUT_DIR}_pose_estimation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

#     # Read the first frame
#     ret, frame = cap.read()

#     while ret:
#         # Perform pose estimation on the current frame
#         results = model(frame)[0]

#         for result in results:
#             # Extract bounding box coordinates, score, and class ID
#             x1, y1, x2, y2, score, class_id = result.boxes.data.tolist()[0]

#             # Draw bounding box
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

#             # Display class name and score
#             label = f'{results.names[int(class_id)]}: {score:.2f}'
#             cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

#             for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()[0]):
#                 # Draw keypoints
#                 cv2.putText(frame, str(keypoint_index), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Write the annotated frame to the output video
#         out.write(frame)

#         # Read the next frame
#         ret, frame = cap.read()

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid

# Model paths
yolo_model_path = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\yolov8x-pose.pt'
video_source = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\KESIM_BASKENT_20240730085000_20240730095909_93482899.mp4"

# Auto-labeling için dizin yapısını ayarlıyoruz
OUTPUT_DIR = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\yolov8pose_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modeli yükle
model = YOLO(yolo_model_path)

def process_video(video_path, model, frame_skip=1):
    # Dosya yolunu ayrıştır
    base_folder, video_file = os.path.split(video_path)
    video_folder_name, _ = os.path.splitext(video_file)

    # Benzersiz bir ID oluştur
    unique_id = str(uuid.uuid4())

    # Yeni çıktı klasörlerini oluştur
    labelling_base = os.path.join(OUTPUT_DIR, video_folder_name)
    detection_images_folder = os.path.join(labelling_base, "images_with_detections")
    labels_folder = os.path.join(labelling_base, "labels")
    os.makedirs(detection_images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Video aç
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılamıyor"

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Her 'frame_skip' karede bir işlem yap
        if frame_count % frame_skip == 0:
            results = model([frame], device="0", conf=0.30)
            human_detected = False

            for result in results:
                if len(result.keypoints.data) > 0:
                    human_detected = True
                    file_base = f"{unique_id}_{frame_count}"

                    # Pose bilgilerini yazdırmak için txt dosyasını kaydet
                    write_pose_to_txt(result.boxes, result.keypoints.data, file_base, labels_folder, dim=3)

                    # Kareyi kaydet
                    save_frame(frame, file_base, detection_images_folder)

            frame_count += 1

        if not human_detected:
            file_base = f"{unique_id}_{frame_count}"
            save_frame(frame, file_base, detection_images_folder)

    cap.release()

def write_pose_to_txt(boxes, keypoints, file_base, output_folder, dim=2):
    """
    Pose keypoints bilgilerini YOLOv8-Pose formatına göre yazdırır.
    Dim = 2 veya Dim = 3 seçeneklerini destekler.
    """
    file_name = f"{file_base}.txt"
    output_path = os.path.join(output_folder, file_name)
    
    with open(output_path, 'w') as file:
        for i, box in enumerate(boxes.xywhn):  # YOLOv8 Pose için box formatı (class, x_center, y_center, width, height)
            class_id = int(boxes.cls[i])  # Nesne sınıfı indeksi
            
            # Kutu bilgilerini normalleştirilmiş formatta yazdır (x_center, y_center, width, height)
            line = f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}"
            
            if dim == 2:
                # Dim = 2: Görünürlük olmadan anahtar noktaları ekleyin
                for kp in keypoints[i]:
                    line += f" {kp[0]} {kp[1]}"  # Her bir anahtar noktasının (px, py) bilgilerini yazdır
                
            elif dim == 3:
                # Dim = 3: Görünürlük bilgisiyle birlikte anahtar noktalarını ekleyin
                for kp in keypoints[i]:
                    line += f" {kp[0]} {kp[1]} {kp[2]}"  # Her bir anahtar noktasının (px, py, visibility) bilgilerini yazdır
            
            # Satırın sonuna kadar yaz ve bir sonraki satıra geç
            file.write(line + "\n")  # Burada her bir keypoint seti bir satıra yazılacak

def save_frame(frame, file_base, folder):
    """
    Tespit edilen frame'i JPEG formatında kaydeder.
    """
    file_name = f"{file_base}.jpg"
    output_path = os.path.join(folder, file_name)
    cv2.imwrite(output_path, frame)

# Video dosyasını işliyoruz
process_video(video_source, model, frame_skip=1)
