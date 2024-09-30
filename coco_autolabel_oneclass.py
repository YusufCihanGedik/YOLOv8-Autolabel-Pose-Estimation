import cv2
import os
from ultralytics import YOLO


def process_image(file_path, model_path, output_folder):
    model = YOLO(model_path)
    
    images = os.listdir(file_path)
    
    for image_name in images:
        full_image_path = os.path.join(file_path, image_name)
        print(f"Processing image: {full_image_path}")

        img = cv2.imread(full_image_path)
        
        if img is None:
            print(f"Error loading image {full_image_path}")
            continue 
        
        detect_poses(model, img, full_image_path, output_folder)


def detect_poses(model, img, image_path, output_folder):
    result = model(img)

    boxes_to_save = []

    for res in result[0].boxes:
        xmin, ymin, xmax, ymax = res.xyxy[0]
        confidence = res.conf[0]
        clas = int(res.cls[0])  # class numarası
        
        print("class:", clas)

        if clas == 5:
            x_min, y_min, x_max, y_max = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            img_height, img_width = img.shape[:2]
            box_center_x = (x_min + x_max) / 2.0 / img_width
            box_center_y = (y_min + y_max) / 2.0 / img_height
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height
            
            boxes_to_save.append((5, box_center_x, box_center_y, box_width, box_height))

    if boxes_to_save:
        write_boxes_to_txt(boxes_to_save, image_path, output_folder)

    cv2.imshow("Detection", cv2.resize(img, (1000, 1000)))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def write_boxes_to_txt(boxes, image_path, output_folder):
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    file_name = f"{file_base}.txt"

    output_path = os.path.join(output_folder, file_name)
    
    with open(output_path, 'w') as file:
        for class_id, x, y, w, h in boxes:
            line = f"{class_id} {x} {y} {w} {h}\n"
            file.write(line)


if __name__ == "__main__":
    model_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\16_09.pt"
    file_path = r"yolov8pose_output\kesim_veri\images"
    
    output_folder = r"yolov8pose_output\kesim_veri\labels"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_image(file_path, model_path, output_folder)
