import os
from ultralytics import YOLO
import cv2

model = YOLO(r'runs/detect/train-3/weights/best.pt')
input_path = r'D:\Projects\plate_blur\test\images'
output_path = r'D:\Projects\plate_blur\blurred_results'

os.makedirs(output_path, exist_ok=True)

for image_name in os.listdir(input_path):
    img = cv2.imread(os.path.join(input_path, image_name))
    results = model(img)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(plate, (101, 101), 0)
            img[y1:y2, x1:x2] = blurred
    
    cv2.imwrite(os.path.join(output_path, f'blurred_{image_name}'), img)

print("Done!")