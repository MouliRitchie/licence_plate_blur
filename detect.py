import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train-3/weights/best.pt')

img = cv2.imread('car.jpg')
results = model(img)

for result in results:
    print(result.boxes)  # add this
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = img[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(plate, (51, 51), 0)
        img[y1:y2, x1:x2] = blurred

cv2.imwrite('result.jpg', img)
cv2.imshow('Result', img)
cv2.waitKey(0)