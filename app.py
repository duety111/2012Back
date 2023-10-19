import os
import cv2
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# 이미지 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# YOLOv4 설정
yolo_net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4.cfg")
with open("yolov4/coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# HTML 폼을 표시하는 라우트
@app.route('/')
def upload_form():
    return render_template('upload.html')

# 이미지 업로드 및 분석 처리
@app.route('/', methods=['POST'])
def upload_image():
    result = ""
    image_path = None

    # 이미지 파일 업로드 및 저장
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            # 이미지 파일 저장
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # 이미지를 OpenCV로 읽기
            image = cv2.imread(image_path)
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

            # YOLOv4 모델로 객체 감지 수행
            yolo_net.setInput(blob)
            outs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # 객체 감지 결과 저장
                        center_x = int(detection[0] * image.shape[1])
                        center_y = int(detection[1] * image.shape[0])
                        w = int(detection[2] * image.shape[1])
                        h = int(detection[3] * image.shape[0])
                        x = center_x - w // 2
                        y = center_y - h // 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    result += f"객체: {label}, 신뢰도: {confidence:.2f}<br>"

    return render_template('upload.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
