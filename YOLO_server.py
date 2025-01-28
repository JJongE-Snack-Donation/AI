from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.binary import Binary
from collections import Counter

#json data 직렬 전송 warning  관련 해결 코드 flask == 3.0.3 필수!!
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, Binary):
            return str(obj)
        return super().default(obj)

app = Flask(__name__)
app.json = CustomJSONProvider(app)

# MongoDB 연결 설정
client = MongoClient('mongodb://localhost:27017/')

# db 클라이언트 지정
db = client['images']

# raw 이미지가 저장되어 있는 collection
image_collection = db['raw_images']

# 데이터 처리가 완료된 이미지가 저장되는 collection
results_collection = db['detect_images']

# 실패한 이미지가 저장되는 collection
failed_detect_collection = db['failed_results']

# YOLOv8 모델 로드
model = YOLO('./ai_model/best.pt')

#객체 count를 위한 f(x)
def add_object_counts_to_image_result(detections, model):
    object_counts = {
        'deer': 0,
        'pig': 0,
        'racoon': 0
    }

    for detection in detections.boxes.data:
        _, _, _, _, confidence, class_id = detection
        if confidence >= 0.8:  # 80% 이상의 확률을 가진 객체만 처리
            class_name = model.names[int(class_id)]
            if class_name in object_counts:
                object_counts[class_name] += 1

    return object_counts

@app.route('/detect', methods=['POST'])
def detect_objects():
    # MongoDB에서 이미지 ID 리스트 받기
    image_ids = request.json['image_ids']
    results = []
    count = 0

    for image_id in image_ids:
        # MongoDB에서 이미지 검색
        image_doc = image_collection.find_one({'_id': ObjectId(image_id)})
        if not image_doc:
            continue
        
        # 이미지 이름 가져오기
        image_name = image_doc['filename']
        print(f"처리하는 이미지 파일 이름: {image_name}")

        # 이미지 데이터를 numpy 배열로 변환
        image_data = image_doc['data']
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 객체 검출 수행
        detections = model(image)[0]
        count += 1

        # 결과 이미지 생성 (바운딩 박스만 표시) 및 Ram 메모리에 임시할당
        img = image.copy()
        annotator = Annotator(img)

        for box in detections.boxes:
            b = box.xyxy[0]  # 바운딩 박스 좌표를 리스트에서 첫 번째 요소로 가져옴
            annotator.box_label(b.tolist(), color=(255, 0, 0))  # b를 리스트로 변환하여 전달

        # RGB에서 BGR로 변환
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 결과 이미지를 바이너리 데이터로 변환
        _, img_encoded = cv2.imencode('.jpg', img_bgr)
        resultImage_binaryData = Binary(img_encoded.tobytes())

        # 객체 카운트 추가
        object_counts = add_object_counts_to_image_result(detections, model)

        # 검출 결과 처리
        image_result = {
            'Image_id': str(image_id),
            'Filename': image_name,
            'Status': 'Success',
            'Detection_binaryData_image': resultImage_binaryData,
            'Detections': [],
            'Object_counts': object_counts
        }
        
        #if : 참지된 객체가 없을시 , else : running result data spend
        if len(detections.boxes.data.tolist()) == 0:
            image_result['Status'] = 'Failed'
            image_result['Reason'] = 'No objects detected'
            failed_detect_collection.insert_one(image_result)
        else:
            valid_detections = 0
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = detection

                print(f"확률값 확인 (%): {confidence}")
                # 80% 이상의 확률을 가진 객체만 처리
                if confidence >= 0.8:
                    valid_detections += 1
                    class_name = model.names[int(class_id)]
                    image_result['Detections'].append({
                        'class': class_name,
                        'confidence': float(confidence * 100),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            if valid_detections > 0:
                results_collection.insert_one(image_result)
            else:
                # 80% 이상의 확률을 가진 객체가 없는 경우
                image_result['Status'] = 'Failed'
                image_result['Reason'] = 'No objects with confidence >= 80%'
                failed_detect_collection.insert_one(image_result)

        results.append(image_result)

    return jsonify({
        'message': 'Detection completed and results saved to MongoDB',
        'image_count': count
    }), 200

if __name__ == '__main__':  
   app.run(host='0.0.0.0', port=5000, debug=True)
