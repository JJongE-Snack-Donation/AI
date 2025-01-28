import requests
import json

# Flask 서버의 URL
url = 'http://localhost:5000/detect'

# MongoDB에 저장된 이미지의 ID 리스트
image_ids = [
    "678def6f3d9346a8a2d5302c",
    "678def6f3d9346a8a2d5302d",
    "678def6f3d9346a8a2d5302e",
    "678def6f3d9346a8a2d5302f",
    "678def6f3d9346a8a2d53030",
    "678def6f3d9346a8a2d53031",
    "678def6f3d9346a8a2d53032",
    "678def6f3d9346a8a2d53033"
]

# 요청 데이터 준비
data = {
    "image_ids": image_ids
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 확인
if response.status_code == 200:
    print("요청 성공:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"요청 실패. 상태 코드: {response.status_code}")
    print(response.text)
