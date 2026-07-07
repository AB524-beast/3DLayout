import cv2
import numpy as np
import requests
import json

# Simple test: white image with dark rectangle (simulates outer wall)
img = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (80, 80), (420, 420), (0, 0, 0), 6)

_, buf = cv2.imencode('.png', img)
files = {'file': ('test.png', buf.tobytes(), 'image/png')}
r = requests.post('http://127.0.0.1:8000/api/v1/process-layout/image?floors=1&method=auto', files=files)
print(f'Status: {r.status_code}')
data = r.json()
print(f'TotalRooms: {data["totalRooms"]}')
if data["rooms"]:
    for room in data["rooms"]:
        print(f'  {room["label"]}: {room["dimensions"]}, area={room["area"]}')
else:
    print('No rooms detected')
    print(json.dumps(data, indent=2))
