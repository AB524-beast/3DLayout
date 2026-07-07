import cv2
import numpy as np
import requests
import json

# Create a 3-room floor plan
img = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Outer walls (thick)
cv2.rectangle(img, (40, 40), (760, 560), (0, 0, 0), 8)

# Internal wall: vertical divider at x=300, with door gap
cv2.line(img, (300, 40), (300, 260), (0, 0, 0), 6)
cv2.line(img, (300, 320), (300, 560), (0, 0, 0), 6)
# Door gap between y=260 and y=320

# Internal wall: horizontal divider at y=300, with door gap
cv2.line(img, (40, 300), (240, 300), (0, 0, 0), 6)
cv2.line(img, (360, 300), (760, 300), (0, 0, 0), 6)
# Door gap between x=240 and x=360

_, buf = cv2.imencode('.png', img)
files = {'file': ('plan3.png', buf.tobytes(), 'image/png')}
r = requests.post('http://127.0.0.1:8000/api/v1/process-layout/image?floors=1&method=auto', files=files)
print(f'Status: {r.status_code}')
data = r.json()
print(f'TotalRooms: {data["totalRooms"]}')
for room in data["rooms"]:
    n_walls = len(room["walls"])
    xs = [w["x1"] for w in room["walls"]] + [w["x2"] for w in room["walls"]]
    ys = [w["y1"] for w in room["walls"]] + [w["y2"] for w in room["walls"]]
    print(f'  {room["label"]}: cx={room["centerX"]:.2f}, cy={room["centerY"]:.2f}, dim={room["dimensions"]}, area={room["area"]}, walls={n_walls}')
