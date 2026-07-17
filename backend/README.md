# Backend — 3D Layout API

FastAPI server that processes blueprint images and detects rooms using computer vision and ML.

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in this directory:

```env
ENVIRONMENT=development
PORT=8000
CORS_ORIGINS=*
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
OTEL_SERVICE_NAME=3d-layout-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## Running

```bash
python main.py
```

Server starts at `http://localhost:8000`.

## Docker

```bash
docker build -t 3d-layout-backend .
docker run -p 8000:8000 3d-layout-backend
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health + model availability status |
| `POST` | `/api/v1/process-layout/image?floors=N` | Upload a blueprint image for room detection |
| `GET` | `/api/v1/process-layout/sample?floors=N` | Get a sample layout from `blueprint_rooms.json` |
| `POST` | `/api/v1/process-layout/json?floors=N` | Process manual room polygon data |
| `POST` | `/api/v1/process-layout/procedural` | Generate layout from square footage specs |
| `POST` | `/api/v1/projects` | Save a project (requires `Authorization: Bearer <jwt>`) |

## Processing Pipeline

1. Image validation (size, format, dimensions)
2. Multi-pass wall detection — 4 sensitivity profiles using Canny edge detection + Hough lines + Otsu thresholding; CAD-style double-line wall detection fallback
3. Watershed segmentation to isolate individual rooms
4. Polygon simplification — RDP approximation, short-edge collapse, orthogonal snapping, Shapely validity repair
5. Scoring — picks best result based on coverage, room count, overlap penalties
6. ML fallback — if OpenCV result is implausible, runs the ONNX model (`models/room_segmenter.onnx`) for semantic segmentation (10 classes)
7. Coordinate transform — pixel space to meter space (14.0m reference height)

## Response Format

```json
{
  "rooms": [
    {
      "label": "Room 1",
      "dimensions": "5.2m x 3.8m",
      "centerX": 0.5,
      "centerY": 0.3,
      "elevationZ": 0.0,
      "isOpenSpace": false,
      "walls": [{ "x1": 0, "y1": 0, "x2": 5.2, "y2": 0 }],
      "area": 19.76
    }
  ],
  "totalRooms": 5,
  "totalFloors": 1,
  "calculatedSqFt": 212.7,
  "segmentationMethod": "opencv",
  "processingTimeMs": 452.1
}
```

## Database

Uses Supabase (PostgreSQL). Schema migrations are in `migrations/`:

- `001_projects_tables.sql` — creates `projects` and `project_rooms` tables

## Project Structure

```
backend/
├── main.py                 # FastAPI app and routes
├── model_inference.py      # ONNX ML room segmentation
├── database.py             # Supabase DB operations
├── tracing.py              # OpenTelemetry setup
├── requirements.txt
├── Dockerfile
├── .env.example
├── migrations/
│   └── 001_projects_tables.sql
├── models/
│   └── room_segmenter.onnx
└── blueprint_rooms.json    # Sample layout data
```
