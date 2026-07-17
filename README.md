# Blueprint Spatial Modeler

A full-stack web application that converts uploaded building blueprint / floor plan images into interactive 3D models. The backend analyzes images using computer vision (OpenCV) and optionally an ONNX ML model to detect rooms and walls, then the frontend renders the result as a navigable 3D scene with Three.js. Detected rooms can be refined in a 2D polygon editor before final visualization.

## Tech Stack

### Frontend

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16.2 (React 19.2) |
| Language | JavaScript (pages/components), TypeScript (config/layout) |
| 3D Rendering | Three.js v0.185 with OrbitControls |
| Post-processing | Bloom, Chromatic Aberration (postprocessing v6) |
| Styling | Tailwind CSS v4 |
| Animation | Framer Motion v12 (Dock UI) |
| Auth / DB | Supabase JS v2.49 |
| Face Tracking | face-api.js v0.22 (optional webcam parallax) |
| Telemetry | OpenTelemetry (client + server) |

### Backend

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (Python 3.11) |
| Server | Uvicorn |
| Image Processing | OpenCV 4.11, Shapely 2.1 |
| ML Inference | ONNX Runtime 1.24 (`models/room_segmenter.onnx`) |
| Database | Supabase (PostgreSQL with Row-Level Security) |
| Telemetry | OpenTelemetry (FastAPI instrumentation, OTLP export) |

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org) 18+
- [Python](https://python.org) 3.11+
- [Supabase](https://supabase.com) project (for auth and persistence)
- Docker (optional, for backend containerization)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/3DLayout.git
cd 3DLayout

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Environment Variables

**Frontend** — create `frontend/.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

**Backend** — create `backend/.env`:

```env
ENVIRONMENT=development
PORT=8000
CORS_ORIGINS=*
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
OTEL_SERVICE_NAME=3d-layout-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

### Running

```bash
# From the project root
npm run dev:frontend   # http://localhost:3000
npm run dev:backend    # http://localhost:8000
```

Or run them separately:

```bash
# Frontend
cd frontend && npm run dev

# Backend
cd backend && python main.py
```

### Docker (Backend)

```bash
cd backend
docker build -t 3d-layout-backend .
docker run -p 8000:8000 3d-layout-backend
```

---

## Features

### Blueprint Upload
Drag-and-drop or file-select an image. The backend detects rooms using multi-pass OpenCV wall detection with ML fallback. Floor count is configurable before upload.

### 3D Visualization
Interactive Three.js viewport with extruded room walls, orbit controls (rotate/zoom/pan), floor-level filtering for multi-story buildings, and blueprint image overlay.

### 2D Correction Editor
Full SVG-based polygon editor for refining detected rooms:

| Action | How |
|--------|-----|
| Select a room | Click on the room fill |
| Move entire room | Click and drag the room fill |
| Move all rooms | Shift + drag on empty space |
| Move a vertex | Select room, then drag a vertex |
| Add vertex on edge | Select room, double-click on an edge |
| Delete vertex | Select room, right-click or double-click a vertex, or press Del |
| Delete edge | Select room, right-click on an edge |
| Undo / Redo | Ctrl+Z / Ctrl+Shift+Z |
| Toggle grid snap | Press G |
| Toggle dimensions | Press D |
| Toggle area labels | Click "Area" in toolbar |
| Pan canvas | Middle-mouse drag |
| Zoom | Scroll wheel |
| Add new room | Click "+ Room" in toolbar |
| Rename room | Double-click room name in sidebar |
| Toggle open/closed | Click "Open"/"Closed" badge in sidebar |
| Delete room | Hover room in sidebar, click "Del" |

### Multi-Floor Support
Floor selector buttons appear in the 3D viewer when the layout has multiple floors.

### Save to Dashboard
Authenticated users can save layouts to Supabase. Saved layouts appear on the Dashboard page with download and delete options.

### Export
- **Screenshot** — downloads the 3D canvas as a PNG
- **JSON** — downloads the full layout data as a `.json` file

### Dashboard
Browse all saved layouts as cards. Download blueprint images and layout JSON. Delete layouts.

### Animated Background
The home page features a 3D perspective grid with an animated scanning light beam, bloom, chromatic aberration, noise, and optional webcam face-tracking parallax.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — `{"status": "running"}` |
| `GET` | `/health` | Health + model availability — `{"status": "ok", "model_available": bool}` |
| `POST` | `/api/v1/process-layout/image?floors=N` | Upload a blueprint image for room detection |
| `GET` | `/api/v1/process-layout/sample?floors=N` | Get a sample layout from `blueprint_rooms.json` |
| `POST` | `/api/v1/process-layout/json?floors=N` | Process manual room polygon data |
| `POST` | `/api/v1/process-layout/procedural` | Generate layout from square footage specs |
| `POST` | `/api/v1/projects` | Save a project (requires `Authorization: Bearer <jwt>`) |

### Response Format (process-layout endpoints)

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
      "walls": [
        { "x1": 0, "y1": 0, "x2": 5.2, "y2": 0 },
        { "x1": 5.2, "y1": 0, "x2": 5.2, "y2": 3.8 }
      ],
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

---

## Backend Processing Pipeline

1. **Image validation** — checks size, format, and dimensions
2. **Multi-pass wall detection** — 4 sensitivity profiles using Canny edge detection + Hough line detection + Otsu thresholding, plus CAD-style double-line wall detection fallback
3. **Watershed segmentation** — isolates individual rooms from the wall structure
4. **Polygon simplification** — RDP approximation, short-edge collapse, orthogonal snapping, Shapely validity repair
5. **Scoring** — picks the best result across all passes based on coverage, room count, and overlap penalties
6. **ML fallback** — if the OpenCV result is implausible, runs the ONNX model (`models/room_segmenter.onnx`) for semantic segmentation across 10 classes (Background, Bedroom, LivingRoom, Kitchen, Bathroom, Dining, Balcony, Storage, Hallway, Other)
7. **Coordinate transform** — converts pixel space to meter space using a 14.0m reference height

---

## Database Schema

Managed via Supabase. Migrations are in `backend/migrations/`.

### `projects`

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid | Primary key |
| `name` | text | Layout name |
| `user_id` | uuid | Foreign key to `auth.users` |
| `image_url` | text | Blueprint image URL in Supabase Storage |
| `total_floors` | integer | Number of floors |
| `created_at` | timestamptz | Creation timestamp |

### `project_rooms`

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid | Primary key |
| `project_id` | uuid | Foreign key to `projects` |
| `label` | text | Room name |
| `dimensions` | text | e.g. "5.2m x 3.8m" |
| `center_x` | float | Center X in meters |
| `center_y` | float | Center Y in meters |
| `elevation_z` | float | Floor elevation |
| `is_open_space` | boolean | Open plan flag |
| `walls` | jsonb | Array of wall segments |
| `area` | float | Area in square meters |

---

## Project Structure

```
3DLayout/
├── README.md                     # This file
├── package.json                  # Root monorepo scripts
├── vercel.json                   # Frontend deployment (Vercel)
├── render.yaml                   # Backend deployment (Render)
│
├── frontend/
│   ├── README.md                 # Frontend-specific docs
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── .env.example
│   └── src/
│       ├── app/
│       │   ├── layout.tsx        # Root layout (NavBar + DockNav + AuthProvider)
│       │   ├── page.js           # Home — orchestrates upload/view/edit panels
│       │   ├── globals.css
│       │   ├── login/page.js     # Auth page
│       │   └── dashboard/page.js # Saved layouts dashboard
│       ├── components/
│       │   ├── BlueprintUploader.js
│       │   ├── RoomExtrusionCanvas.js
│       │   ├── RoomCorrectionEditor.js
│       │   ├── NavBar.js
│       │   ├── Dock/
│       │   │   ├── Dock.js       # macOS-style animated dock
│       │   │   └── DockNav.js    # Dock navigation items
│       │   └── GridScan/
│       │       ├── GridScan.js   # Animated 3D grid background
│       │       └── GridScan.css
│       ├── context/
│       │   └── AuthContext.js    # Supabase auth & layout persistence
│       ├── lib/
│       │   └── supabaseClient.js # Supabase client singleton
│       └── instrumentation.ts    # Client-side OpenTelemetry
│
├── backend/
│   ├── README.md                 # Backend-specific docs
│   ├── main.py                   # FastAPI app and all routes
│   ├── model_inference.py        # ONNX ML room segmentation
│   ├── database.py               # Supabase DB CRUD
│   ├── tracing.py                # OpenTelemetry setup
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   ├── .dockerignore
│   ├── blueprint_rooms.json      # Sample layout data
│   ├── migrations/
│   │   └── 001_projects_tables.sql
│   └── models/
│       └── room_segmenter.onnx   # Trained ONNX model
│
├── test_detection.py             # Local test scripts
└── test_detection2.py
```

---

## Deployment

### Frontend (Vercel)

Configured via `vercel.json`. Push to GitHub and connect the repo on [vercel.com](https://vercel.com).

### Backend (Render)

Configured via `render.yaml`. Push to GitHub and create a Web Service on [render.com](https://render.com) pointing to the `backend/` directory with start command `python main.py`.

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev:frontend` | Start frontend dev server |
| `npm run dev:backend` | Start backend dev server |
| `npm run build` | Build frontend for production |

---

## License

MIT
