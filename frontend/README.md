# Frontend — Blueprint Spatial Modeler

Next.js 16 application with a Three.js 3D viewport, SVG correction editor, and Supabase auth.

## Setup

```bash
cd frontend
npm install
```

## Environment Variables

Create a `.env.local` file in this directory:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Running

```bash
npm run dev        # http://localhost:3000
npm run build      # production build
npm run start      # production server
npm run lint       # ESLint
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Main page — upload blueprints, view 3D models, edit layouts |
| `/login` | Sign in / create account |
| `/dashboard` | Browse saved layouts, download, delete |

## Key Components

| Component | Purpose |
|-----------|---------|
| `BlueprintUploader` | Drag-and-drop image upload, sample layout loader |
| `RoomExtrusionCanvas` | Three.js 3D viewport with extruded room walls |
| `RoomCorrectionEditor` | SVG polygon editor — drag vertices/rooms, add/delete points, undo/redo, grid snap, dimension & area labels |
| `NavBar` | Top navigation bar with auth controls |
| `DockNav` | macOS-style animated bottom navigation dock |
| `GridScan` | Animated 3D grid background with scan-line shader, bloom, chromatic aberration |

## Editor Interactions

- **Click room** — select for vertex editing
- **Drag room fill** — move the entire room
- **Shift+drag empty space** — move all rooms together
- **Drag vertex** — move individual vertex (snaps to grid when Grid is on)
- **Double-click edge** — add vertex at midpoint
- **Right-click edge/vertex** — delete
- **Ctrl+Z / Ctrl+Shift+Z** — undo / redo
- **G** — toggle grid snap
- **D** — toggle dimension labels
- **Del** — remove selected vertex
- **Escape** — deselect all

## Project Structure

```
frontend/src/
├── app/
│   ├── layout.tsx          # Root layout (NavBar + DockNav + AuthProvider)
│   ├── page.js             # Home page — orchestrates upload/view/edit panels
│   ├── globals.css
│   ├── login/page.js       # Auth page
│   └── dashboard/page.js   # Saved layouts dashboard
├── components/
│   ├── BlueprintUploader.js
│   ├── RoomExtrusionCanvas.js
│   ├── RoomCorrectionEditor.js
│   ├── NavBar.js
│   ├── Dock/
│   │   ├── Dock.js
│   │   └── DockNav.js
│   └── GridScan/
│       └── GridScan.js
├── context/
│   └── AuthContext.js      # Supabase auth & layout persistence
├── lib/
│   └── supabaseClient.js   # Supabase client singleton
└── instrumentation.ts      # OpenTelemetry client setup
```
