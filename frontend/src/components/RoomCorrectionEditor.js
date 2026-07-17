"use client";

import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
} from "react";

const PLANE_METERS = 14.0;
const MAX_HISTORY = 50;
const GRID_SNAP_SIZE = 20;

function wallsToPixelVertices(walls, pxToMeter, imgW, imgH) {
  if (!walls || walls.length === 0) return [];
  const seen = new Set();
  const vertices = [];
  for (const w of walls) {
    const key = `${w.x1.toFixed(6)},${w.y1.toFixed(6)}`;
    if (!seen.has(key)) {
      seen.add(key);
      vertices.push({
        x: w.x1 * pxToMeter + imgW / 2,
        y: w.y1 * pxToMeter + imgH / 2,
      });
    }
  }
  return vertices;
}

function pixelVerticesToMeters(vertices, pxToMeter, imgW, imgH) {
  return vertices.map((v) => [
    (v.x - imgW / 2) / pxToMeter,
    (v.y - imgH / 2) / pxToMeter,
  ]);
}

function metersToWalls(ptsM) {
  const walls = [];
  const n = ptsM.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = ptsM[i];
    const [x2, y2] = ptsM[(i + 1) % n];
    walls.push({ x1, y1, x2, y2 });
  }
  return walls;
}

function computeRoomFromVertices(
  vertices,
  pxToMeter,
  imgW,
  imgH,
  label,
  isOpenSpace,
  elevationZ
) {
  const ptsM = pixelVerticesToMeters(vertices, pxToMeter, imgW, imgH);
  const xs = ptsM.map((p) => p[0]);
  const ys = ptsM.map((p) => p[1]);
  const bbW = Math.max(...xs) - Math.min(...xs);
  const bbH = Math.max(...ys) - Math.min(...ys);
  const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;
  const centerY = ys.reduce((a, b) => a + b, 0) / ys.length;

  const n = ptsM.length;
  let polyArea = 0;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = ptsM[i];
    const [x2, y2] = ptsM[(i + 1) % n];
    polyArea += x1 * y2 - x2 * y1;
  }
  polyArea = Math.abs(polyArea) / 2;

  return {
    label,
    dimensions: `${bbW.toFixed(1)}m x ${bbH.toFixed(1)}m`,
    centerX,
    centerY,
    elevationZ: elevationZ || 0,
    isOpenSpace: !!isOpenSpace,
    walls: metersToWalls(ptsM),
    area: Math.round(polyArea * 100) / 100,
  };
}

function edgeLength(p1, p2, pxToMeter) {
  const dx = (p2.x - p1.x) / pxToMeter;
  const dy = (p2.y - p1.y) / pxToMeter;
  return Math.sqrt(dx * dx + dy * dy);
}

function roomPixelArea(vertices) {
  let area = 0;
  const n = vertices.length;
  for (let i = 0; i < n; i++) {
    const { x: x1, y: y1 } = vertices[i];
    const { x: x2, y: y2 } = vertices[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

let globalRoomIdCounter = 1;

const ROOM_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
  "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
  "#84cc16", "#e879f9", "#14b8a6", "#f43f5e",
];

function roomColor(idx) {
  return ROOM_COLORS[idx % ROOM_COLORS.length];
}

function historyReducer(state, action) {
  switch (action.type) {
    case "PUSH": {
      const newPast = [...state.past, state.present].slice(-MAX_HISTORY);
      return { past: newPast, present: action.rooms, future: [] };
    }
    case "UNDO": {
      if (state.past.length === 0) return state;
      const prev = state.past[state.past.length - 1];
      return {
        past: state.past.slice(0, -1),
        present: prev,
        future: [state.present, ...state.future].slice(0, MAX_HISTORY),
      };
    }
    case "REDO": {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      return {
        past: [...state.past, state.present].slice(-MAX_HISTORY),
        present: next,
        future: state.future.slice(1),
      };
    }
    case "SET": {
      return { past: [], present: action.rooms, future: [] };
    }
    default:
      return state;
  }
}

export default function RoomCorrectionEditor({
  layoutData,
  imageUrl,
  onConfirm,
  onCancel,
  onSaveAndGoBack,
  saving,
}) {
  const [imgDims, setImgDims] = useState(null);
  const [selectedRoomId, setSelectedRoomId] = useState(null);
  const [selectedVertex, setSelectedVertex] = useState(null);
  const [hoveredEdge, setHoveredEdge] = useState(null);
  const [hoveredVertex, setHoveredVertex] = useState(null);
  const [editingLabel, setEditingLabel] = useState(null);
  const [gridSnap, setGridSnap] = useState(false);
  const [showDimensions, setShowDimensions] = useState(true);
  const [showArea, setShowArea] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [tooltip, setTooltip] = useState(null);
  const [activeTool, setActiveTool] = useState("select");
  const [isDragging, setIsDragging] = useState(false);

  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const dragRef = useRef(null);
  const panStartRef = useRef(null);
  const lastAddRef = useRef(0);

  const pxToMeter = imgDims ? imgDims.h / PLANE_METERS : 1;

  const initialRooms = useMemo(() => {
    if (!imgDims || !layoutData?.rooms) return null;
    return layoutData.rooms.map((r) => ({
      id: globalRoomIdCounter++,
      label: r.label || "Room",
      isOpenSpace: !!r.isOpenSpace,
      elevationZ: r.elevationZ || 0,
      vertices: wallsToPixelVertices(
        r.walls || [],
        imgDims.h / PLANE_METERS,
        imgDims.w,
        imgDims.h
      ),
    }));
  }, [imgDims, layoutData]);

  const [history, dispatch] = useReducer(historyReducer, {
    past: [],
    present: [],
    future: [],
  });

  const rooms = history.present;
  const canUndo = history.past.length > 0;
  const canRedo = history.future.length > 0;

  const [prevInitialKey, setPrevInitialKey] = useState(null);
  if (initialRooms && initialRooms !== prevInitialKey) {
    setPrevInitialKey(initialRooms);
    dispatch({ type: "SET", rooms: initialRooms });
  }

  const pushHistory = useCallback(
    (newRooms) => {
      dispatch({ type: "PUSH", rooms: newRooms });
    },
    [dispatch]
  );

  const handleImageLoad = (e) => {
    setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  };

  const snapToGrid = useCallback(
    (x, y) => {
      if (!gridSnap) return { x, y };
      return {
        x: Math.round(x / GRID_SNAP_SIZE) * GRID_SNAP_SIZE,
        y: Math.round(y / GRID_SNAP_SIZE) * GRID_SNAP_SIZE,
      };
    },
    [gridSnap]
  );

  const svgPoint = useCallback(
    (clientX, clientY) => {
      const svg = svgRef.current;
      if (!svg) return { x: 0, y: 0 };
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return { x: 0, y: 0 };
      const svgPt = pt.matrixTransform(ctm.inverse());
      return snapToGrid(svgPt.x, svgPt.y);
    },
    [snapToGrid]
  );

  const svgPointRaw = useCallback((clientX, clientY) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    return pt.matrixTransform(ctm.inverse());
  }, []);

  const handlePointerDown = useCallback(
    (roomId, vertexIdx) => (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (roomId !== selectedRoomId) return;
      e.target.setPointerCapture(e.pointerId);
      setSelectedVertex({ roomId, vertexIdx });
      dragRef.current = { roomId, vertexIdx, startX: e.clientX, startY: e.clientY };
      setIsDragging(false);
    },
    [selectedRoomId]
  );

  const handlePointerMove = useCallback(
    (e) => {
      const drag = dragRef.current;
      if (!drag) return;

      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        setIsDragging(true);
      }

      const { x, y } = svgPoint(e.clientX, e.clientY);
      const newRooms = history.present.map((r) => {
        if (r.id !== drag.roomId) return r;
        const newVertices = r.vertices.map((v, idx) =>
          idx === drag.vertexIdx ? { x, y } : v
        );
        return { ...r, vertices: newVertices };
      });
      dispatch({ type: "SET", rooms: newRooms });
    },
    [svgPoint, history.present]
  );

  const handlePointerUp = useCallback(() => {
    if (dragRef.current && isDragging) {
      pushHistory(history.present);
    }
    dragRef.current = null;
    setIsDragging(false);
  }, [history.present, isDragging, pushHistory]);

  useEffect(() => {
    const mp = (e) => handlePointerMove(e);
    const mu = () => handlePointerUp();
    window.addEventListener("pointermove", mp);
    window.addEventListener("pointerup", mu);
    return () => {
      window.removeEventListener("pointermove", mp);
      window.removeEventListener("pointerup", mu);
    };
  }, [handlePointerMove, handlePointerUp]);

  const handleEdgeDoubleClick = useCallback(
    (roomId, edgeIdx) => (e) => {
      e.stopPropagation();
      if (roomId !== selectedRoomId) return;
      const { x, y } = svgPoint(e.clientX, e.clientY);
      const newRooms = history.present.map((r) => {
        if (r.id !== roomId) return r;
        const newVertices = [...r.vertices];
        newVertices.splice(edgeIdx + 1, 0, { x, y });
        return { ...r, vertices: newVertices };
      });
      pushHistory(newRooms);
    },
    [svgPoint, history.present, pushHistory, selectedRoomId]
  );

  const handleVertexDelete = useCallback(
    (roomId, vertexIdx) => (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (roomId !== selectedRoomId) return;
      const newRooms = history.present.map((r) => {
        if (r.id !== roomId || r.vertices.length <= 3) return r;
        return {
          ...r,
          vertices: r.vertices.filter((_, i) => i !== vertexIdx),
        };
      });
      pushHistory(newRooms);
      setSelectedVertex(null);
    },
    [history.present, pushHistory, selectedRoomId]
  );

  const handleEdgeAddVertex = useCallback(
    (roomId, edgeIdx) => (e) => {
      e.stopPropagation();
      e.preventDefault();
      const now = Date.now();
      if (now - lastAddRef.current < 250) return;
      lastAddRef.current = now;
      const newRooms = history.present.map((r) => {
        if (r.id !== roomId) return r;
        const a = r.vertices[edgeIdx];
        const b = r.vertices[(edgeIdx + 1) % r.vertices.length];
        const midX = (a.x + b.x) / 2;
        const midY = (a.y + b.y) / 2;
        const newVertices = [...r.vertices];
        newVertices.splice(edgeIdx + 1, 0, { x: midX, y: midY });
        return { ...r, vertices: newVertices };
      });
      pushHistory(newRooms);
    },
    [history.present, pushHistory]
  );

  const handleEdgeDelete = useCallback(
    (roomId, edgeIdx) => (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (roomId !== selectedRoomId) return;
      const newRooms = history.present.map((r) => {
        if (r.id !== roomId || r.vertices.length <= 3) return r;
        const removeIdx = (edgeIdx + 1) % r.vertices.length;
        return {
          ...r,
          vertices: r.vertices.filter((_, i) => i !== removeIdx),
        };
      });
      pushHistory(newRooms);
      setHoveredEdge(null);
    },
    [history.present, pushHistory, selectedRoomId]
  );

  const handleVertexContextMenu = useCallback(
    (roomId, vertexIdx) => (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (roomId !== selectedRoomId) return;
      const newRooms = history.present.map((r) => {
        if (r.id !== roomId || r.vertices.length <= 3) return r;
        return {
          ...r,
          vertices: r.vertices.filter((_, i) => i !== vertexIdx),
        };
      });
      pushHistory(newRooms);
      setSelectedVertex(null);
    },
    [history.present, pushHistory, selectedRoomId]
  );

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          dispatch({ type: "REDO" });
        } else {
          dispatch({ type: "UNDO" });
        }
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault();
        dispatch({ type: "REDO" });
        return;
      }

      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedVertex && selectedRoomId) {
          const { roomId, vertexIdx } = selectedVertex;
          const newRooms = history.present.map((r) => {
            if (r.id !== roomId || r.vertices.length <= 3) return r;
            return {
              ...r,
              vertices: r.vertices.filter((_, i) => i !== vertexIdx),
            };
          });
          pushHistory(newRooms);
          setSelectedVertex(null);
        }
      }
      if (e.key === "Escape") {
        setSelectedVertex(null);
        setHoveredEdge(null);
        setSelectedRoomId(null);
        setActiveTool("select");
      }
      if (e.key === "g" && !e.ctrlKey && !e.metaKey) {
        setGridSnap((prev) => !prev);
      }
      if (e.key === "d" && !e.ctrlKey && !e.metaKey) {
        setShowDimensions((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedVertex, selectedRoomId, history.present, pushHistory]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handleWheel = (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((prev) => Math.min(5, Math.max(0.3, prev * delta)));
    };
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, []);

  const handleMiddlePointerDown = useCallback((e) => {
    if (e.button === 1) {
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    }
  }, [pan]);

  const handleMiddlePointerMove = useCallback(
    (e) => {
      if (!isPanning || !panStartRef.current) return;
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      setPan({
        x: panStartRef.current.panX + dx,
        y: panStartRef.current.panY + dy,
      });
    },
    [isPanning]
  );

  const handleMiddlePointerUp = useCallback(() => {
    setIsPanning(false);
    panStartRef.current = null;
  }, []);

  useEffect(() => {
    const mv = (e) => handleMiddlePointerMove(e);
    const mu = () => handleMiddlePointerUp();
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", mu);
    return () => {
      window.removeEventListener("pointermove", mv);
      window.removeEventListener("pointerup", mu);
    };
  }, [handleMiddlePointerMove, handleMiddlePointerUp]);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleAddRoom = () => {
    if (!imgDims) return;
    const cx = imgDims.w / 2;
    const cy = imgDims.h / 2;
    const half = Math.min(imgDims.w, imgDims.h) * 0.1;
    const newRoom = {
      id: globalRoomIdCounter++,
      label: `Room ${rooms.length + 1}`,
      isOpenSpace: false,
      elevationZ: 0,
      vertices: [
        { x: cx - half, y: cy - half },
        { x: cx + half, y: cy - half },
        { x: cx + half, y: cy + half },
        { x: cx - half, y: cy + half },
      ],
    };
    pushHistory([...rooms, newRoom]);
    setSelectedRoomId(newRoom.id);
  };

  const handleDeleteRoom = (roomId) => {
    pushHistory(rooms.filter((r) => r.id !== roomId));
    if (selectedRoomId === roomId) setSelectedRoomId(null);
  };

  const handleLabelChange = (roomId, newLabel) => {
    const newRooms = rooms.map((r) =>
      r.id === roomId ? { ...r, label: newLabel } : r
    );
    dispatch({ type: "SET", rooms: newRooms });
  };

  const handleLabelBlur = () => {
    setEditingLabel(null);
    pushHistory(rooms);
  };

  const handleToggleOpenSpace = (roomId) => {
    const newRooms = rooms.map((r) =>
      r.id === roomId ? { ...r, isOpenSpace: !r.isOpenSpace } : r
    );
    pushHistory(newRooms);
  };

  const handleConfirm = () => {
    if (!imgDims) return;
    const updatedRooms = rooms.map((r) =>
      computeRoomFromVertices(
        r.vertices,
        pxToMeter,
        imgDims.w,
        imgDims.h,
        r.label,
        r.isOpenSpace,
        r.elevationZ
      )
    );
    const calculatedSqFt =
      Math.round(
        updatedRooms.reduce((sum, r) => sum + r.area * 10.764, 0) * 10
      ) / 10;

    onConfirm({
      rooms: updatedRooms,
      totalRooms: updatedRooms.length,
      totalFloors: layoutData?.totalFloors || 1,
      calculatedSqFt,
      segmentationMethod: layoutData?.segmentationMethod || "manual",
    });
  };

  const viewBox = imgDims
    ? `0 0 ${imgDims.w} ${imgDims.h}`
    : "0 0 100 100";

  const selectedRoom = rooms.find((r) => r.id === selectedRoomId);

  return (
    <div className="flex flex-col w-full h-full bg-gray-950 rounded-2xl border border-gray-800/60 overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800/60 bg-gray-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <button
            onClick={onCancel}
            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-gray-800/60 hover:bg-gray-700/60 text-gray-400 hover:text-white text-[10px] font-bold uppercase rounded-lg transition-all"
            title="Back to 3D View"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            <span className="hidden sm:inline">Back</span>
          </button>
          <div className="h-4 w-px bg-gray-700" />
          <h2 className="text-sm font-bold text-gray-200">Layout Editor</h2>
          <div className="h-4 w-px bg-gray-700" />

          <div className="flex items-center gap-1">
            <button
              onClick={() => dispatch({ type: "UNDO" })}
              disabled={!canUndo}
              className="p-1.5 rounded-lg bg-gray-800/60 hover:bg-gray-700/60 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              title="Undo (Ctrl+Z)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-gray-300">
                <path d="M3 7v6h6M3 13a9 9 0 1 1 2.6-6.3L3 9" />
              </svg>
            </button>
            <button
              onClick={() => dispatch({ type: "REDO" })}
              disabled={!canRedo}
              className="p-1.5 rounded-lg bg-gray-800/60 hover:bg-gray-700/60 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              title="Redo (Ctrl+Shift+Z)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-gray-300">
                <path d="M21 7v6h-6M21 13a9 9 0 1 0-2.6-6.3L21 9" />
              </svg>
            </button>
          </div>

          <div className="h-4 w-px bg-gray-700" />

          <button
            onClick={() => setGridSnap((p) => !p)}
            className={`px-2 py-1 text-[10px] font-bold rounded-lg transition-all border ${
              gridSnap
                ? "bg-blue-600/20 text-blue-400 border-blue-600/40"
                : "bg-gray-800/60 text-gray-500 border-gray-700 hover:text-gray-400"
            }`}
            title="Toggle Grid Snap (G)"
          >
            Grid
          </button>
          <button
            onClick={() => setShowDimensions((p) => !p)}
            className={`px-2 py-1 text-[10px] font-bold rounded-lg transition-all border ${
              showDimensions
                ? "bg-emerald-600/20 text-emerald-400 border-emerald-600/40"
                : "bg-gray-800/60 text-gray-500 border-gray-700 hover:text-gray-400"
            }`}
            title="Toggle Dimensions (D)"
          >
            Dims
          </button>
          <button
            onClick={() => setShowArea((p) => !p)}
            className={`px-2 py-1 text-[10px] font-bold rounded-lg transition-all border ${
              showArea
                ? "bg-purple-600/20 text-purple-400 border-purple-600/40"
                : "bg-gray-800/60 text-gray-500 border-gray-700 hover:text-gray-400"
            }`}
            title="Toggle Area Labels"
          >
            Area
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-600 font-mono">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={resetView}
            className="px-2 py-1 text-[10px] font-bold rounded-lg bg-gray-800/60 text-gray-500 border border-gray-700 hover:text-gray-400 transition-all"
            title="Reset View"
          >
            Reset
          </button>
          <div className="h-4 w-px bg-gray-700" />
          <button
            onClick={handleAddRoom}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            + Room
          </button>
          <button
            onClick={onCancel}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            Cancel
          </button>
          {onSaveAndGoBack && (
            <button
              onClick={onSaveAndGoBack}
              disabled={saving}
              className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save & Return"}
            </button>
          )}
          <button
            onClick={handleConfirm}
            className="px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            Confirm
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Canvas */}
        <div
          ref={containerRef}
          className="flex-1 relative overflow-hidden bg-black"
          style={{ cursor: isPanning ? "grabbing" : "default" }}
          onMouseDown={handleMiddlePointerDown}
        >
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Blueprint"
              onLoad={handleImageLoad}
              className="absolute inset-0 w-full h-full object-contain opacity-25 pointer-events-none"
              draggable={false}
            />
          )}

          {gridSnap && imgDims && (
            <svg
              viewBox={viewBox}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ opacity: 0.15 }}
            >
              <defs>
                <pattern id="grid" width={GRID_SNAP_SIZE} height={GRID_SNAP_SIZE} patternUnits="userSpaceOnUse">
                  <path
                    d={`M ${GRID_SNAP_SIZE} 0 L 0 0 0 ${GRID_SNAP_SIZE}`}
                    fill="none"
                    stroke="#64748b"
                    strokeWidth="0.5"
                  />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          )}

          {imgDims && (
            <svg
              ref={svgRef}
              viewBox={viewBox}
              className="absolute inset-0 w-full h-full"
              style={{
                touchAction: "none",
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
              }}
            >
              <defs>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="shadow">
                  <feDropShadow dx="0" dy="1" stdDeviation="2" floodColor="#000" floodOpacity="0.5" />
                </filter>
              </defs>

              {rooms.map((room, roomIdx) => {
                const color = roomColor(roomIdx);
                const pts = room.vertices;
                if (pts.length < 3) return null;
                const pathD =
                  pts
                    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`)
                    .join(" ") + " Z";

                const isSelected = selectedRoomId === room.id;
                const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
                const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
                const pixelArea = roomPixelArea(pts);
                const areaM2 = (pixelArea / (pxToMeter * pxToMeter)).toFixed(1);

                const bbW_px = Math.max(...pts.map((p) => p.x)) - Math.min(...pts.map((p) => p.x));
                const bbH_px = Math.max(...pts.map((p) => p.y)) - Math.min(...pts.map((p) => p.y));
                const bbW_m = (bbW_px / pxToMeter).toFixed(1);
                const bbH_m = (bbH_px / pxToMeter).toFixed(1);

                return (
                  <g key={room.id}>
                    {/* Room fill */}
                    <path
                      d={pathD}
                      fill={color}
                      fillOpacity={isSelected ? 0.2 : 0.08}
                      stroke="none"
                      style={{ pointerEvents: "none" }}
                    />

                    {/* Edges */}
                    {pts.map((v, i) => {
                      const a = pts[i];
                      const b = pts[(i + 1) % pts.length];
                      const mx = (a.x + b.x) / 2;
                      const my = (a.y + b.y) / 2;
                      const isHovered =
                        hoveredEdge?.roomId === room.id &&
                        hoveredEdge?.edgeIdx === i;
                      const edgeLen = edgeLength(a, b, pxToMeter);

                      return (
                        <g key={`edge-${room.id}-${i}`}>
                          {/* Hit area */}
                          <line
                            x1={a.x}
                            y1={a.y}
                            x2={b.x}
                            y2={b.y}
                            stroke="transparent"
                            strokeWidth={isSelected ? 16 : 0}
                            style={{ cursor: isSelected ? "crosshair" : "default" }}
                            onDoubleClick={isSelected ? handleEdgeDoubleClick(room.id, i) : undefined}
                            onContextMenu={isSelected ? handleEdgeDelete(room.id, i) : undefined}
                            onMouseEnter={() => {
                              if (isSelected) {
                                setHoveredEdge({ roomId: room.id, edgeIdx: i });
                                setTooltip({
                                  x: mx,
                                  y: my - 14,
                                  text: `${edgeLen.toFixed(2)}m`,
                                });
                              }
                            }}
                            onMouseLeave={() => {
                              setHoveredEdge((prev) =>
                                prev?.roomId === room.id && prev?.edgeIdx === i
                                  ? null
                                  : prev
                              );
                              setTooltip(null);
                            }}
                          />
                          {/* Visible edge */}
                          <line
                            x1={a.x}
                            y1={a.y}
                            x2={b.x}
                            y2={b.y}
                            stroke={color}
                            strokeWidth={isHovered ? 5 : isSelected ? 3.5 : 2.5}
                            strokeOpacity={isHovered ? 1 : isSelected ? 0.85 : 0.45}
                            strokeDasharray={isSelected ? "none" : "6 4"}
                            style={{ pointerEvents: "none" }}
                          />
                          {/* Edge midpoint add button */}
                          {isSelected && (
                            <g
                              style={{ cursor: "pointer" }}
                              onClick={handleEdgeAddVertex(room.id, i)}
                              onContextMenu={handleEdgeDelete(room.id, i)}
                            >
                              <circle
                                cx={mx}
                                cy={my}
                                r={isHovered ? 12 : 9}
                                fill="#0f172a"
                                stroke={isHovered ? "#22c55e" : color}
                                strokeWidth={2}
                                fillOpacity={isHovered ? 1 : 0.85}
                              />
                              <text
                                x={mx}
                                y={my}
                                textAnchor="middle"
                                dominantBaseline="central"
                                fill={isHovered ? "#22c55e" : "white"}
                                fontSize={isHovered ? 16 : 13}
                                fontWeight="bold"
                                style={{ pointerEvents: "none" }}
                              >
                                +
                              </text>
                              {isHovered && (
                                <text
                                  x={mx}
                                  y={my + 22}
                                  textAnchor="middle"
                                  dominantBaseline="central"
                                  fill="#94a3b8"
                                  fontSize={9}
                                  fontFamily="monospace"
                                  style={{ pointerEvents: "none" }}
                                >
                                  add point
                                </text>
                              )}
                            </g>
                          )}
                        </g>
                      );
                    })}

                    {/* Vertices */}
                    {pts.map((v, vIdx) => {
                      const isSel =
                        selectedVertex?.roomId === room.id &&
                        selectedVertex?.vertexIdx === vIdx;
                      const isHov =
                        hoveredVertex?.roomId === room.id &&
                        hoveredVertex?.vertexIdx === vIdx;
                      return (
                        <g key={`vertex-${room.id}-${vIdx}`}>
                          {isSel && (
                            <circle
                              cx={v.x}
                              cy={v.y}
                              r={16}
                              fill="none"
                              stroke={color}
                              strokeWidth={2}
                              strokeOpacity={0.3}
                              strokeDasharray="4 3"
                              style={{ pointerEvents: "none" }}
                            />
                          )}
                          <circle
                            cx={v.x}
                            cy={v.y}
                            r={isSelected ? (isSel ? 9 : isHov ? 8 : 6) : 0}
                            fill={isSel ? color : isHov ? color : color}
                            fillOpacity={isSel ? 1 : isHov ? 0.9 : 0.8}
                            stroke={isSel ? "#fbbf24" : isHov ? "#fbbf24" : "white"}
                            strokeWidth={isSel ? 3 : isHov ? 2.5 : 2}
                            style={{
                              cursor: isSelected ? "grab" : "default",
                              touchAction: "none",
                              filter: isSel ? "url(#glow)" : "none",
                              pointerEvents: isSelected ? "auto" : "none",
                            }}
                            onPointerDown={handlePointerDown(room.id, vIdx)}
                            onDoubleClick={isSelected ? handleVertexDelete(room.id, vIdx) : undefined}
                            onContextMenu={isSelected ? handleVertexContextMenu(room.id, vIdx) : undefined}
                            onMouseEnter={() => {
                              if (isSelected) {
                                setHoveredVertex({ roomId: room.id, vertexIdx: vIdx });
                                const meterX = ((v.x - imgDims.w / 2) / pxToMeter).toFixed(2);
                                const meterY = ((v.y - imgDims.h / 2) / pxToMeter).toFixed(2);
                                setTooltip({
                                  x: v.x + 14,
                                  y: v.y - 14,
                                  text: `(${meterX}, ${meterY})`,
                                });
                              }
                            }}
                            onMouseLeave={() => {
                              setHoveredVertex(null);
                              setTooltip(null);
                            }}
                          />
                        </g>
                      );
                    })}

                    {/* Room label */}
                    <text
                      x={cx}
                      y={cy - (showDimensions || showArea ? 8 : 0)}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="white"
                      fontSize={isSelected ? 13 : 11}
                      fontWeight="bold"
                      filter={isSelected ? "url(#shadow)" : "none"}
                      style={{ pointerEvents: "none" }}
                    >
                      {room.label}
                    </text>

                    {/* Dimensions */}
                    {showDimensions && (
                      <text
                        x={cx}
                        y={cy + 6}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="#94a3b8"
                        fontSize={9}
                        fontFamily="monospace"
                        style={{ pointerEvents: "none" }}
                      >
                        {bbW_m}m x {bbH_m}m
                      </text>
                    )}

                    {/* Area */}
                    {showArea && (
                      <text
                        x={cx}
                        y={cy + (showDimensions ? 18 : 8)}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="#64748b"
                        fontSize={8}
                        fontFamily="monospace"
                        style={{ pointerEvents: "none" }}
                      >
                        {areaM2}m²
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Tooltip */}
              {tooltip && (
                <g style={{ pointerEvents: "none" }}>
                  <rect
                    x={tooltip.x - 30}
                    y={tooltip.y - 12}
                    width={60}
                    height={18}
                    rx={4}
                    fill="#1e293b"
                    stroke="#334155"
                    strokeWidth={1}
                  />
                  <text
                    x={tooltip.x}
                    y={tooltip.y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#e2e8f0"
                    fontSize={9}
                    fontFamily="monospace"
                  >
                    {tooltip.text}
                  </text>
                </g>
              )}
            </svg>
          )}

          {!imageUrl && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-xs">
              No blueprint image loaded
            </div>
          )}

          {/* Zoom controls overlay */}
          {imgDims && (
            <div className="absolute bottom-3 left-3 flex items-center gap-1 z-20">
              <button
                onClick={() => setZoom((p) => Math.min(5, p * 1.2))}
                className="w-7 h-7 flex items-center justify-center rounded-lg bg-gray-900/90 border border-gray-800 text-gray-400 hover:text-white transition-all text-sm font-bold"
              >
                +
              </button>
              <button
                onClick={() => setZoom((p) => Math.max(0.3, p / 1.2))}
                className="w-7 h-7 flex items-center justify-center rounded-lg bg-gray-900/90 border border-gray-800 text-gray-400 hover:text-white transition-all text-sm font-bold"
              >
                -
              </button>
              <button
                onClick={resetView}
                className="h-7 px-2 flex items-center justify-center rounded-lg bg-gray-900/90 border border-gray-800 text-gray-400 hover:text-white transition-all text-[10px] font-bold"
              >
                Fit
              </button>
            </div>
          )}

          {/* Keyboard hints overlay */}
          {imgDims && (
            <div className="absolute top-3 right-3 z-20 bg-gray-900/90 border border-gray-800/60 rounded-lg px-2.5 py-1.5 text-[9px] text-gray-500 space-y-0.5 font-mono">
              <div><span className="text-gray-400">Scroll</span> Zoom</div>
              <div><span className="text-gray-400">Middle-drag</span> Pan</div>
              <div><span className="text-gray-400">Ctrl+Z</span> Undo</div>
              <div><span className="text-gray-400">G</span> Grid snap</div>
              <div><span className="text-gray-400">D</span> Dimensions</div>
              <div><span className="text-gray-400">Del</span> Remove vertex</div>
            </div>
          )}

          {/* Room selection hint */}
          {!selectedRoomId && imgDims && rooms.length > 0 && (
            <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-20 bg-amber-600/90 backdrop-blur-sm border border-amber-500/50 rounded-xl px-4 py-2 text-xs text-white font-semibold shadow-xl animate-pulse">
              Select a room to start editing
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-72 border-l border-gray-800/60 bg-gray-900/40 overflow-y-auto flex flex-col">
          <div className="px-3 py-2.5 border-b border-gray-800/40">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                Rooms ({rooms.length})
              </span>
              {selectedRoom && (
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: roomColor(rooms.indexOf(selectedRoom)) }}
                />
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {rooms.map((room, roomIdx) => {
              const color = roomColor(roomIdx);
              const isSelected = selectedRoomId === room.id;
              const pixelArea = roomPixelArea(room.vertices);
              const areaM2 = (pixelArea / (pxToMeter * pxToMeter)).toFixed(1);
              const bbW_px = Math.max(...room.vertices.map((p) => p.x)) - Math.min(...room.vertices.map((p) => p.x));
              const bbH_px = Math.max(...room.vertices.map((p) => p.y)) - Math.min(...room.vertices.map((p) => p.y));

              return (
                <div
                  key={room.id}
                  className={`p-2.5 rounded-xl border transition-all cursor-pointer group ${
                    isSelected
                      ? "border-gray-600/80 bg-gray-800/70 ring-1 ring-gray-700/40"
                      : "border-gray-800/40 bg-gray-900/30 hover:bg-gray-800/30 hover:border-gray-700/40"
                  }`}
                  onClick={() => setSelectedRoomId(room.id)}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span
                      className="w-3 h-3 rounded-md shrink-0 border border-white/10"
                      style={{ backgroundColor: color }}
                    />
                    {editingLabel === room.id ? (
                      <input
                        autoFocus
                        value={room.label}
                        onChange={(e) => handleLabelChange(room.id, e.target.value)}
                        onBlur={handleLabelBlur}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleLabelBlur();
                          if (e.key === "Escape") {
                            setEditingLabel(null);
                          }
                        }}
                        className="flex-1 bg-gray-800 border border-gray-600 rounded-md px-2 py-0.5 text-xs text-white outline-none focus:border-blue-500 transition-colors"
                      />
                    ) : (
                      <span
                        className="flex-1 text-xs text-gray-200 font-semibold truncate"
                        onDoubleClick={() => setEditingLabel(room.id)}
                      >
                        {room.label}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-[10px] text-gray-500 font-mono">
                      {(bbW_px / pxToMeter).toFixed(1)}m x {(bbH_px / pxToMeter).toFixed(1)}m
                    </span>
                    <span className="text-gray-700">·</span>
                    <span className="text-[10px] text-gray-500 font-mono">
                      {areaM2}m²
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleOpenSpace(room.id);
                      }}
                      className={`text-[9px] px-2 py-0.5 rounded-md font-bold uppercase transition-all ${
                        room.isOpenSpace
                          ? "bg-green-600/20 text-green-400 border border-green-600/30"
                          : "bg-gray-800/60 text-gray-500 border border-gray-700/60 hover:text-gray-400"
                      }`}
                    >
                      {room.isOpenSpace ? "Open" : "Closed"}
                    </button>

                    <span className="text-[9px] text-gray-600 font-mono ml-auto">
                      {room.vertices.length}pts
                    </span>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteRoom(room.id);
                      }}
                      className="text-[9px] px-1.5 py-0.5 rounded-md bg-red-900/20 text-red-400/70 border border-red-800/20 font-bold uppercase opacity-0 group-hover:opacity-100 hover:bg-red-900/40 hover:text-red-400 transition-all"
                    >
                      Del
                    </button>
                  </div>
                </div>
              );
            })}

            {rooms.length === 0 && (
              <div className="text-[10px] text-gray-600 text-center py-8 space-y-2">
                <p>No rooms yet.</p>
                <button
                  onClick={handleAddRoom}
                  className="px-3 py-1.5 bg-blue-600/20 text-blue-400 border border-blue-600/30 rounded-lg text-[10px] font-bold uppercase hover:bg-blue-600/30 transition-all"
                >
                  + Add First Room
                </button>
              </div>
            )}
          </div>

          {/* Sidebar summary */}
          {rooms.length > 0 && (
            <div className="px-3 py-2.5 border-t border-gray-800/40 bg-gray-900/60">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-gray-500">Total Area</span>
                <span className="text-gray-300 font-mono font-bold">
                  {rooms
                    .reduce(
                      (sum, r) =>
                        sum + roomPixelArea(r.vertices) / (pxToMeter * pxToMeter),
                      0
                    )
                    .toFixed(1)}m²
                </span>
              </div>
              <div className="flex items-center justify-between text-[10px] mt-0.5">
                <span className="text-gray-500">Total sq.ft</span>
                <span className="text-gray-300 font-mono font-bold">
                  {rooms
                    .reduce(
                      (sum, r) =>
                        sum +
                        (roomPixelArea(r.vertices) / (pxToMeter * pxToMeter)) *
                          10.764,
                      0
                    )
                    .toFixed(0)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
