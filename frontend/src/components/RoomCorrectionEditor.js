"use client";

import React, { useState, useRef, useCallback, useEffect, useMemo } from "react";

const PLANE_METERS = 14.0;

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

  return {
    label,
    dimensions: `${bbW.toFixed(1)}m x ${bbH.toFixed(1)}m`,
    centerX,
    centerY,
    elevationZ: elevationZ || 0,
    isOpenSpace: !!isOpenSpace,
    walls: metersToWalls(ptsM),
    area: Math.round(bbW * bbH * 100) / 100,
  };
}

let nextRoomId = 1;

const ROOM_COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#f97316",
];

function roomColor(idx) {
  return ROOM_COLORS[idx % ROOM_COLORS.length];
}

export default function RoomCorrectionEditor({
  layoutData,
  imageUrl,
  onConfirm,
  onCancel,
}) {
  const [imgDims, setImgDims] = useState(null);
  const [selectedRoomId, setSelectedRoomId] = useState(null);
  const [dragInfo, setDragInfo] = useState(null);
  const [editingLabel, setEditingLabel] = useState(null);
  const svgRef = useRef(null);

  const pxToMeter = imgDims ? imgDims.h / PLANE_METERS : 1;

  const initialRooms = useMemo(() => {
    if (!imgDims || !layoutData?.rooms) return null;
    return layoutData.rooms.map((r) => ({
      id: nextRoomId++,
      label: r.label || "Room",
      isOpenSpace: !!r.isOpenSpace,
      elevationZ: r.elevationZ || 0,
      vertices: wallsToPixelVertices(r.walls || [], imgDims.h / PLANE_METERS, imgDims.w, imgDims.h),
    }));
  }, [imgDims, layoutData]);

  const [rooms, setRooms] = useState([]);
  const [prevInitialKey, setPrevInitialKey] = useState(null);

  if (initialRooms && initialRooms !== prevInitialKey) {
    setPrevInitialKey(initialRooms);
    setRooms(initialRooms);
  }

  const handleImageLoad = (e) => {
    setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  };

  const svgPoint = useCallback((clientX, clientY) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const transformed = pt.matrixTransform(ctm.inverse());
    return { x: transformed.x, y: transformed.y };
  }, []);

  const handlePointerDown = (roomId, vertexIdx) => (e) => {
    e.stopPropagation();
    e.preventDefault();
    setSelectedRoomId(roomId);
    setDragInfo({ roomId, vertexIdx });
  };

  const handlePointerMove = useCallback(
    (e) => {
      if (!dragInfo) return;
      const { x, y } = svgPoint(e.clientX, e.clientY);
      setRooms((prev) =>
        prev.map((r) => {
          if (r.id !== dragInfo.roomId) return r;
          const newVertices = r.vertices.map((v, idx) =>
            idx === dragInfo.vertexIdx ? { x, y } : v
          );
          return { ...r, vertices: newVertices };
        })
      );
    },
    [dragInfo, svgPoint]
  );

  const handlePointerUp = useCallback(() => setDragInfo(null), []);

  useEffect(() => {
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [handlePointerMove, handlePointerUp]);

  const handleEdgeDoubleClick = (roomId, edgeIdx) => (e) => {
    e.stopPropagation();
    const { x, y } = svgPoint(e.clientX, e.clientY);
    setRooms((prev) =>
      prev.map((r) => {
        if (r.id !== roomId) return r;
        const newVertices = [...r.vertices];
        newVertices.splice(edgeIdx + 1, 0, { x, y });
        return { ...r, vertices: newVertices };
      })
    );
  };

  const handleVertexDelete = (roomId, vertexIdx) => (e) => {
    e.stopPropagation();
    e.preventDefault();
    setRooms((prev) =>
      prev.map((r) => {
        if (r.id !== roomId || r.vertices.length <= 3) return r;
        return { ...r, vertices: r.vertices.filter((_, i) => i !== vertexIdx) };
      })
    );
  };

  const handleAddRoom = () => {
    if (!imgDims) return;
    const cx = imgDims.w / 2;
    const cy = imgDims.h / 2;
    const half = Math.min(imgDims.w, imgDims.h) * 0.1;
    const newRoom = {
      id: nextRoomId++,
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
    setRooms((prev) => [...prev, newRoom]);
    setSelectedRoomId(newRoom.id);
  };

  const handleDeleteRoom = (roomId) => {
    setRooms((prev) => prev.filter((r) => r.id !== roomId));
    if (selectedRoomId === roomId) setSelectedRoomId(null);
  };

  const handleLabelChange = (roomId, newLabel) => {
    setRooms((prev) =>
      prev.map((r) => (r.id === roomId ? { ...r, label: newLabel } : r))
    );
  };

  const handleToggleOpenSpace = (roomId) => {
    setRooms((prev) =>
      prev.map((r) =>
        r.id === roomId ? { ...r, isOpenSpace: !r.isOpenSpace } : r
      )
    );
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
    const calculatedSqFt = Math.round(
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

  return (
    <div className="flex flex-col w-full h-full bg-gray-950 rounded-2xl border border-gray-800/60 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/60 bg-gray-900/60">
        <div>
          <h2 className="text-sm font-bold text-gray-200">Room Correction Editor</h2>
          <p className="text-[10px] text-gray-500">
            Drag vertices to adjust walls. Double-click edges to add points.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleAddRoom}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            + Add Room
          </button>
          <button
            onClick={onCancel}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all"
          >
            Confirm
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <div className="flex-1 relative overflow-hidden bg-black">
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Blueprint"
              onLoad={handleImageLoad}
              className="absolute inset-0 w-full h-full object-contain opacity-30 pointer-events-none"
            />
          )}
          {imgDims && (
            <svg
              ref={svgRef}
              viewBox={viewBox}
              className="absolute inset-0 w-full h-full"
              style={{ cursor: dragInfo ? "grabbing" : "default" }}
            >
              {rooms.map((room, roomIdx) => {
                const color = roomColor(roomIdx);
                const pts = room.vertices;
                if (pts.length < 3) return null;
                const pathD =
                  pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ") + " Z";

                const edgeElements = [];
                for (let i = 0; i < pts.length; i++) {
                  const a = pts[i];
                  const b = pts[(i + 1) % pts.length];
                  const midX = (a.x + b.x) / 2;
                  const midY = (a.y + b.y) / 2;
                  edgeElements.push(
                    <line
                      key={`edge-${room.id}-${i}`}
                      x1={a.x}
                      y1={a.y}
                      x2={b.x}
                      y2={b.y}
                      stroke={color}
                      strokeWidth={3}
                      strokeOpacity={0.5}
                      style={{ cursor: "crosshair" }}
                      onDoubleClick={handleEdgeDoubleClick(room.id, i)}
                    />
                  );
                  edgeElements.push(
                    <circle
                      key={`mid-${room.id}-${i}`}
                      cx={midX}
                      cy={midY}
                      r={4}
                      fill={color}
                      fillOpacity={0.2}
                      stroke="none"
                      style={{ pointerEvents: "none" }}
                    />
                  );
                }

                return (
                  <g key={room.id}>
                    <path
                      d={pathD}
                      fill={color}
                      fillOpacity={selectedRoomId === room.id ? 0.15 : 0.08}
                      stroke={color}
                      strokeWidth={selectedRoomId === room.id ? 3 : 2}
                      strokeOpacity={0.8}
                      onClick={() => setSelectedRoomId(room.id)}
                      style={{ cursor: "pointer" }}
                    />
                    {edgeElements}
                    {pts.map((v, vIdx) => (
                      <g key={`vertex-${room.id}-${vIdx}`}>
                        <circle
                          cx={v.x}
                          cy={v.y}
                          r={7}
                          fill={color}
                          fillOpacity={0.9}
                          stroke="white"
                          strokeWidth={2}
                          style={{ cursor: "grab" }}
                          onPointerDown={handlePointerDown(room.id, vIdx)}
                        />
                        <circle
                          cx={v.x}
                          cy={v.y}
                          r={10}
                          fill="transparent"
                          style={{ cursor: "pointer" }}
                          onDoubleClick={handleVertexDelete(room.id, vIdx)}
                        />
                      </g>
                    ))}
                    {(() => {
                      const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
                      const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
                      return (
                        <text
                          x={cx}
                          y={cy}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fill="white"
                          fontSize={12}
                          fontWeight="bold"
                          style={{ pointerEvents: "none" }}
                        >
                          {room.label}
                        </text>
                      );
                    })()}
                  </g>
                );
              })}
            </svg>
          )}
          {!imageUrl && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-xs">
              No blueprint image loaded
            </div>
          )}
        </div>

        <div className="w-64 border-l border-gray-800/60 bg-gray-900/40 overflow-y-auto p-3 space-y-2">
          <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
            Rooms ({rooms.length})
          </div>
          {rooms.map((room, roomIdx) => {
            const color = roomColor(roomIdx);
            return (
              <div
                key={room.id}
                className={`p-2 rounded-lg border transition-all cursor-pointer ${
                  selectedRoomId === room.id
                    ? "border-gray-600 bg-gray-800/60"
                    : "border-gray-800/40 bg-gray-900/40 hover:bg-gray-800/30"
                }`}
                onClick={() => setSelectedRoomId(room.id)}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  {editingLabel === room.id ? (
                    <input
                      autoFocus
                      value={room.label}
                      onChange={(e) => handleLabelChange(room.id, e.target.value)}
                      onBlur={() => setEditingLabel(null)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") setEditingLabel(null);
                      }}
                      className="flex-1 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-white outline-none"
                    />
                  ) : (
                    <span
                      className="flex-1 text-xs text-gray-200 font-semibold truncate"
                      onDoubleClick={() => setEditingLabel(room.id)}
                    >
                      {room.label}
                    </span>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleToggleOpenSpace(room.id);
                    }}
                    className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase transition-all ${
                      room.isOpenSpace
                        ? "bg-green-600/20 text-green-400 border border-green-600/30"
                        : "bg-gray-800 text-gray-500 border border-gray-700"
                    }`}
                  >
                    {room.isOpenSpace ? "Open" : "Closed"}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteRoom(room.id);
                    }}
                    className="text-[9px] px-1.5 py-0.5 rounded bg-red-900/30 text-red-400 border border-red-800/30 font-bold uppercase hover:bg-red-900/50 transition-all"
                  >
                    Del
                  </button>
                </div>
                <div className="text-[9px] text-gray-500 font-mono">
                  {room.vertices.length} vertices &middot;{" "}
                  {room.isOpenSpace ? "open space" : "enclosed"}
                </div>
              </div>
            );
          })}
          {rooms.length === 0 && (
            <div className="text-[10px] text-gray-600 text-center py-4">
              No rooms. Click &quot;+ Add Room&quot; to begin.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
