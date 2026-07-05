"use client";
import React, { useState } from 'react';
import BlueprintUploader from '@/components/BlueprintUploader';

export default function HomePage() {
  const [layoutData, setLayoutData] = useState(null);
  const [activeFloor, setActiveFloor] = useState(0);

  const handleGenerationSuccess = (data) => {
    console.log("Successfully loaded 3D Environment Model Data:", data);
    setLayoutData(data);
    setActiveFloor(0); // Reset to base floor on load
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-blue-500/30">
      {/* Dynamic Background Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-6 py-12">
        {/* Header Block Section */}
        <header className="text-center max-w-2xl mx-auto mb-10 space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-xs font-semibold text-blue-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            Universal 3D Spatial Engine
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-b from-white to-gray-400 bg-clip-text text-transparent">
            Blueprint Spatial Modeler
          </h1>
          <p className="text-sm text-gray-400 leading-relaxed">
            Drop an image schematic layout or specify dimensions procedurally to extrude your flat structural maps into immediate orthogonal 3D models.
          </p>
        </header>

        {/* Workspace Layout Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT PANEL: Immediate Generation Controller */}
          <div className="lg:col-span-5 bg-gray-950/60 border border-gray-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-2xl space-y-6">
            <div>
              <h2 className="text-base font-bold text-gray-200">Layout Specifications</h2>
              <p className="text-xs text-gray-500">Configure parameters to instantly assemble environmental geometries.</p>
            </div>
            
            <hr className="border-gray-900" />
            
            {/* The Integrated Parameter Engine Uploader component injected directly on page layout */}
            <BlueprintUploader onUploadSuccess={handleGenerationSuccess} />
          </div>

          {/* RIGHT PANEL: Interactive WebGL Canvas Projections Viewer Bounding Box */}
          <div className="lg:col-span-7 flex flex-col h-[580px] bg-gray-950/40 border border-gray-800/60 rounded-2xl overflow-hidden backdrop-blur-md relative shadow-2xl">
            {layoutData ? (
              <>
                {/* Active Interactive Top Bar Overlays */}
                <div className="absolute top-4 left-4 z-20 bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 flex items-center gap-4 text-xs">
                  <div>
                    <span className="text-gray-500 mr-1">Parsed Rooms:</span>
                    <span className="font-bold text-blue-400">{layoutData.totalRooms}</span>
                  </div>
                  <div className="w-px h-3 bg-gray-800" />
                  <div>
                    <span className="text-gray-500 mr-1">Layers:</span>
                    <span className="font-bold text-purple-400">{layoutData.totalFloors || 1} Story</span>
                  </div>
                </div>

                {/* Multi-Floor Toggle Controller Matrix */}
                {layoutData.totalFloors > 1 && (
                  <div className="absolute top-4 right-4 z-20 flex gap-1 bg-gray-900/90 border border-gray-800 p-1 rounded-lg">
                    {Array.from({ length: layoutData.totalFloors }).map((_, idx) => (
                      <button
                        key={idx}
                        onClick={() => setActiveFloor(idx)}
                        className={`px-2.5 py-1 text-[10px] font-bold uppercase rounded-md transition-colors ${
                          activeFloor === idx 
                            ? 'bg-blue-600 text-white' 
                            : 'text-gray-400 hover:text-white hover:bg-gray-800'
                        }`}
                      >
                        Lvl {idx + 1}
                      </button>
                    ))}
                  </div>
                )}

                {/* Interactive View Matrix Bounding Window Placeholder */}
                <div className="w-full h-full flex flex-col items-center justify-center bg-gray-950 text-center relative p-6">
                  {/* Your Three.js or WebGL Canvas element initializes directly right here */}
                  <div className="space-y-2 pointer-events-none z-10 max-w-sm">
                    <svg className="w-12 h-12 text-blue-500/40 mx-auto animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                    </svg>
                    <p className="text-xs font-semibold text-gray-300">Initial Flat Ortho View Mounted</p>
                    <p className="text-[11px] text-gray-500 leading-normal">
                      Rendering structural wall matrices centered for level {activeFloor + 1}. Rotate or click viewports to modify perspectives.
                    </p>
                  </div>
                  
                  {/* Structural Coordinate Debug Ledger */}
                  <div className="absolute bottom-4 left-4 right-4 max-h-32 overflow-y-auto bg-black/50 rounded-xl p-3 border border-gray-900 text-left font-mono text-[10px] text-gray-500 space-y-1">
                    <div>{`// Initializing Top-Down View Perspective`}</div>
                    {layoutData.rooms.map((rm, i) => (
                      <div key={i}>
                        {`-> Extruded "${rm.label}" | Elevation Z: ${rm.elevationZ}m | Area: ${rm.area.toFixed(1)}m²`}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              /* Blank State Canvas Empty Matrix UI Window View */
              <div className="w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-gray-900/60 rounded-2xl bg-gray-950/20 text-center p-8">
                <div className="w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center border border-gray-800 text-gray-600 mb-4">
                  🧱
                </div>
                <h3 className="text-sm font-semibold text-gray-400">WebGL Canvas Target Viewport</h3>
                <p className="text-xs text-gray-600 max-w-xs mt-1">
                  Configure structural variables on the control deck to launch the real-time layout matrix calculations.
                </p>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}