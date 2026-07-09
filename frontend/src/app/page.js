"use client";

import React, { useState, useEffect } from 'react';
import BlueprintUploader from '@/components/BlueprintUploader';
import RoomExtrusionCanvas from '@/components/RoomExtrusionCanvas';
import { GridScan } from '@/components/GridScan/GridScan';

export default function HomePage() {
  const [layoutData, setLayoutData] = useState(null);
  const [activeFloor, setActiveFloor] = useState(0);
  const [uploadedImageUrl, setUploadedImageUrl] = useState("");

  useEffect(() => {
    return () => {
      if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    };
  }, [uploadedImageUrl]);

  const handleGenerationSuccess = (data, localFile) => {
    setLayoutData(data);
    setActiveFloor(0);

    setUploadedImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return "";
    });

    if (localFile) {
      const objectUrl = URL.createObjectURL(localFile);
      setUploadedImageUrl(objectUrl);
    } else {
      setUploadedImageUrl("");
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <GridScan
          sensitivity={0.55}
          lineThickness={1}
          linesColor="#1e293b"
          gridScale={0.1}
          scanColor="#3b82f6"
          scanOpacity={0.3}
          enablePost
          bloomIntensity={0.3}
          chromaticAberration={0.002}
          noiseIntensity={0.01}
          scanDuration={3}
          scanDelay={2}
          scanDirection="pingpong"
        />
      </div>
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-6 py-12 space-y-8">
        <header className="text-center max-w-2xl mx-auto space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-xs font-semibold text-blue-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            Universal 3D Spatial Engine
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-b from-white to-gray-400 bg-clip-text text-transparent">
            Blueprint Spatial Modeler
          </h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-5 bg-gray-950/60 border border-gray-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-2xl space-y-6">
            <div>
              <h2 className="text-base font-bold text-gray-200">Layout Specifications</h2>
              <p className="text-xs text-gray-500">Upload an image schematic or build spaces step-by-step.</p>
            </div>
            <hr className="border-gray-900" />
            <BlueprintUploader onUploadSuccess={handleGenerationSuccess} />
          </div>

          <div className="lg:col-span-7 flex flex-col h-[580px] border border-gray-800/60 rounded-2xl overflow-hidden relative shadow-2xl">
            {layoutData ? (
              <>
                <div className="absolute top-4 left-4 z-20 bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 flex items-center gap-4 text-xs">
                  <div>
                    <span className="text-gray-500 mr-1">Parsed Rooms:</span>
                    <span className="font-bold text-blue-400">{layoutData.totalRooms}</span>
                  </div>
                  <div className="w-px h-3 bg-gray-800" />
                  <div>
                    <span className="text-gray-500 mr-1">Total Footprint:</span>
                    <span className="font-bold text-purple-400">{layoutData.calculatedSqFt || 1500} sq.ft.</span>
                  </div>
                </div>

                {layoutData.totalFloors > 1 && (
                  <div className="absolute top-4 right-4 z-20 flex gap-1 bg-gray-900/90 border border-gray-800 p-1 rounded-lg">
                    {Array.from({ length: layoutData.totalFloors }).map((_, idx) => (
                      <button
                        key={idx}
                        onClick={() => setActiveFloor(idx)}
                        className={`px-2.5 py-1 text-[10px] font-bold uppercase rounded-md transition-all ${
                          activeFloor === idx ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                        }`}
                      >
                        Floor {idx + 1}
                      </button>
                    ))}
                  </div>
                )}

                <div className="w-full h-full relative z-10">
                  <RoomExtrusionCanvas 
                    layoutData={layoutData} 
                    activeFloor={activeFloor} 
                    imageUrl={uploadedImageUrl} 
                  />
                </div>
                
                <div className="absolute bottom-4 left-4 right-4 max-h-24 overflow-y-auto bg-black/80 rounded-xl p-3 border border-gray-900 text-left font-mono text-[10px] text-gray-400 z-20 custom-scrollbar">
                  <div className="text-blue-400 font-bold mb-1">{"// Active Environment Projection Streams:"}</div>
                  {layoutData.rooms?.map((rm, i) => {
                    const computedArea = typeof rm.area === 'number' 
                      ? rm.area.toFixed(1) 
                      : parseFloat(rm.area || 0).toFixed(1);

                    return (
                      <div key={i} className="truncate">
                        {`-> Mounted Layer: "${rm.label || `Room ${i + 1}`}" | Dimensions: ${rm.dimensions || 'Dynamic'} | Area: ${computedArea}m²`}
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="relative z-10 w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-gray-900/60 rounded-2xl bg-gray-950/40 text-center p-8">
                <div className="w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center border border-gray-800 text-lg mb-4">🧱</div>
                <h3 className="text-sm font-semibold text-gray-400">Interactive 3D Viewport Matrix</h3>
                <p className="text-xs text-gray-600 max-w-xs mt-1">
                  Upload an image schematic layout or execute a procedural configuration wizard stack to look at your model.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}