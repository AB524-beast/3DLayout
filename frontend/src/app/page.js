"use client";

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import BlueprintUploader from '@/components/BlueprintUploader';
import RoomExtrusionCanvas from '@/components/RoomExtrusionCanvas';
import RoomCorrectionEditor from '@/components/RoomCorrectionEditor';
import { GridScan } from '@/components/GridScan/GridScan';
import { useAuth } from '@/context/AuthContext';

export default function HomePage() {
  const { user, saveLayout } = useAuth();
  const [layoutData, setLayoutData] = useState(null);
  const [activeFloor, setActiveFloor] = useState(0);
  const [uploadedImageUrl, setUploadedImageUrl] = useState("");
  const [localFile, setLocalFile] = useState(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState(null);
  const [threeRenderer, setThreeRenderer] = useState(null);
  const [showEditor, setShowEditor] = useState(false);

  useEffect(() => {
    return () => {
      if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    };
  }, [uploadedImageUrl]);

  const handleGenerationSuccess = (data, file) => {
    setLayoutData(data);
    setSavedMsg(null);
    setActiveFloor(0);
    setLocalFile(file || null);

    setUploadedImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return "";
    });

    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setUploadedImageUrl(objectUrl);
    } else {
      setUploadedImageUrl("");
    }
  };

  const handleSave = async () => {
    if (!layoutData || !user) return;
    setSaving(true);
    setSavedMsg(null);
    try {
      await saveLayout(
        `layout_${Date.now()}.json`,
        localFile,
        JSON.stringify(layoutData)
      );
      setSavedMsg("Saved to dashboard");
    } catch (err) {
      setSavedMsg("Save failed: " + err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDownloadScreenshot = useCallback(() => {
    if (!threeRenderer) return;
    const canvas = threeRenderer.domElement;
    const dataUrl = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `3d-layout-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [threeRenderer]);

  const handleDownloadJSON = useCallback(() => {
    if (!layoutData) return;
    const blob = new Blob([JSON.stringify(layoutData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `layout-data-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [layoutData]);

  const onRendererReady = useCallback((renderer) => {
    setThreeRenderer(renderer);
  }, []);

  const handleCorrectionConfirm = useCallback((correctedData) => {
    setLayoutData(correctedData);
    setShowEditor(false);
  }, []);

  const handleCorrectionCancel = useCallback(() => {
    setShowEditor(false);
  }, []);

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
              <p className="text-xs text-gray-500">Upload an image schematic to extract and visualize your layout in 3D.</p>
            </div>
            <hr className="border-gray-900" />
            <BlueprintUploader onUploadSuccess={handleGenerationSuccess} />

            {layoutData && user && (
              <div className="space-y-2">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="w-full bg-gray-900 hover:bg-gray-800 border border-gray-800 text-gray-300 font-semibold py-2 rounded-xl text-xs uppercase tracking-wider transition-all disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save Layout to Dashboard"}
                </button>
                {savedMsg && (
                  <p className="text-xs text-green-400 text-center">{savedMsg}</p>
                )}
              </div>
            )}
          </div>

          <div className="lg:col-span-7 flex flex-col h-[640px] border border-gray-800/60 rounded-2xl overflow-hidden relative shadow-2xl">
            {layoutData ? (
              showEditor ? (
                <RoomCorrectionEditor
                  layoutData={layoutData}
                  imageUrl={uploadedImageUrl}
                  onConfirm={handleCorrectionConfirm}
                  onCancel={handleCorrectionCancel}
                />
              ) : (
              <>
                <div className="absolute top-4 left-4 z-20 bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 flex items-center gap-4 text-xs backdrop-blur-sm">
                  <div>
                    <span className="text-gray-500 mr-1">Rooms:</span>
                    <span className="font-bold text-blue-400">{layoutData.totalRooms}</span>
                  </div>
                  <div className="w-px h-3 bg-gray-800" />
                  <div>
                    <span className="text-gray-500 mr-1">Footprint:</span>
                    <span className="font-bold text-purple-400">{layoutData.calculatedSqFt || 0} sq.ft.</span>
                  </div>
                  {layoutData.segmentationMethod && (
                    <>
                      <div className="w-px h-3 bg-gray-800" />
                      <div>
                        <span className="text-gray-500 mr-1">Engine:</span>
                        <span className={`font-bold ${layoutData.segmentationMethod === 'ml' ? 'text-green-400' : 'text-amber-400'}`}>
                          {layoutData.segmentationMethod === 'ml' ? 'ML' : 'OpenCV'}
                        </span>
                      </div>
                    </>
                  )}
                </div>

                {layoutData.totalFloors > 1 && (
                  <div className="absolute top-4 right-4 z-20 flex gap-1 bg-gray-900/90 border border-gray-800 p-1 rounded-lg backdrop-blur-sm">
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
                    onRendererReady={onRendererReady}
                  />
                </div>
                
                <div className="absolute bottom-4 left-4 right-4 flex items-end gap-2 z-20">
                  <div className="flex-1 max-h-24 overflow-y-auto bg-black/80 backdrop-blur-sm rounded-xl p-3 border border-gray-900 text-left font-mono text-[10px] text-gray-400 custom-scrollbar">
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
                  <div className="flex flex-col gap-1.5 shrink-0">
                    <button
                      onClick={() => setShowEditor(true)}
                      className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all whitespace-nowrap shadow-lg shadow-amber-600/20"
                      title="Open room correction editor"
                    >
                      Correct Layout
                    </button>
                    <button
                      onClick={handleDownloadScreenshot}
                      disabled={!threeRenderer}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all disabled:opacity-40 whitespace-nowrap"
                      title="Download 3D viewport as PNG"
                    >
                      Screenshot
                    </button>
                    <button
                      onClick={handleDownloadJSON}
                      disabled={!layoutData}
                      className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all disabled:opacity-40 whitespace-nowrap"
                      title="Download layout data as JSON"
                    >
                      JSON
                    </button>
                    <Link
                      href="/dashboard"
                      className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-[10px] font-bold uppercase rounded-lg transition-all whitespace-nowrap text-center"
                    >
                      Dashboard
                    </Link>
                  </div>
                </div>
              </>
              )
            ) : (
              <div className="relative z-10 w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-gray-900/60 rounded-2xl bg-gray-950/40 text-center p-8">
                <div className="w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center border border-gray-800 text-lg mb-4">🧱</div>
                <h3 className="text-sm font-semibold text-gray-400">Interactive 3D Viewport Matrix</h3>
                <p className="text-xs text-gray-600 max-w-xs mt-1">
                  Upload an image schematic layout to generate your 3D model.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}