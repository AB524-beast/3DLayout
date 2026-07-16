"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showInfo, setShowInfo] = useState(true);
  const hideInfoTimer = useRef(null);

  useEffect(() => {
    return () => {
      if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    };
  }, [uploadedImageUrl]);

  useEffect(() => {
    if (layoutData && !showEditor) {
      setIsFullscreen(true);
      setShowInfo(true);
      if (hideInfoTimer.current) clearTimeout(hideInfoTimer.current);
      hideInfoTimer.current = setTimeout(() => setShowInfo(false), 5000);
      return () => { if (hideInfoTimer.current) clearTimeout(hideInfoTimer.current); };
    } else {
      setIsFullscreen(false);
    }
  }, [layoutData, showEditor]);

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

  const handleBackToUpload = useCallback(() => {
    setLayoutData(null);
    setUploadedImageUrl("");
    setLocalFile(null);
    setShowEditor(false);
    setIsFullscreen(false);
  }, []);

  if (isFullscreen && layoutData && !showEditor) {
    return (
      <div className="fixed inset-0 z-40 bg-black">
        <div className="absolute inset-0">
          <RoomExtrusionCanvas
            layoutData={layoutData}
            activeFloor={activeFloor}
            imageUrl={uploadedImageUrl}
            onRendererReady={onRendererReady}
          />
        </div>

        {showInfo && (
          <div className="absolute top-3 left-3 z-50 animate-fade-in">
            <div className="bg-gray-900/85 backdrop-blur-md border border-gray-700/50 rounded-xl px-3 py-2 flex items-center gap-3 text-xs shadow-xl">
              <div>
                <span className="text-gray-500 mr-1">Rooms:</span>
                <span className="font-bold text-blue-400">{layoutData.totalRooms}</span>
              </div>
              <div className="w-px h-3 bg-gray-700" />
              <div>
                <span className="text-gray-500 mr-1">Footprint:</span>
                <span className="font-bold text-purple-400">{layoutData.calculatedSqFt || 0} sq.ft.</span>
              </div>
              {layoutData.segmentationMethod && (
                <>
                  <div className="w-px h-3 bg-gray-700" />
                  <div>
                    <span className="text-gray-500 mr-1">Engine:</span>
                    <span className={`font-bold ${layoutData.segmentationMethod === 'ml' ? 'text-green-400' : 'text-amber-400'}`}>
                      {layoutData.segmentationMethod === 'ml' ? 'ML' : layoutData.segmentationMethod === 'procedural' ? 'Procedural' : 'OpenCV'}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {layoutData.totalFloors > 1 && (
          <div className="absolute top-3 right-3 z-50 flex gap-1 bg-gray-900/85 backdrop-blur-md border border-gray-700/50 p-1 rounded-lg shadow-xl">
            {Array.from({ length: layoutData.totalFloors }).map((_, idx) => (
              <button
                key={idx}
                onClick={() => setActiveFloor(idx)}
                className={`px-2.5 py-1 text-[10px] font-bold uppercase rounded-md transition-all ${
                  activeFloor === idx ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                F{idx + 1}
              </button>
            ))}
          </div>
        )}

        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-50 animate-slide-up">
          <div className="bg-gray-900/85 backdrop-blur-md border border-gray-700/50 rounded-2xl px-3 py-2 flex items-center gap-1.5 shadow-xl">
            <button
              onClick={handleBackToUpload}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-[10px] font-bold uppercase rounded-xl transition-all"
              title="Back to upload"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              <span className="hidden sm:inline">Back</span>
            </button>

            <div className="w-px h-5 bg-gray-700" />

            <button
              onClick={() => setShowEditor(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-[10px] font-bold uppercase rounded-xl transition-all"
              title="Correct layout"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
              <span className="hidden sm:inline">Edit</span>
            </button>

            <button
              onClick={handleDownloadScreenshot}
              disabled={!threeRenderer}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase rounded-xl transition-all disabled:opacity-40"
              title="Download 3D as PNG"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
              </svg>
              <span className="hidden sm:inline">Capture</span>
            </button>

            <button
              onClick={handleDownloadJSON}
              disabled={!layoutData}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-[10px] font-bold uppercase rounded-xl transition-all disabled:opacity-40"
              title="Download JSON"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span className="hidden sm:inline">JSON</span>
            </button>

            {user && (
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-[10px] font-bold uppercase rounded-xl transition-all disabled:opacity-50"
                title="Save to dashboard"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                  <polyline points="17 21 17 13 7 13 7 21" />
                  <polyline points="7 3 7 8 15 8" />
                </svg>
                <span className="hidden sm:inline">{saving ? "Saving..." : "Save"}</span>
              </button>
            )}

            <div className="w-px h-5 bg-gray-700" />

            <Link
              href="/dashboard"
              className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-[10px] font-bold uppercase rounded-xl transition-all"
              title="Dashboard"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
              </svg>
              <span className="hidden sm:inline">Dashboard</span>
            </Link>
          </div>
        </div>

        {showInfo && (
          <div className="absolute bottom-20 left-3 z-50 max-w-[260px] sm:max-w-xs animate-fade-in">
            <div className="bg-gray-900/80 backdrop-blur-sm rounded-xl p-2.5 border border-gray-800/60 text-left font-mono text-[9px] text-gray-400 custom-scrollbar max-h-32 overflow-y-auto shadow-xl">
              {layoutData.rooms?.slice(0, 6).map((rm, i) => (
                <div key={i} className="truncate leading-relaxed">
                  <span className="text-blue-400 font-bold">{i + 1}.</span> {rm.label || `Room ${i + 1}`} | {rm.dimensions || ''} | {typeof rm.area === 'number' ? rm.area.toFixed(1) : '0'}m2
                </div>
              ))}
              {layoutData.rooms?.length > 6 && (
                <div className="text-gray-600 mt-0.5">...+{layoutData.rooms.length - 6} more</div>
              )}
            </div>
          </div>
        )}

        <button
          onClick={() => setShowInfo((p) => !p)}
          className="absolute top-3 left-1/2 -translate-x-1/2 z-50 w-8 h-8 flex items-center justify-center bg-gray-900/70 backdrop-blur-sm rounded-full border border-gray-700/50 text-gray-400 hover:text-white transition-all"
          title="Toggle info"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
        </button>

        {savedMsg && (
          <div className="absolute top-14 left-1/2 -translate-x-1/2 z-50 bg-emerald-900/85 backdrop-blur-md border border-emerald-700/50 rounded-xl px-4 py-2 text-xs text-emerald-300 font-semibold shadow-xl animate-slide-up">
            {savedMsg}
          </div>
        )}
      </div>
    );
  }

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

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-6 sm:space-y-8">
        <header className="text-center max-w-2xl mx-auto space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-xs font-semibold text-blue-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            Universal 3D Spatial Engine
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight bg-gradient-to-b from-white to-gray-400 bg-clip-text text-transparent">
            Blueprint Spatial Modeler
          </h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
          <div className="lg:col-span-5 bg-gray-950/60 border border-gray-800/80 rounded-2xl p-5 sm:p-6 backdrop-blur-xl shadow-2xl space-y-5">
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

          <div className="lg:col-span-7 flex flex-col h-[50vh] sm:h-[640px] border border-gray-800/60 rounded-2xl overflow-hidden relative shadow-2xl">
            {layoutData ? (
              showEditor ? (
                <RoomCorrectionEditor
                  layoutData={layoutData}
                  imageUrl={uploadedImageUrl}
                  onConfirm={handleCorrectionConfirm}
                  onCancel={handleCorrectionCancel}
                />
              ) : (
                <div className="w-full h-full flex flex-col">
                  <div className="w-full flex-1 relative z-10">
                    <RoomExtrusionCanvas
                      layoutData={layoutData}
                      activeFloor={activeFloor}
                      imageUrl={uploadedImageUrl}
                      onRendererReady={onRendererReady}
                    />
                  </div>

                  <div className="flex items-center gap-2 px-3 py-2 bg-gray-950/90 border-t border-gray-800/60">
                    <button
                      onClick={() => setShowEditor(true)}
                      className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all whitespace-nowrap"
                    >
                      Correct
                    </button>
                    <button
                      onClick={handleDownloadScreenshot}
                      disabled={!threeRenderer}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all disabled:opacity-40 whitespace-nowrap"
                    >
                      Screenshot
                    </button>
                    <button
                      onClick={handleDownloadJSON}
                      disabled={!layoutData}
                      className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-[10px] font-bold uppercase rounded-lg transition-all disabled:opacity-40 whitespace-nowrap"
                    >
                      JSON
                    </button>
                    <div className="flex-1" />
                    <Link
                      href="/dashboard"
                      className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-[10px] font-bold uppercase rounded-lg transition-all whitespace-nowrap text-center"
                    >
                      Dashboard
                    </Link>
                  </div>
                </div>
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
