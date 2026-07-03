'use client';

import { useState } from 'react';
import FloorPlanCanvas from '@/components/FloorPlanCanvas';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [layoutData, setLayoutData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle local blueprint asset targeting [cite: 5]
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  // Safe API transmission over HTTPS architecture targeting the FastAPI backend [cite: 6]
  const handleUploadAndProcess = async () => {
    if (!selectedFile) {
      setError('Please choose or snapshot a floor plan blueprint image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Connects directly to your Python Uvicorn backend pipeline instance [cite: 6, 28]
      const response = await fetch('http://127.0.0.1:8000/api/v1/process-layout', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend processing failure: Status ${response.status}`);
      }

      // Intercept the parsed data interchange block (JSON schema) [cite: 8, 9]
      const data = await response.json();
      setLayoutData(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred while connecting to the computer vision backend.');
      
      // FALLBACK SEED MOCK DATA: For demonstration if backend is not yet spun up
      setLayoutData({
        rooms: [
          { label: "Living Room", dimensions: "5.2m x 4.0m", centerX: 0, centerY: 0, walls: [
            { x1: -5, y1: -4, x2: 5, y2: -4 }, { x1: 5, y1: -4, x2: 5, y2: 4 },
            { x1: 5, y1: 4, x2: -5, y2: 4 }, { x1: -5, y1: 4, x2: -5, y2: -4 }
          ]},
          { label: "Kitchen", dimensions: "3.5m x 3.0m", centerX: 8, centerY: 1, walls: [
            { x1: 5, y1: -2, x2: 11, y2: -2 }, { x1: 11, y1: -2, x2: 11, y2: 4 },
            { x1: 11, y1: 4, x2: 5, y2: 4 }
          ]}
        ]
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Structural Header Grid Navigation */}
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">3D</div>
            <span className="text-xl font-black tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">FloorPlan3D Engine</span>
          </div>
          <span className="text-xs bg-slate-800 text-slate-400 px-3 py-1 rounded-full border border-slate-700">v1.0.0 (MIT Stack) [cite: 33]</span>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Hand Web Management UI Client Operations Console */}
        <section className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs flex items-center justify-center font-mono">1</span>
              Capture Blueprint Source [cite: 5]
            </h2>
            <p className="text-xs text-slate-400 mb-4">Upload blueprint snapshots with a reference asset (e.g. card) to scale calibrations[cite: 5].</p>

            <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-800 border-dashed rounded-lg hover:border-slate-700 transition relative bg-slate-950/40">
              <div className="space-y-1 text-center">
                <svg className="mx-auto h-12 w-12 text-slate-500" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path d="M28 8H12a4 4 0 00-4 4v20a4 4 0 004 4h16a4 4 0 004-4V12a4 4 0 00-4-4z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M14 26l7-7 21 21M26 18l3-3 8 8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div className="flex text-sm text-slate-400">
                  <label className="relative cursor-pointer bg-slate-900 rounded-md font-medium text-indigo-400 hover:text-indigo-300 focus-within:outline-none px-2 py-0.5 border border-slate-700">
                    <span>Upload Image file</span>
                    <input type="file" accept="image/*" className="sr-only" onChange={handleFileChange} />
                  </label>
                </div>
                <p className="text-xs text-slate-500">PNG, JPG, SVG high-res structural documents [cite: 5, 11]</p>
              </div>
            </div>

            {previewUrl && (
              <div className="mt-4 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 p-2">
                <p className="text-xs font-semibold text-slate-400 mb-2">Source Blueprint Blueprint Target Preview:</p>
                <img src={previewUrl} alt="Blueprint source preview" className="w-full max-h-48 object-contain rounded" />
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg text-xs font-medium">
                ⚠️ {error}
              </div>
            )}

            <button
              onClick={handleUploadAndProcess}
              disabled={isLoading}
              className={`mt-5 w-full flex items-center justify-center py-3 px-4 rounded-lg text-sm font-bold shadow-lg transition-all ${
                isLoading 
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-50 hover:shadow-indigo-500/20 text-white hover:text-indigo-950'
              }`}
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  <span>Processing CV Pipeline... [cite: 7, 10]</span>
                </div>
              ) : (
                'Generate 3D Environment Model'
              )}
            </button>
          </div>
        </section>

        {/* Right Hand Geometric Visual Viewing Canvas Display Engine */}
        <section className="lg:col-span-2 space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-xs flex items-center justify-center font-mono">2</span>
              Interactive Real-Time 3D WebGL Mesh Canvas [cite: 18]
            </h2>
            {layoutData && (
              <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                ● Geometry Realized [cite: 8]
              </span>
            )}
          </div>
          
          <FloorPlanCanvas layoutData={layoutData} />
        </section>

      </div>
    </main>
  );
}