'use client';

import { useState } from 'react';
import FloorPlanCanvas from '@/components/FloorPlanCanvas';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [layoutData, setLayoutData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [activeRoom, setActiveRoom] = useState(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file (PNG, JPG, JPEG).');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError('Image too large. Maximum size is 10MB.');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setNotice(null);
      setLayoutData(null);
      setActiveRoom(null);
    }
  };

  const handleUploadAndProcess = async () => {
    if (!selectedFile) {
      setError('Please choose a floor plan blueprint image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setNotice(null);
    setActiveRoom(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/api/v1/process-layout`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Backend error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        setNotice(data.error);
      }
      
      setLayoutData(data);
    } catch (err) {
      console.error('Upload error:', err);
      
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Cannot connect to backend server. Please ensure the backend is running on ' + API_URL);
      } else {
        setError(err.message || 'An error occurred while processing the blueprint.');
      }
      
      setLayoutData(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRoomClick = (roomIndex) => {
    setActiveRoom(activeRoom === roomIndex ? null : roomIndex);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">3D</div>
            <span className="text-xl font-black tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">FloorPlan3D Engine</span>
          </div>
          <span className="text-xs bg-slate-800 text-slate-400 px-3 py-1 rounded-full border border-slate-700">v1.0.1</span>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Panel - Upload & Room List */}
        <section className="space-y-6">
          {/* Upload Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs flex items-center justify-center font-mono">1</span>
              Capture Blueprint Source
            </h2>
            <p className="text-xs text-slate-400 mb-4">Upload blueprint snapshots to generate 3D room layouts.</p>

            <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-800 border-dashed rounded-lg hover:border-slate-700 transition relative bg-slate-950/40">
              <div className="space-y-1 text-center">
                <svg className="mx-auto h-12 w-12 text-slate-500" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path d="M28 8H12a4 4 0 00-4 4v20a4 4 0 004 4h16a4 4 0 004-4V12a4 4 0 00-4-4z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M14 26l7-7 21 21M26 18l3-3 8 8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div className="flex text-sm text-slate-400 justify-center">
                  <label className="relative cursor-pointer bg-slate-900 rounded-md font-medium text-indigo-400 hover:text-indigo-300 focus-within:outline-none px-2 py-0.5 border border-slate-700">
                    <span>Upload Image file</span>
                    <input type="file" accept="image/*" className="sr-only" onChange={handleFileChange} />
                  </label>
                </div>
                <p className="text-xs text-slate-500">PNG, JPG, JPEG supported</p>
              </div>
            </div>

            {previewUrl && (
              <div className="mt-4 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 p-2">
                <p className="text-xs font-semibold text-slate-400 mb-2">Source Blueprint Preview:</p>
                <img src={previewUrl} alt="Blueprint preview" className="w-full max-h-48 object-contain rounded" />
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg text-xs font-medium">
                ⚠️ {error}
              </div>
            )}

            {notice && (
              <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded-lg text-xs font-medium">
                ℹ️ {notice} Showing default room layout.
              </div>
            )}

            <button
              onClick={handleUploadAndProcess}
              disabled={isLoading || !selectedFile}
              className={`mt-5 w-full flex items-center justify-center py-3 px-4 rounded-lg text-sm font-bold shadow-lg transition-all ${
                isLoading || !selectedFile
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-500 hover:shadow-indigo-500/20 text-white'
              }`}
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  <span>Processing CV Pipeline...</span>
                </div>
              ) : (
                'Generate 3D Environment Model'
              )}
            </button>
          </div>

          {/* Room List / Interactive Buttons */}
          {layoutData && layoutData.rooms && layoutData.rooms.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
              <h2 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 text-xs flex items-center justify-center font-mono">2</span>
                Detected Rooms ({layoutData.totalRooms || layoutData.rooms.length})
              </h2>
              <p className="text-xs text-slate-400 mb-4">Click a room to highlight it in the 3D view.</p>
              
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {layoutData.rooms.map((room, index) => (
                  <button
                    key={index}
                    onClick={() => handleRoomClick(index)}
                    className={`w-full text-left p-3 rounded-lg border transition-all duration-200 ${
                      activeRoom === index
                        ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-lg shadow-indigo-500/10'
                        : 'bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${activeRoom === index ? 'bg-indigo-400 animate-pulse' : 'bg-slate-500'}`} />
                        <span className="font-semibold text-sm">{room.label || `Room ${index + 1}`}</span>
                      </div>
                      <span className="text-xs text-slate-400 font-mono">{room.dimensions}</span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500 pl-4">
                      Center: ({room.centerX?.toFixed(1)}m, {room.centerY?.toFixed(1)}m) • {room.walls?.length || 0} walls
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Right Panel - 3D Canvas */}
        <section className="lg:col-span-2 space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-xs flex items-center justify-center font-mono">3</span>
              Interactive Real-Time 3D WebGL Mesh Canvas
            </h2>
            {layoutData && (
              <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                ● Geometry Realized
              </span>
            )}
          </div>
          
          <FloorPlanCanvas layoutData={layoutData} activeRoom={activeRoom} />
        </section>

      </div>
    </main>
  );
}