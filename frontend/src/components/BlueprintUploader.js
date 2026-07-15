"use client";

import React, { useState, useEffect } from 'react';

export default function BlueprintUploader({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [numFloors, setNumFloors] = useState(1);
  const [backendOk, setBackendOk] = useState(null);

  const BACKEND_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000") + "/api/v1/process-layout";

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
    let cancelled = false;

    const checkHealth = async (attempt = 1) => {
      try {
        const res = await fetch(`${base}/health`, { method: 'GET' });
        if (!cancelled) setBackendOk(res.ok);
        if (!res.ok && attempt < 5 && !cancelled) {
          setTimeout(() => checkHealth(attempt + 1), attempt * 3000);
        }
      } catch {
        if (!cancelled) setBackendOk(false);
        if (attempt < 5 && !cancelled) {
          setTimeout(() => checkHealth(attempt + 1), attempt * 3000);
        }
      }
    };

    checkHealth();
    return () => { cancelled = true; };
  }, []);

  const loadSampleLayout = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/sample?floors=${numFloors}`);
      if (!response.ok) {
        let detail = '';
        try { const e = await response.json(); detail = e.detail || ''; } catch {}
        throw new Error(detail || `Server error (${response.status})`);
      }
      const data = await response.json();
      if (onUploadSuccess) onUploadSuccess(data, null);
    } catch (err) {
      setError(err.message || 'Error loading sample layout.');
    } finally {
      setIsLoading(false);
    }
  };

  const processImageFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please provide a valid schematic picture file.');
      return;
    }
    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BACKEND_URL}/image?floors=${numFloors}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let detail = '';
        try { const e = await response.json(); detail = e.detail || ''; } catch {}
        throw new Error(detail || `Server error (${response.status})`);
      }
      const data = await response.json();
      if (onUploadSuccess) onUploadSuccess(data, file);
    } catch (err) {
      setError(err.message || 'Error running server parser.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full space-y-4">
      {backendOk === false && (
        <div className="p-3 bg-yellow-950/40 border border-yellow-900/60 rounded-xl text-xs text-yellow-400">
          ⚠️ Backend server not reachable. Make sure it&apos;s running at {process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"}.
        </div>
      )}

      <div className="space-y-4">
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-3">
          <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Target Floors to Extrude</label>
          <input type="number" min="1" max="5" value={numFloors} onChange={(e) => setNumFloors(parseInt(e.target.value) || 1)} className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-xs font-semibold text-white text-center focus:outline-none" />
        </div>
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files?.[0]) processImageFile(e.dataTransfer.files[0]); }}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${isDragging ? 'border-blue-500 bg-blue-950/20 scale-[1.01]' : 'border-gray-800 bg-gray-950 hover:border-gray-700'} ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
        >
          <input type="file" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" onChange={(e) => e.target.files?.[0] && processImageFile(e.target.files[0])} accept="image/*" />
          <p className="text-sm font-medium text-gray-300">{isLoading ? "Running recognition logic..." : "Drag & drop image blueprint layout here, or click to browse"}</p>
        </div>
        <button
          type="button"
          onClick={loadSampleLayout}
          disabled={isLoading}
          className="w-full bg-gray-900 hover:bg-gray-800 border border-gray-800 text-gray-300 font-semibold py-2 rounded-xl text-xs uppercase tracking-wider transition-all disabled:opacity-50"
        >
          🗂️ Load Cached Sample Layout
        </button>
      </div>

      {error && <div className="p-3 bg-red-950/40 border border-red-900/60 rounded-xl text-xs text-red-400">⚠️ {error}</div>}
    </div>
  );
}
