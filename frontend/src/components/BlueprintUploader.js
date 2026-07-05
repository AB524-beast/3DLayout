"use client";

import React, { useState } from 'react';

export default function BlueprintUploader({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Consolidated input specifications
  const [useManualInput, setUseManualInput] = useState(false);
  const [numRooms, setNumRooms] = useState(4);
  const [targetSqFt, setTargetSqFt] = useState(1500);
  const [numFloors, setNumFloors] = useState(1);

  // Point explicitly to your local FastAPI service port
  const BACKEND_URL = "http://127.0.0.1:8000/api/v1/process-layout";

  const processFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please provide a valid JPEG or PNG blueprint file.');
      return;
    }

    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const targetUrl = `${BACKEND_URL}?num_rooms=${numRooms}&sq_ft=${targetSqFt}&floors=${numFloors}`;
      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to extract building layout.');
      }
      
      const data = await response.json();
      if (onUploadSuccess) onUploadSuccess(data);
    } catch (err) {
      setError(err.message || 'Error connecting to the backend extraction service.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const targetUrl = `${BACKEND_URL}?num_rooms=${numRooms}&sq_ft=${targetSqFt}&floors=${numFloors}`;
      const response = await fetch(targetUrl, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to map procedural mesh parameters.');
      }
      
      const data = await response.json();
      if (onUploadSuccess) onUploadSuccess(data);
    } catch (err) {
      setError(err.message || 'Failed processing your custom dimensions.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Parameter Control Deck Grid */}
      <div className="bg-gray-900/90 border border-gray-800 rounded-xl p-4 grid grid-cols-3 gap-3">
        <div>
          <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Target Rooms</label>
          <input 
            type="number" min="1" max="30"
            value={numRooms} onChange={(e) => setNumRooms(parseInt(e.target.value) || 1)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500 text-center"
            disabled={isLoading}
          />
        </div>
        <div>
          <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Total Floors</label>
          <input 
            type="number" min="1" max="10"
            value={numFloors} onChange={(e) => setNumFloors(parseInt(e.target.value) || 1)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500 text-center"
            disabled={isLoading}
          />
        </div>
        <div>
          <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Area (Sq. Ft.)</label>
          <input 
            type="number" min="100" max="25000" step="100"
            value={targetSqFt} onChange={(e) => setTargetSqFt(parseInt(e.target.value) || 500)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500 text-center"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Primary Input Source Selection Tabs */}
      <div className="flex border-b border-gray-800 text-xs font-bold tracking-wide uppercase">
        <button 
          type="button"
          onClick={() => setUseManualInput(false)}
          className={`flex-1 pb-2 text-center border-b-2 transition-colors ${!useManualInput ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-400'}`}
        >
          📷 Extract From Image
        </button>
        <button 
          type="button"
          onClick={() => setUseManualInput(true)}
          className={`flex-1 pb-2 text-center border-b-2 transition-colors ${useManualInput ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-400'}`}
        >
          ⚡ Procedural Generation
        </button>
      </div>

      {!useManualInput ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]); }}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
            ${isDragging ? 'border-blue-500 bg-blue-950/20 scale-[1.01]' : 'border-gray-800 bg-gray-950 hover:border-gray-700'
            } ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
        >
          <input 
            type="file" 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
            onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
            accept="image/*"
            disabled={isLoading}
          />
          <p className="text-sm font-medium text-gray-300">
            {isLoading ? "Extruding structural metrics..." : "Drag & drop blueprint image here, or browse"}
          </p>
        </div>
      ) : (
        <button
          onClick={handleManualSubmit}
          disabled={isLoading}
          className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-xl text-xs tracking-wider uppercase transition-all disabled:bg-gray-800 disabled:text-gray-500"
        >
          {isLoading ? "Calculating Coordinate Matrices..." : "Generate Stacked 3D Environment"}
        </button>
      )}

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-900/60 rounded-xl text-xs text-red-400 font-medium">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}