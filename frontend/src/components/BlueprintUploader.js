"use client";

import React, { useState } from 'react';

export default function BlueprintUploader({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useManualInput, setUseManualInput] = useState(false); 
  const [wizardStep, setWizardStep] = useState(1);

  const [numFloors, setNumFloors] = useState(1);
  const [targetSqFt, setTargetSqFt] = useState(1200);
  const [numRooms, setNumRooms] = useState(4);
  const [roomsConfig, setRoomsConfig] = useState([]);

  const BACKEND_URL = "http://127.0.0.1:8000/api/v1/process-layout";

  const initializeWizardStepTwo = () => {
    const baselineRoomSize = Math.floor((targetSqFt * 0.75) / numRooms); 
    const initialRooms = Array.from({ length: numRooms }).map((_, idx) => ({
      name: idx === 0 ? "Living Room" : idx === 1 ? "Bedroom" : `Room ${idx + 1}`,
      floorAssigned: 1,
      isOpenSpace: false,
      roomSqFt: baselineRoomSize
    }));
    setRoomsConfig(initialRooms);
    setWizardStep(2);
    setError(null);
  };

  const updateRoomProperty = (index, key, value) => {
    const updated = [...roomsConfig];
    updated[index][key] = value;
    setRoomsConfig(updated);
  };

  const totalAllocated = roomsConfig.reduce((sum, r) => sum + (parseFloat(r.roomSqFt) || 0), 0);
  const freeSpaceLeft = targetSqFt - totalAllocated;

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    if (freeSpaceLeft < 0) {
      setError(`Your configurations exceed maximum boundaries.`);
      return;
    }
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/procedural`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_sq_ft: targetSqFt,
          total_floors: numFloors,
          rooms: roomsConfig
        }),
      });
      if (!response.ok) throw new Error('Procedural matrix mapping failed.');
      const data = await response.json();
      if (onUploadSuccess) onUploadSuccess(data, null);
    } catch (err) {
      setError(err.message || 'Error processing metrics.');
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
      if (!response.ok) throw new Error('Backend failed to analyze blueprint image.');
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
      <div className="flex border-b border-gray-800 text-xs font-bold tracking-wide uppercase">
        <button type="button" onClick={() => { setUseManualInput(false); setWizardStep(1); }} className={`flex-1 pb-2 text-center border-b-2 transition-colors ${!useManualInput ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-400'}`}>
          📷 Extract From Image
        </button>
        <button type="button" onClick={() => setUseManualInput(true)} className={`flex-1 pb-2 text-center border-b-2 transition-colors ${useManualInput ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-400'}`}>
          ⚡ Guided Step Wizard
        </button>
      </div>

      {!useManualInput ? (
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
        </div>
      ) : (
        <div className="bg-gray-950 border border-gray-800 rounded-xl p-4 space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-xs font-bold text-blue-400 uppercase">Step {wizardStep} of 2</span>
          </div>

          {wizardStep === 1 ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Total Floors</label>
                  <input type="number" min="1" max="5" value={numFloors} onChange={(e) => setNumFloors(parseInt(e.target.value) || 1)} className="w-full bg-gray-900 border border-gray-800 rounded-lg p-2 text-xs font-semibold text-white text-center focus:outline-none" />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-gray-400 uppercase mb-1">Rooms</label>
                  <input type="number" min="1" max="15" value={numRooms} onChange={(e) => setNumRooms(parseInt(e.target.value) || 1)} className="w-full bg-gray-900 border border-gray-800 rounded-lg p-2 text-xs font-semibold text-white text-center focus:outline-none" />
                </div>
              </div>
              <button type="button" onClick={initializeWizardStepTwo} className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 rounded-xl text-xs uppercase tracking-wider transition-all">
                Next Step →
              </button>
            </div>
          ) : (
            <form onSubmit={handleManualSubmit} className="space-y-4">
              <div className="max-h-56 overflow-y-auto space-y-3 pr-1 custom-scrollbar">
                {roomsConfig.map((room, index) => (
                  <div key={index} className="p-3 bg-gray-900 border border-gray-800 rounded-xl space-y-2">
                    <div className="grid grid-cols-12 gap-2">
                      <input type="text" value={room.name} onChange={(e) => updateRoomProperty(index, "name", e.target.value)} className="col-span-6 bg-gray-950 border border-gray-800 rounded-lg p-1.5 text-xs text-white" required />
                      <input type="number" value={room.roomSqFt} onChange={(e) => updateRoomProperty(index, "roomSqFt", parseInt(e.target.value) || 0)} className="col-span-6 bg-gray-950 border border-gray-800 rounded-lg p-1.5 text-xs text-white" required />
                    </div>
                  </div>
                ))}
              </div>
              <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-xl text-xs font-bold uppercase">Generate 3D Model</button>
            </form>
          )}
        </div>
      )}

      {error && <div className="p-3 bg-red-950/40 border border-red-900/60 rounded-xl text-xs text-red-400">⚠️ {error}</div>}
    </div>
  );
}