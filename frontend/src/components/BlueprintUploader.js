import React, { useState } from 'react';

export default function BlueprintUploader({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const processFile = async (file) => {
    // Validate file presence and format types before starting upload stream
    if (!file || !file.type.startsWith('image/')) {
      setError('Please provide a valid JPEG or PNG blueprint file.');
      return;
    }

    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/v1/process-layout', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to parse blueprint assets via Computer Vision engine.');
      }
      
      const data = await response.json();
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
    } catch (err) {
      setError(err.message || 'Network interface error occurred during processing.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto p-4">
      {/* The 'relative' class added below isolates the invisible full-bleed file 
        input bounding-box so it matches the dashed container dimensions perfectly.
      */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files?.[0]) {
            processFile(e.dataTransfer.files[0]);
          }
        }}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${isDragging 
            ? 'border-blue-500 bg-blue-50/50 scale-[1.01]' 
            : 'border-gray-300 bg-white hover:border-gray-400'
          } ${isLoading ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <input 
          type="file" 
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
          accept="image/jpeg, image/png, image/jpg"
          disabled={isLoading}
        />
        
        <div className="flex flex-col items-center justify-center space-y-3">
          {/* Animated visual loading vector indicator */}
          <svg
            className={`w-10 h-10 transition-transform duration-200 ${isDragging ? 'text-blue-500 scale-110' : 'text-gray-400'}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
            />
          </svg>

          <p className={`text-sm font-semibold transition-colors duration-150 ${isDragging ? 'text-blue-600' : 'text-gray-700'}`}>
            {isLoading 
              ? "Analyzing blueprints structure..." 
              : isDragging 
                ? "Drop here to start conversion!" 
                : "Drag & Drop blueprint here, or click to browse"
            }
          </p>
          
          {!isLoading && (
            <p className="text-xs text-gray-400">
              Supports JPEG, JPG, and PNG layout images
            </p>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-600 font-medium">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}