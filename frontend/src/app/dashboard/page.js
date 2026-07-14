"use client";

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { GridScan } from '@/components/GridScan/GridScan';
import { useAuth } from '../../context/AuthContext';

export default function DashboardPage() {
  const { user, loading, getMyLayouts, deleteLayout } = useAuth();
  const router = useRouter();
  const [layouts, setLayouts] = useState([]);
  const [fetching, setFetching] = useState(true);
  const [downloading, setDownloading] = useState(null);
  const [deleting, setDeleting] = useState(null);

  const handleDownload = async (layout) => {
    if (!layout.image_url) return;
    setDownloading(layout.id);
    try {
      const res = await fetch(layout.image_url);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = layout.filename?.replace(/\.json$/, '.png') || 'blueprint.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
    } finally {
      setDownloading(null);
    }
  };

  const handleDelete = async (layoutId) => {
    if (!confirm('Delete this layout permanently?')) return;
    setDeleting(layoutId);
    try {
      await deleteLayout(layoutId);
      setLayouts((prev) => prev.filter((l) => l.id !== layoutId));
    } catch (err) {
      alert('Delete failed: ' + err.message);
    } finally {
      setDeleting(null);
    }
  };

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
      return;
    }
    if (user) {
      getMyLayouts()
        .then((data) => setLayouts(data.layouts || []))
        .catch(() => {})
        .finally(() => setFetching(false));
    }
  }, [user, loading]);

  if (loading || fetching) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <p className="text-gray-500 text-sm">Loading...</p>
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

      <div className="relative max-w-5xl mx-auto px-6 py-12">
        <div className="flex items-center justify-between mb-8 animate-slide-up">
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight">Dashboard</h1>
            <p className="text-sm text-gray-500 mt-1">Welcome back, {user?.name}</p>
          </div>
          <Link
            href="/"
            className="text-xs font-semibold bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-xl transition-all hover:scale-105 active:scale-95"
          >
            + New Layout
          </Link>
        </div>

        {layouts.length === 0 ? (
          <div className="border-2 border-dashed border-gray-900/60 rounded-2xl bg-gray-950/40 text-center p-16 animate-scale-in">
            <div className="w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center border border-gray-800 text-lg mx-auto mb-4">🏗️</div>
            <h3 className="text-sm font-semibold text-gray-400">No saved layouts yet</h3>
            <p className="text-xs text-gray-600 mt-1 max-w-xs mx-auto">
              Create your first layout from the home page and save it here.
            </p>
            <Link
              href="/"
              className="inline-block mt-4 text-xs font-semibold bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-xl transition-all hover:scale-105 active:scale-95"
            >
              Go to Home
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {layouts.map((layout, index) => (
              <div
                key={layout.id}
                className="bg-gray-950/60 border border-gray-800/80 rounded-2xl p-5 backdrop-blur-xl hover:border-gray-700 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/5 animate-slide-up"
                style={{ animationDelay: `${index * 60}ms` }}
              >
                {layout.image_url && (
                  <img
                    src={layout.image_url}
                    alt={layout.filename}
                    className="w-full h-32 object-cover rounded-xl mb-3 border border-gray-800 transition-transform duration-300 hover:scale-[1.02]"
                  />
                )}
                <div className="text-xs font-bold text-blue-400 mb-1">{layout.filename}</div>
                <div className="text-[10px] text-gray-500">
                  Created: {new Date(layout.created_at).toLocaleDateString()}
                </div>
                {layout.room_data && (
                  <div className="mt-2 text-[10px] text-gray-600">
                    {layout.room_data.rooms?.length || 0} rooms
                  </div>
                )}
                {layout.image_url && (
                  <button
                    onClick={() => handleDownload(layout)}
                    disabled={downloading === layout.id}
                    className="mt-3 w-full bg-gray-900 hover:bg-gray-800 border border-gray-800 text-gray-300 font-semibold py-1.5 rounded-lg text-[10px] uppercase tracking-wider transition-all duration-200 disabled:opacity-50 hover:scale-[1.02] active:scale-95"
                  >
                    {downloading === layout.id ? 'Downloading...' : 'Download Blueprint'}
                  </button>
                )}
                <button
                  onClick={() => handleDelete(layout.id)}
                  disabled={deleting === layout.id}
                  className="mt-2 w-full bg-red-950/50 hover:bg-red-950 border border-red-900/60 text-red-400 font-semibold py-1.5 rounded-lg text-[10px] uppercase tracking-wider transition-all duration-200 disabled:opacity-50 hover:scale-[1.02] active:scale-95"
                >
                  {deleting === layout.id ? 'Deleting...' : 'Delete'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
