"use client";

import React from 'react';
import Link from 'next/link';
import { useAuth } from '../context/AuthContext';

export default function NavBar() {
  const { user, logout } = useAuth();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3 bg-black/80 backdrop-blur-md border-b border-gray-900">
      <Link href="/" className="text-sm font-bold text-white tracking-tight">
        Blueprint Spatial Modeler
      </Link>
      <div className="flex items-center gap-4">
        {user ? (
          <>
            <span className="text-xs text-gray-400">{user.name}</span>
            <button
              onClick={logout}
              className="text-xs font-semibold text-gray-400 hover:text-white transition-colors"
            >
              Sign Out
            </button>
          </>
        ) : (
          <Link
            href="/login"
            className="text-xs font-semibold text-gray-400 hover:text-white transition-colors"
          >
            Sign In
          </Link>
        )}
      </div>
    </nav>
  );
}
