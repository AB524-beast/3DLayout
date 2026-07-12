"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getSupabase } from '../lib/supabaseClient';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const supabase = getSupabase();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUser(formatUser(session.user));
      }
      setLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ? formatUser(session.user) : null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const login = useCallback(async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
    setUser(formatUser(data.user));
    return data;
  }, []);

  const register = useCallback(async (name, email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { name } },
    });
    if (error) throw new Error(error.message);
    if (data.user) setUser(formatUser(data.user));
    return data;
  }, []);

  const logout = useCallback(async () => {
    await supabase.auth.signOut();
    setUser(null);
  }, []);

  const uploadImage = useCallback(async (file) => {
    const filePath = `avatars/${Date.now()}_${file.name}`;
    const { error } = await supabase.storage
      .from('user-images')
      .upload(filePath, file);
    if (error) throw new Error(error.message);
    const { data: publicUrlData } = supabase.storage
      .from('user-images')
      .getPublicUrl(filePath);
    return publicUrlData.publicUrl;
  }, []);

  const saveLayout = useCallback(async (filename, imageData, roomData) => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session?.user) throw new Error('Not authenticated');
    let imageUrl = null;
    if (imageData instanceof File) {
      imageUrl = await uploadImage(imageData);
    }
    const { error } = await supabase.from('layouts').insert({
      user_id: session.user.id,
      filename,
      image_url: imageUrl,
      room_data: typeof roomData === 'string' ? JSON.parse(roomData) : roomData,
    });
    if (error) throw new Error(error.message);
  }, [uploadImage]);

  const getMyLayouts = useCallback(async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session?.user) return { layouts: [] };
    const { data, error } = await supabase
      .from('layouts')
      .select('*')
      .eq('user_id', session.user.id)
      .order('created_at', { ascending: false });
    if (error) throw new Error(error.message);
    return { layouts: data || [] };
  }, []);

  return (
    <AuthContext.Provider value={{ user, token: null, loading, login, register, logout, saveLayout, getMyLayouts, uploadImage }}>
      {children}
    </AuthContext.Provider>
  );
}

function formatUser(u) {
  return {
    id: u.id,
    name: u.user_metadata?.name || '',
    email: u.email,
  };
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
