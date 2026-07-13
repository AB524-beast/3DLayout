import { createClient } from '@supabase/supabase-js'

let supabase = null

function getClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      'Missing Supabase env vars. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.'
    )
  }
  return createClient(supabaseUrl, supabaseAnonKey)
}

export function getSupabase() {
  if (typeof window === 'undefined') return null
  if (!supabase) {
    supabase = getClient()
  }
  return supabase
}
