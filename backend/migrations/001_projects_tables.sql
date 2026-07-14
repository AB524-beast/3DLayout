create table public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  name text,
  image_url text,
  total_floors int default 1,
  created_at timestamptz default now()
);

create table public.project_rooms (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete cascade not null,
  label text,
  dimensions text,
  "centerX" float8,
  "centerY" float8,
  "elevationZ" float8,
  "isOpenSpace" boolean,
  walls jsonb,
  area float8
);

alter table public.projects enable row level security;
alter table public.project_rooms enable row level security;

create policy "Users manage own projects"
  on public.projects for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "Users manage own project rooms"
  on public.project_rooms for all
  using (exists (select 1 from public.projects p where p.id = project_id and p.user_id = auth.uid()));
