-- ============================================================================
-- Migration: keepalive -- dedicated table for the daily Supabase keepalive
-- ============================================================================
-- Supabase free-tier projects pause after 7 days with no real API activity
-- against the database -- dashboard visits don't count. The
-- .github/workflows/supabase-keepalive.yml workflow pings this table once a
-- day with the anon key to generate that activity.
--
-- This table exists solely so that ping has something to SELECT that
-- returns 200. It holds no user data -- just a single fixed row -- so
-- granting `anon` SELECT here cannot expose anything, even if RLS were
-- later disabled by mistake.
--
-- Deliberately separate from public.profiles: profiles holds user email
-- addresses and must never be granted to anon, so the keepalive ping is
-- pointed at this table instead of reusing an existing one.
-- ============================================================================

create table if not exists public.keepalive (
  id smallint primary key,
  note text not null default 'row exists so the daily keepalive ping returns 200'
);

insert into public.keepalive (id) values (1) on conflict (id) do nothing;

alter table public.keepalive enable row level security;

drop policy if exists "keepalive is publicly readable" on public.keepalive;
create policy "keepalive is publicly readable" on public.keepalive
  for select to anon using (true);

grant usage on schema public to anon;
grant select on public.keepalive to anon;
