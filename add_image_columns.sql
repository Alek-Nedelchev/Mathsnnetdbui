-- Add image-related columns to mathnet table
-- Run this in Supabase SQL Editor before running update_images.py

alter table mathnet add column if not exists has_images boolean default false;
alter table mathnet add column if not exists num_images integer default 0;
alter table mathnet add column if not exists images_data jsonb default null;
