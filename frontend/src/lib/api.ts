import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// --- Types ---

export interface Camera {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  is_active: boolean;
  zone_id?: number;
  description?: string;
  created_at?: string;
}

export interface StreamSource {
  id: number;
  camera_id: number;
  name: string;
  source_type: 'video_file' | 'webcam' | 'rtsp';
  source_path: string;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  current_frame: number;
  is_active: boolean;
  latitude?: number;
  longitude?: number;
}

export interface PlaybackStatus {
  state: 'stopped' | 'playing' | 'paused';
  source_count: number;
  target_fps: number;
  queue_size: number;
  sources: any[];
}

export interface TrackPathPoint {
  camera_id: number;
  latitude: number;
  longitude: number;
  name: string;
}

export interface GlobalTrack {
  id: string;
  status: 'active' | 'lost' | 'finished';
  first_seen: string;
  last_seen: string;
  camera_sequence: number[];
  tracklet_count: number;
}

export interface SearchResult {
  track: GlobalTrack;
  score: number;
  path_points: TrackPathPoint[];
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  active_tracks: number;
  cpu_usage?: number;
  memory_usage?: number;
}

// --- API Client ---

const API_BASE = 'http://localhost:8000/api/v1';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API Error: ${res.statusText}`);
  }
  return res.json();
}

export const camerasApi = {
  list: () => fetchJson<Camera[]>('/cameras/'),
  create: (data: any) => fetchJson<Camera>('/cameras/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }),
  delete: (id: number) => fetchJson(`/cameras/${id}`, { method: 'DELETE' }),
  activate: (id: number) => fetchJson(`/cameras/${id}/activate`, { method: 'POST' }),
  deactivate: (id: number) => fetchJson(`/cameras/${id}/deactivate`, { method: 'POST' }),
};

export const streamsApi = {
  getSources: () => fetchJson<StreamSource[]>('/streams/sources'),
  addSource: (data: any) => fetchJson<StreamSource>('/streams/sources', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }),
  removeSource: (id: number) => fetchJson(`/streams/sources/${id}`, { method: 'DELETE' }),
  play: () => fetchJson('/streams/play', { method: 'POST' }),
  pause: () => fetchJson('/streams/pause', { method: 'POST' }),
  stop: () => fetchJson('/streams/stop', { method: 'POST' }),
  getStatus: () => fetchJson<PlaybackStatus>('/streams/status'),
  setFps: (fps: number) => fetchJson('/streams/fps', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fps }),
  }),
};

export const tracksApi = {
  listActive: () => fetchJson<GlobalTrack[]>('/tracks/active'),
  searchByImage: async (file: File, limit: number = 5, mode: 'auto' | 'face' | 'body' = 'auto'): Promise<SearchResult[]> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/tracks/search/image?limit=${limit}&mode=${mode}`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `API Error: ${res.statusText}`);
    }
    return res.json();
  }
};

export const createTrackingSocket = (
  onMessage: (data: any) => void,
  onDisconnect: () => void,
  onConnect: () => void,
  onError: (error: Event) => void
) => {
  const ws = new WebSocket('ws://localhost:8000/api/v1/ws/tracks');

  ws.onopen = () => {
    onConnect();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  ws.onclose = () => {
    onDisconnect();
  };

  ws.onerror = (error) => {
    onError(error);
  };

  return ws;
};

const api = {
  cameras: camerasApi,
  streams: streamsApi,
  tracks: tracksApi,
  systemInfo: () => fetchJson<SystemHealth>('/health'),
};

export default api;
