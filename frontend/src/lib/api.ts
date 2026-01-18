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

export interface CameraSequenceItem {
  camera_id: number;
  first_seen?: string;
}

export interface Transition {
  from_camera: number;
  to_camera: number;
  transition_time: string;
  time_at_from: number;
}

export interface GlobalTrack {
  id: string;
  status: 'active' | 'lost' | 'finished';
  first_seen?: string;
  last_seen: string;
  camera_sequence: number[];
  tracklet_count: number;
}

export interface SearchResult {
  track: GlobalTrack;
  score: number;
  path_points: TrackPathPoint[];
  camera_sequence?: CameraSequenceItem[];
  transitions?: Transition[];
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  active_tracks: number;
  cpu_usage?: number;
  memory_usage?: number;
}

export interface VideoMetadata {
  id: string;
  filename: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  duration: number;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  uploaded_at: string;
  last_used?: string;
  use_count: number;
  description?: string;
  tags: string[];
}

// --- API Client ---

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_BASE = `${BASE_URL}/api/v1`;

// Helper for WebSocket URL
export function getWsUrl(path: string): string {
  if (path.startsWith('ws')) return path; // Already full URL
  const protocol = BASE_URL.startsWith('https') ? 'wss' : 'ws';
  const host = BASE_URL.replace(/^https?:\/\//, '');
  // Remove trailing slashes from host to avoid double slash
  const cleanHost = host.replace(/\/$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${protocol}://${cleanHost}${cleanPath}`;
}

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

  uploadVideo: async (data: {
    camera_id: number;
    name?: string;
    file: File;
    latitude?: number;
    longitude?: number;
  }) => {
    const formData = new FormData();
    formData.append('camera_id', String(data.camera_id));
    if (data.name) formData.append('name', data.name);
    if (data.latitude) formData.append('latitude', String(data.latitude));
    if (data.longitude) formData.append('longitude', String(data.longitude));
    formData.append('file', data.file);

    const res = await fetch(`${API_BASE}/streams/sources/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail);
    }

    return res.json();
  },

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
  
  createSourceFromLibrary: async (data: {
    camera_id: number;
    video_id: string;
    name?: string;
    latitude?: number;
    longitude?: number;
  }) => {
    const formData = new FormData();
    formData.append('camera_id', String(data.camera_id));
    formData.append('video_id', data.video_id);
    if (data.name) formData.append('name', data.name);
    if (data.latitude) formData.append('latitude', String(data.latitude));
    if (data.longitude) formData.append('longitude', String(data.longitude));

    const res = await fetch(`${API_BASE}/streams/sources/from-library`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to create source' }));
      throw new Error(error.detail);
    }

    return res.json();
  },
};

export const tracksApi = {
  listActive: () => fetchJson<GlobalTrack[]>('/tracks/active'),
  searchByImage: async (file: File, limit: number = 5): Promise<SearchResult[]> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/tracks/search/image?limit=${limit}`, {
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
  const wsUrl = getWsUrl('/api/v1/ws/tracks');
  const ws = new WebSocket(wsUrl);

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

export const videoLibraryApi = {
  listVideos: () => fetchJson<VideoMetadata[]>('/video-library/videos'),
  
  uploadToLibrary: async (file: File, description?: string, tags?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (description) formData.append('description', description);
    if (tags) formData.append('tags', tags);

    const res = await fetch(`${API_BASE}/video-library/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail);
    }

    return res.json();
  },
  
  getVideoInfo: (videoId: string) => fetchJson<VideoMetadata>(`/video-library/videos/${videoId}`),
  deleteVideo: (videoId: string) => fetchJson(`/video-library/videos/${videoId}`, { method: 'DELETE' }),
};

const api = {
  cameras: camerasApi,
  streams: streamsApi,
  tracks: tracksApi,
  videoLibrary: videoLibraryApi,
  systemInfo: () => fetchJson<SystemHealth>('/health'),
};

export default api;
