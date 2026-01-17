'use client';

import { create } from 'zustand';
import type { Camera, GlobalTrack, SystemHealth } from './api';

interface AppState {
  // System
  systemHealth: SystemHealth | null;
  isConnected: boolean;

  // Cameras
  cameras: Camera[];
  selectedCamera: Camera | null;

  // Tracks
  activeTracks: GlobalTrack[];
  selectedTrack: GlobalTrack | null;

  // Real-time events
  recentEvents: Array<{
    type: string;
    data: unknown;
    timestamp: Date;
  }>;

  // Actions
  setSystemHealth: (health: SystemHealth) => void;
  setConnected: (connected: boolean) => void;
  setCameras: (cameras: Camera[]) => void;
  selectCamera: (camera: Camera | null) => void;
  setActiveTracks: (tracks: GlobalTrack[]) => void;
  selectTrack: (track: GlobalTrack | null) => void;
  addEvent: (type: string, data: unknown) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  systemHealth: null,
  isConnected: false,
  cameras: [],
  selectedCamera: null,
  activeTracks: [],
  selectedTrack: null,
  recentEvents: [],

  // Actions
  setSystemHealth: (health) => set({ systemHealth: health }),
  setConnected: (connected) => set({ isConnected: connected }),
  setCameras: (cameras) => set({ cameras }),
  selectCamera: (camera) => set({ selectedCamera: camera }),
  setActiveTracks: (tracks) => set({ activeTracks: tracks }),
  selectTrack: (track) => set({ selectedTrack: track }),
  addEvent: (type, data) =>
    set((state) => ({
      recentEvents: [
        { type, data, timestamp: new Date() },
        ...state.recentEvents.slice(0, 99), // Keep last 100 events
      ],
    })),
}));
