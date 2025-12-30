'use client';

import { useEffect, useState } from 'react';
import { Header } from '@/components/layout/header';
import { StatsCards } from '@/components/dashboard/stats-cards';
import { CameraGrid } from '@/components/dashboard/camera-grid';
import { ActivityFeed } from '@/components/dashboard/activity-feed';
import api, { createTrackingSocket, type Camera, type SystemHealth } from '@/lib/api';
import { useAppStore } from '@/lib/store';

export default function DashboardPage() {
  const { cameras, setCameras, recentEvents, addEvent, isConnected, setConnected } = useAppStore();
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch initial data
  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        
        // Fetch system health
        const health = await api.systemInfo();
        setSystemHealth(health);
        
        // Fetch cameras
        const cameraList = await api.cameras.list();
        setCameras(cameraList);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchData();
  }, [setCameras]);

  // WebSocket connection
  useEffect(() => {
    const ws = createTrackingSocket(
      (data) => {
        // Handle incoming events
        const event = data as { type: string };
        addEvent(event.type, data);
      },
      () => setConnected(false),
      () => setConnected(true),
      () => setConnected(false)
    );

    return () => {
      ws?.close();
    };
  }, [addEvent, setConnected]);

  const activeCameras = cameras.filter(c => c.is_active).length;

  return (
    <div className="flex flex-1 flex-col">
      <Header title="Dashboard" />
      
      <main className="flex-1 space-y-6 p-6">
        {/* Stats Cards */}
        <StatsCards
          activeTracks={systemHealth?.active_tracks ?? 0}
          totalCameras={cameras.length}
          activeCameras={activeCameras}
          detectionsPerSecond={0}
        />

        {/* Main Grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Camera Grid - 2 columns */}
          <div className="lg:col-span-2">
            <CameraGrid cameras={cameras} />
          </div>

          {/* Activity Feed - 1 column */}
          <div>
            <ActivityFeed events={recentEvents} />
          </div>
        </div>

        {/* Connection Status Banner */}
        {!isConnected && !isLoading && (
          <div className="fixed bottom-4 left-1/2 -translate-x-1/2 transform">
            <div className="rounded-lg bg-yellow-500/10 px-4 py-2 text-sm text-yellow-600">
              ⚠️ WebSocket disconnected - Real-time updates unavailable
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
