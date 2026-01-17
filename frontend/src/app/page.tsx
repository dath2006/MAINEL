'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/lib/store';
import api from '@/lib/api';
import { Header } from '@/components/layout/header';
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/layout/app-sidebar"
import { StatsCards } from '@/components/dashboard/stats-cards';
import { CameraGrid } from '@/components/dashboard/camera-grid';
import { ActivityFeed } from '@/components/dashboard/activity-feed';
import { Separator } from '@/components/ui/separator';

export default function Dashboard() {
  const {
    systemHealth,
    activeTracks,
    cameras,
    setSystemHealth,
    setCameras,
    setActiveTracks,
    recentEvents,
    addEvent
  } = useAppStore();

  // Initial Data Fetch
  useEffect(() => {
    const init = async () => {
      try {
        const [health, cams, tracks] = await Promise.all([
          api.systemInfo(),
          api.cameras.list(),
          api.tracks.listActive()
        ]);
        setSystemHealth(health);
        setCameras(cams);
        setActiveTracks(tracks);
      } catch (e) {
        console.error("Init failed", e);
      }
    };
    init();

    // Polling System Health
    const interval = setInterval(async () => {
      try {
        const health = await api.systemInfo();
        setSystemHealth(health);
      } catch (e) { }
    }, 5000);
    return () => clearInterval(interval);
  }, [setSystemHealth, setCameras, setActiveTracks]);

  return (
    <div className="flex flex-col h-screen bg-black overflow-hidden font-mono">
      <Header title="Mission_Control" />

      <main className="flex-1 grid grid-cols-12 grid-rows-1 gap-0 overflow-hidden">
        {/* L: Stats & Cameras (8 cols) */}
        <div className="col-span-12 lg:col-span-8 flex flex-col border-r border-[#262626]">
          {/* Top: Stats Slab */}
          <div className="h-48 border-b border-[#262626]">
            <StatsCards
              activeTracks={systemHealth?.active_tracks || 0}
              totalCameras={cameras.length}
              activeCameras={cameras.filter(c => c.is_active).length}
              detectionsPerSecond={0} // TODO: Add real DPS
            />
          </div>

          {/* Bottom: Cameras Grid */}
          <div className="flex-1 bg-[#050505]">
            <CameraGrid
              cameras={cameras}
              onAddCamera={() => console.log('Add Camera')}
            />
          </div>
        </div>

        {/* R: Activity Feed (4 cols) */}
        <div className="col-span-12 lg:col-span-4 h-full bg-[#020202]">
          <ActivityFeed events={recentEvents} />
        </div>
      </main>
    </div>
  );
}
