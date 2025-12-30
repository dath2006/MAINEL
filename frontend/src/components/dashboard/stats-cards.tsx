'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Camera, Users, Activity, Cpu } from 'lucide-react';

interface StatsCardsProps {
  activeTracks: number;
  totalCameras: number;
  activeCameras: number;
  detectionsPerSecond: number;
}

export function StatsCards({
  activeTracks = 0,
  totalCameras = 0,
  activeCameras = 0,
  detectionsPerSecond = 0,
}: StatsCardsProps) {
  const stats = [
    {
      title: 'Active Tracks',
      value: activeTracks,
      icon: Users,
      description: 'People being tracked',
      color: 'text-blue-500',
    },
    {
      title: 'Total Cameras',
      value: totalCameras,
      icon: Camera,
      description: `${activeCameras} active`,
      color: 'text-green-500',
    },
    {
      title: 'Detections/sec',
      value: detectionsPerSecond.toFixed(1),
      icon: Activity,
      description: 'Real-time processing',
      color: 'text-purple-500',
    },
    {
      title: 'GPU Usage',
      value: '45%',
      icon: Cpu,
      description: 'CUDA acceleration',
      color: 'text-orange-500',
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.title}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
            <stat.icon className={`h-4 w-4 ${stat.color}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stat.value}</div>
            <p className="text-xs text-muted-foreground">{stat.description}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
