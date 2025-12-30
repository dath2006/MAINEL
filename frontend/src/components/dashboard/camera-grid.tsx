'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Camera } from '@/lib/api';
import { Camera as CameraIcon, Video, VideoOff } from 'lucide-react';

interface CameraGridProps {
  cameras: Camera[];
  onCameraSelect?: (camera: Camera) => void;
}

export function CameraGrid({ cameras = [], onCameraSelect }: CameraGridProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">Camera Overview</CardTitle>
        <Button variant="outline" size="sm">
          Add Camera
        </Button>
      </CardHeader>
      <CardContent>
        {cameras.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-muted-foreground">
            <div className="text-center">
              <CameraIcon className="mx-auto h-8 w-8 opacity-50" />
              <p className="mt-2 text-sm">No cameras configured</p>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {cameras.map((camera) => (
              <div
                key={camera.id}
                className="group relative cursor-pointer overflow-hidden rounded-lg border bg-muted/50 transition-all hover:border-primary"
                onClick={() => onCameraSelect?.(camera)}
              >
                {/* Placeholder for camera preview */}
                <div className="flex h-24 items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                  {camera.is_active ? (
                    <Video className="h-8 w-8 text-green-500/50" />
                  ) : (
                    <VideoOff className="h-8 w-8 text-red-500/50" />
                  )}
                </div>
                
                {/* Camera info */}
                <div className="p-3">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{camera.name}</span>
                    <Badge variant={camera.is_active ? 'default' : 'secondary'}>
                      {camera.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
