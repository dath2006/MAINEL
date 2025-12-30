'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Trash2, Maximize2 } from 'lucide-react';

interface VideoFeedProps {
  cameraId: number;
  sourceName?: string;
  isActive?: boolean;
  onDelete?: () => void;
  onMaximize?: () => void;
}

export function VideoFeed({
  cameraId,
  sourceName = `Camera ${cameraId}`,
  isActive = false,
  onDelete,
  onMaximize,
}: VideoFeedProps) {
  const [trackCount, setTrackCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [frameSrc, setFrameSrc] = useState<string>('');
  const [frameCount, setFrameCount] = useState(0);

  // Receive frames via custom event from parent
  useEffect(() => {
    const handleFrame = (event: CustomEvent) => {
      const data = event.detail;
      if (data.camera_id !== cameraId) return;

      // Update frame
      if (data.frame_data) {
        setFrameSrc(`data:image/jpeg;base64,${data.frame_data}`);
        setFrameCount((c) => c + 1);
      }

      // Update stats
      setTrackCount(data.track_count || 0);
      setFps(Math.round(data.fps) || 0);
    };

    window.addEventListener('video-frame' as any, handleFrame);
    return () => window.removeEventListener('video-frame' as any, handleFrame);
  }, [cameraId]);

  return (
    <Card className="overflow-hidden relative group">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{sourceName}</CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant={isActive ? 'default' : 'secondary'}>
            {trackCount} tracks
          </Badge>
          {fps > 0 && (
            <Badge variant="outline" className="text-xs">
              {fps} FPS
            </Badge>
          )}
          {onMaximize && (
            <Button
                variant="ghost" 
                size="icon" 
                onClick={onMaximize}
                className="h-6 w-6 text-muted-foreground hover:text-primary"
            >
                <Maximize2 className="h-4 w-4" />
            </Button>
          )}
          {onDelete && (
            <Button 
                variant="ghost" 
                size="icon" 
                onClick={onDelete}
                className="h-6 w-6 text-muted-foreground hover:text-destructive"
            >
                <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {frameSrc ? (
          <img
            src={frameSrc}
            alt={sourceName}
            className="w-full object-cover"
            style={{ minHeight: '180px' }}
          />
        ) : (
          <div className="flex h-44 items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
            <p className="text-sm text-muted-foreground">
              {isActive ? 'Waiting for frames...' : 'No video stream'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
