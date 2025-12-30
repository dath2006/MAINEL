'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { X, Minimize2, Clock, Activity } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

interface FullscreenViewProps {
  source: any; // Using any for now to match backend response structure easily
  onClose: () => void;
}

export function FullscreenView({ source, onClose }: FullscreenViewProps) {
  const [frameSrc, setFrameSrc] = useState<string>('');
  const [tracks, setTracks] = useState<any[]>([]);
  const [fps, setFps] = useState(0);

  // Stats for duration
  const progress = source.total_frames > 0 
    ? (source.current_frame / source.total_frames) * 100 
    : 0;

  const formatTime = (frames: number, fps: number) => {
    if (!fps) return '00:00';
    const seconds = Math.floor(frames / fps);
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    const handleFrame = (event: CustomEvent) => {
      const data = event.detail;
      if (data.camera_id !== source.camera_id) return;

      if (data.frame_data) {
        setFrameSrc(`data:image/jpeg;base64,${data.frame_data}`);
      }
      
      // Update tracks list
      if (data.tracks) {
        setTracks(data.tracks);
      }
      
      setFps(Math.round(data.fps) || 0);
    };

    window.addEventListener('video-frame' as any, handleFrame);
    return () => window.removeEventListener('video-frame' as any, handleFrame);
  }, [source.camera_id]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex flex-1 overflow-hidden">
        {/* Main Video Area */}
        <div className="flex-1 flex flex-col p-4 relative">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold tracking-tight">{source.name}</h2>
                <Button variant="ghost" size="icon" onClick={onClose}>
                    <X className="h-6 w-6" />
                </Button>
            </div>

            <div className="flex-1 bg-black rounded-lg overflow-hidden flex items-center justify-center relative border border-border">
                {frameSrc ? (
                    <img 
                        src={frameSrc} 
                        className="h-full w-full object-contain" 
                        alt={source.name}
                    />
                ) : (
                    <div className="text-muted-foreground">Waiting for stream...</div>
                )}
                
                {/* Overlay Controls */}
                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
                     {source.source_type === 'video_file' && (
                        <div className="space-y-2">
                             <div className="flex justify-between text-xs text-white/80">
                                <span>{formatTime(source.current_frame, source.fps)}</span>
                                <span>{formatTime(source.total_frames, source.fps)}</span>
                            </div>
                            <Progress value={progress} className="h-1" />
                        </div>
                     )}
                     <div className="flex gap-4 mt-2 text-white/90 text-sm">
                        <span className="flex items-center gap-1"><Activity className="w-4 h-4" /> {fps} FPS</span>
                        <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> Frame: {source.current_frame}</span>
                     </div>
                </div>
            </div>
        </div>

        {/* Sidebar - Tracking Details */}
        <div className="w-80 border-l bg-card p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Live Analysis</h3>
                <Badge variant="outline">{tracks.length} Objects</Badge>
            </div>
            
            <div className="space-y-4">
                {tracks.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                        No objects detected
                    </div>
                ) : (
                    <div className="space-y-2">
                        {tracks.map((track: any) => (
                            <Card key={track.track_id} className="overflow-hidden">
                                <CardContent className="p-3">
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex flex-col gap-1">
                                            <div className="flex items-center gap-2">
                                                <Badge variant={track.state === 'CONFIRMED' ? 'default' : 'secondary'}>
                                                    ID: {track.track_id}
                                                </Badge>
                                                {track.global_id && (
                                                    <Badge variant="outline" className="border-blue-500 text-blue-500">
                                                        G-ID: {track.global_id.slice(0, 8)}
                                                    </Badge>
                                                )}
                                            </div>
                                            <span className="text-sm font-medium capitalize">{track.class_name}</span>
                                        </div>
                                        <span className="text-xs text-muted-foreground">
                                            {Math.round(track.confidence * 100)}%
                                        </span>
                                    </div>
                                    <div className="flex gap-2 text-xs text-muted-foreground">
                                        <span>State: {track.state}</span>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}
            </div>
        </div>
      </div>
    </div>
  );
}
