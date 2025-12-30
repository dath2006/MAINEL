'use client';

import { useEffect, useState, useCallback } from 'react';
import { Header } from '@/components/layout/header';
import { VideoFeed } from '@/components/tracking/video-feed';
import { StreamControls } from '@/components/tracking/stream-controls';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Plus, Upload, Video, Camera } from 'lucide-react';
import { streamsApi, type StreamSource, type PlaybackStatus } from '@/lib/api';
import { FullscreenView } from '@/components/tracking/fullscreen-view';
import { PersonGallery } from '@/components/tracking/person-gallery';

export default function TrackingPage() {
  const [sources, setSources] = useState<StreamSource[]>([]);
  const [status, setStatus] = useState<PlaybackStatus>({
    state: 'stopped',
    source_count: 0,
    target_fps: 30,
    queue_size: 0,
    sources: [],
  });
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newSource, setNewSource] = useState({
    camera_id: 1,
    source_path: '',
    source_type: 'video_file' as 'video_file' | 'webcam' | 'rtsp',
    name: '',
  });

  const [focusedSource, setFocusedSource] = useState<StreamSource | null>(null);

  // Fetch sources and status
  const fetchData = useCallback(async () => {
    try {
      const [sourcesData, statusData] = await Promise.all([
        streamsApi.getSources(),
        streamsApi.getStatus(),
      ]);
      setSources(sourcesData);
      setStatus(statusData);
    } catch (error) {
      console.error('Failed to fetch stream data:', error);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Poll for status updates
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // WebSocket state
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [frameCount, setFrameCount] = useState(0);

  // WebSocket for real-time frame updates
  useEffect(() => {
    console.log('Connecting to WebSocket...');
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/tracks');
    
    ws.onopen = () => {
      console.log('WebSocket connected!');
      setWsStatus('connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Dispatch custom event for VideoFeed components
        if (data.type === 'frame') {
          setFrameCount((c) => c + 1);
          window.dispatchEvent(new CustomEvent('video-frame', { detail: data }));
        }
      } catch (e) {
        console.error('WebSocket parse error:', e);
      }
    };
    
    ws.onerror = (e) => {
      console.error('WebSocket error:', e);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsStatus('disconnected');
    };

    return () => ws.close();
  }, []);

  const [addError, setAddError] = useState<string | null>(null);

  const handleAddSource = async () => {
    setAddError(null);
    
    if (!newSource.source_path.trim()) {
      setAddError('Please enter a source path');
      return;
    }
    
    try {
      await streamsApi.addSource({
        camera_id: newSource.camera_id,
        source_type: newSource.source_type,
        source_path: newSource.source_path,
        name: newSource.name || undefined,
      });
      setIsAddDialogOpen(false);
      setNewSource({ camera_id: 1, source_path: '', source_type: 'video_file', name: '' });
      fetchData();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Failed to add source';
      setAddError(message);
      console.error('Failed to add source:', error);
    }
  };

  const handleDeleteSource = async (sourceId: number) => {
    try {
      await streamsApi.removeSource(sourceId);
      fetchData();
    } catch (error) {
      console.error('Failed to remove source:', error);
    }
  };

  const handlePlay = async () => {
    await streamsApi.play();
    fetchData();
  };

  const handlePause = async () => {
    await streamsApi.pause();
    fetchData();
  };

  const handleStop = async () => {
    await streamsApi.stop();
    fetchData();
  };

  const handleFpsChange = async (fps: number) => {
    await streamsApi.setFps(fps);
  };

  return (
    <div className="flex flex-1 flex-col">
      <Header title="Live Tracking" />
      
      <main className="flex-1 space-y-4 p-6">
        {/* Debug Status Bar */}
        <div className="flex items-center gap-4 rounded bg-slate-800 p-2 text-xs">
          <span>WS: <strong className={wsStatus === 'connected' ? 'text-green-400' : 'text-red-400'}>{wsStatus}</strong></span>
          <span>Frames received: <strong>{frameCount}</strong></span>
          <span>Sources: <strong>{sources.length}</strong></span>
          <span>State: <strong>{status.state}</strong></span>
        </div>
        
        {/* Stream Controls */}
        <StreamControls
          onPlay={handlePlay}
          onPause={handlePause}
          onStop={handleStop}
          state={status.state as 'stopped' | 'playing' | 'paused'}
          sourceCount={sources.length}
          fps={status.target_fps}
          onFpsChange={handleFpsChange}
        />

        {/* Fullscreen Overlay */}
        {focusedSource && (
            <FullscreenView 
                source={focusedSource} 
                onClose={() => setFocusedSource(null)} 
            />
        )}

        {/* Add Source Button */}
        <div className="flex justify-end">
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                Add Source
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Video Source</DialogTitle>
              </DialogHeader>
              <Tabs defaultValue="file" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="file" onClick={() => setNewSource(s => ({ ...s, source_type: 'video_file' }))}>
                    <Video className="mr-2 h-4 w-4" />
                    Video File
                  </TabsTrigger>
                  <TabsTrigger value="webcam" onClick={() => setNewSource(s => ({ ...s, source_type: 'webcam' }))}>
                    <Camera className="mr-2 h-4 w-4" />
                    Webcam
                  </TabsTrigger>
                  <TabsTrigger value="rtsp" onClick={() => setNewSource(s => ({ ...s, source_type: 'rtsp' }))}>
                    <Upload className="mr-2 h-4 w-4" />
                    RTSP/IP
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="file" className="space-y-4">
                  <Input
                    placeholder="Video file path (e.g., C:/videos/cam1.mp4)"
                    value={newSource.source_path}
                    onChange={(e) => setNewSource({ ...newSource, source_path: e.target.value })}
                  />
                </TabsContent>

                <TabsContent value="webcam" className="space-y-4">
                  <Input
                    placeholder="Camera index (0 for default webcam)"
                    type="number"
                    min={0}
                    value={newSource.source_path}
                    onChange={(e) => setNewSource({ ...newSource, source_path: e.target.value })}
                  />
                </TabsContent>

                <TabsContent value="rtsp" className="space-y-4">
                  <Input
                    placeholder="RTSP URL (e.g., rtsp://ip:port/stream)"
                    value={newSource.source_path}
                    onChange={(e) => setNewSource({ ...newSource, source_path: e.target.value })}
                  />
                </TabsContent>
              </Tabs>

              <div className="space-y-4 pt-4">
                <Input
                  placeholder="Camera ID"
                  type="number"
                  min={1}
                  value={newSource.camera_id}
                  onChange={(e) => setNewSource({ ...newSource, camera_id: parseInt(e.target.value) || 1 })}
                />
                <Input
                  placeholder="Display name (optional)"
                  value={newSource.name}
                  onChange={(e) => setNewSource({ ...newSource, name: e.target.value })}
                />
                {addError && (
                  <p className="text-sm text-destructive">{addError}</p>
                )}
                <Button onClick={handleAddSource} className="w-full">
                  Add Source
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Video Grid */}
        {sources.length === 0 ? (
          <Card>
            <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
              <Video className="mb-4 h-12 w-12 opacity-50" />
              <p>No video sources added</p>
              <p className="text-sm">Click "Add Source" to add video files or cameras</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {sources.map((source) => (
              <VideoFeed
                key={source.id}
                cameraId={source.camera_id}
                sourceName={source.name}
                isActive={source.is_active && status.state === 'playing'}
                onDelete={() => handleDeleteSource(source.id)}
                onMaximize={() => setFocusedSource(source)}
              />
            ))}
          </div>
        )}

        {/* Person Gallery */}
        <div className="mt-6">
          <PersonGallery />
        </div>
      </main>
    </div>
  );
}
