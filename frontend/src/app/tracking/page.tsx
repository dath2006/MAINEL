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
import { streamsApi, tracksApi, type StreamSource, type PlaybackStatus, type SearchResult } from '@/lib/api';
import { FullscreenView } from '@/components/tracking/fullscreen-view';
import { PersonGallery } from '@/components/tracking/person-gallery';
import dynamic from 'next/dynamic';
import { Search, Map as MapIcon, Grid } from 'lucide-react';

const Map = dynamic(() => import('@/components/tracking/Map'), { 
  ssr: false,
  loading: () => <div className="h-[400px] w-full animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
});

const MultiTrackMap = dynamic(() => import('@/components/tracking/MultiTrackMap'), { 
  ssr: false,
  loading: () => <div className="h-[400px] w-full animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
});

const TrackTimeline = dynamic(() => import('@/components/tracking/TrackTimeline'), { 
  ssr: false 
});

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
    latitude: 0.0,
    longitude: 0.0,
  });

  const [activeTab, setActiveTab] = useState('feeds');
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

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
        latitude: newSource.latitude,
        longitude: newSource.longitude,
      });
      setIsAddDialogOpen(false);
      setNewSource({ 
        camera_id: 1, 
        source_path: '', 
        source_type: 'video_file', 
        name: '', 
        latitude: 0.0, 
        longitude: 0.0 
      });
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

  const handleSearch = async () => {
    if (!searchFile) return;
    
    setIsSearching(true);
    try {
        const results = await tracksApi.searchByImage(searchFile);
        setSearchResults(results);
        if (results.length > 0) {
            setSelectedResult(results[0]);
            setActiveTab('map');
        }
    } catch (error) {
        console.error("Search failed", error);
    } finally {
        setIsSearching(false);
    }
  };

  return (
    <div className="flex flex-1 flex-col">
      <Header title="Live Tracking" />
      
      <main className="flex-1 space-y-4 p-6">
        {/* Debug Status Bar */}
        <div className="flex items-center justify-between rounded bg-slate-800 p-2 text-xs text-white">
          <div className="flex gap-4">
            <span>WS: <strong className={wsStatus === 'connected' ? 'text-green-400' : 'text-red-400'}>{wsStatus}</strong></span>
            <span>Frames: <strong>{frameCount}</strong></span>
            <span>Sources: <strong>{sources.length}</strong></span>
            <span>State: <strong>{status.state}</strong></span>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="flex gap-2">
                <Input 
                    type="file" 
                    accept="image/*" 
                    className="h-8 w-60 text-xs text-black"
                    onChange={(e) => setSearchFile(e.target.files?.[0] || null)}
                />
                <Button size="sm" onClick={handleSearch} disabled={!searchFile || isSearching}>
                    {isSearching ? 'Searching...' : 'Search Person'}
                </Button>
            </div>
            <div className="text-xs text-muted-foreground hidden md:block">
                Tip: Click on the map to place a camera automatically.
            </div>
          </div>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <div className="flex items-center justify-between mb-4">
                <TabsList>
                    <TabsTrigger value="feeds" className="gap-2">
                        <Grid className="h-4 w-4" /> Live Feeds
                    </TabsTrigger>
                    <TabsTrigger value="map" className="gap-2">
                        <MapIcon className="h-4 w-4" /> Map View
                    </TabsTrigger>
                </TabsList>
                
                {/* Add Source Button moved here */}
                <div className="flex justify-end">
                  <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
                      <Button onClick={() => {
                          const nextId = sources.length > 0 ? Math.max(...sources.map(s => s.camera_id)) + 1 : 1;
                          setNewSource(s => ({ ...s, camera_id: nextId }));
                          setIsAddDialogOpen(true);
                      }}>
                        <Plus className="mr-2 h-4 w-4" />
                        Add Source
                      </Button>
                    <DialogContent className="max-w-xl">
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

                      <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Camera ID</label>
                            <Input
                                type="number"
                                min={1}
                                value={newSource.camera_id}
                                onChange={(e) => setNewSource({ ...newSource, camera_id: parseInt(e.target.value) || 1 })}
                            />
                          </div>
                           <div className="space-y-2">
                            <label className="text-sm font-medium">Display Name</label>
                            <Input
                                value={newSource.name}
                                onChange={(e) => setNewSource({ ...newSource, name: e.target.value })}
                            />
                          </div>
                           <div className="space-y-2">
                            <label className="text-sm font-medium">Latitude</label>
                            <Input
                                type="number"
                                step="0.000001"
                                value={newSource.latitude}
                                onChange={(e) => setNewSource({ ...newSource, latitude: parseFloat(e.target.value) || 0 })}
                            />
                          </div>
                           <div className="space-y-2">
                            <label className="text-sm font-medium">Longitude</label>
                            <Input
                                type="number"
                                step="0.000001"
                                value={newSource.longitude}
                                onChange={(e) => setNewSource({ ...newSource, longitude: parseFloat(e.target.value) || 0 })}
                            />
                          </div>
                      </div>

                      <div className="pt-4">
                        {addError && (
                          <p className="text-sm text-destructive mb-2">{addError}</p>
                        )}
                        <Button onClick={handleAddSource} className="w-full">
                          Add Source
                        </Button>
                      </div>
                    </DialogContent>
                  </Dialog>
                </div>
            </div>

            <TabsContent value="feeds" className="space-y-4">
                {/* Stream Controls - Only visible in feed view */}
                <StreamControls
                  onPlay={handlePlay}
                  onPause={handlePause}
                  onStop={handleStop}
                  state={status.state as 'stopped' | 'playing' | 'paused'}
                  sourceCount={sources.length}
                  fps={status.target_fps}
                  onFpsChange={handleFpsChange}
                />

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
                  <div className="grid gap-4 md:grid-cols-2">
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
            </TabsContent>
            
            <TabsContent value="map" className="h-[600px] w-full border rounded-lg overflow-hidden relative">
                 {/* Use MultiTrackMap for real-time updates */}
                 <MultiTrackMap 
                    sources={sources} 
                    activeTracks={selectedResult ? [{
                        globalId: selectedResult.track.id,
                        pathPoints: selectedResult.path_points || [],
                    }] : []}
                    selectedTrackId={selectedResult?.track.id}
                    onMapClick={(lat, lng) => {
                        const nextId = sources.length > 0 ? Math.max(...sources.map(s => s.camera_id)) + 1 : 1;
                        setNewSource(prev => ({
                            ...prev,
                            latitude: lat,
                            longitude: lng,
                            camera_id: nextId
                        }));
                        setIsAddDialogOpen(true);
                    }}
                 />
                 
                 {/* Search Results Overlay + Timeline */}
                 {searchResults.length > 0 && (
                     <div className="absolute top-4 right-4 z-[1000] w-72 bg-background/95 backdrop-blur shadow-lg rounded-lg border max-h-[550px] overflow-hidden flex flex-col">
                        <div className="p-4 border-b">
                            <h3 className="font-semibold">Search Results</h3>
                            <p className="text-xs text-muted-foreground">{searchResults.length} match(es) found</p>
                        </div>
                        <div className="flex-1 overflow-auto p-2">
                            <div className="space-y-2 mb-4">
                                {searchResults.map((result, idx) => (
                                    <div 
                                        key={result.track.id} 
                                        className={`p-3 rounded-lg border cursor-pointer transition-all hover:bg-accent ${
                                            selectedResult?.track.id === result.track.id 
                                                ? 'bg-accent border-primary shadow-sm' 
                                                : 'bg-card'
                                        }`}
                                        onClick={() => setSelectedResult(result)}
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-sm font-medium">Match #{idx + 1}</span>
                                            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                                                {(result.score * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            Cameras: {result.track.camera_sequence?.join(' → ') || 'N/A'}
                                        </div>
                                    </div>
                                ))}
                            </div>
                            
                            {/* Timeline for selected result */}
                            {selectedResult && (
                                <div className="border-t pt-4">
                                    <TrackTimeline
                                        globalId={selectedResult.track.id}
                                        events={selectedResult.path_points?.map((pt: any, i: number) => ({
                                            camera_id: pt.camera_id,
                                            camera_name: pt.name,
                                            timestamp: selectedResult.track.last_seen || new Date().toISOString(),
                                            latitude: pt.latitude,
                                            longitude: pt.longitude,
                                        })) || []}
                                    />
                                </div>
                            )}
                        </div>
                     </div>
                 )}
            </TabsContent>
        </Tabs>

        {/* Person Gallery */}
        <div className="mt-6">
          <PersonGallery />
        </div>
      </main>
    </div>
  );
}
