'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { Header } from '@/components/layout/header';
import { VideoFeed } from '@/components/tracking/video-feed';
import { StreamControls } from '@/components/tracking/stream-controls';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Plus, FolderOpen, Users, RefreshCw, Trash2 } from 'lucide-react';
import { streamsApi, tracksApi, type StreamSource, type PlaybackStatus, type SearchResult } from '@/lib/api';
import dynamic from 'next/dynamic';

const MultiTrackMap = dynamic(() => import('@/components/tracking/MultiTrackMap'), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#111] animate-pulse flex items-center justify-center text-xs text-[#666] uppercase tracking-widest">Loading Map Module...</div>
});

const SearchResultsPanel = dynamic(() => import('@/components/tracking/SearchResultsPanel'), {
  ssr: false
});

// Inline Person Gallery for this layout
interface PersonEntry {
  global_id: string;
  last_camera_id: number;
  last_seen: string;
  appearance_count: number;
  thumbnail: string | null;
}

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
    file: null as File | null,
    latitude: 0.0,
    longitude: 0.0,
  });

  const [activeTab, setActiveTab] = useState('feeds');
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [focusedSource, setFocusedSource] = useState<StreamSource | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Gallery state
  const [persons, setPersons] = useState<PersonEntry[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [galleryError, setGalleryError] = useState<string | null>(null);
  const apiUrl = "http://localhost:8000/api/v1/streams";

  const fetchGallery = async () => {
    try {
      setGalleryLoading(true);
      const response = await fetch(`${apiUrl}/gallery`);
      if (!response.ok) throw new Error("Failed to fetch gallery");
      const data = await response.json();
      setPersons(data.persons || []);
      setGalleryError(null);
    } catch (err) {
      setGalleryError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setGalleryLoading(false);
    }
  };

  const clearGallery = async () => {
    try {
      const response = await fetch(`${apiUrl}/gallery`, { method: "DELETE" });
      if (!response.ok) throw new Error("Failed to clear gallery");
      setPersons([]);
    } catch (err) {
      setGalleryError(err instanceof Error ? err.message : "Unknown error");
    }
  };

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
    fetchGallery();
    const interval = setInterval(fetchData, 2000);
    const galleryInterval = setInterval(fetchGallery, 5000);
    return () => {
      clearInterval(interval);
      clearInterval(galleryInterval);
    };
  }, [fetchData]);

  // WebSocket state
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [frameCount, setFrameCount] = useState(0);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/tracks');
    ws.onopen = () => setWsStatus('connected');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'frame') {
          setFrameCount((c) => c + 1);
          window.dispatchEvent(new CustomEvent('video-frame', { detail: data }));
        }
      } catch (e) { console.error('WS Error', e); }
    };
    ws.onclose = () => setWsStatus('disconnected');
    return () => ws.close();
  }, []);

  const [addError, setAddError] = useState<string | null>(null);

  const handleAddSource = async () => {
    setAddError(null);
    try {
      if (newSource.source_type === 'video_file' && newSource.file) {
        await streamsApi.uploadVideo({
          camera_id: newSource.camera_id,
          name: newSource.name,
          file: newSource.file,
          latitude: newSource.latitude,
          longitude: newSource.longitude,
        });
      } else {
        await streamsApi.addSource({
          camera_id: newSource.camera_id,
          source_type: newSource.source_type,
          source_path: newSource.source_path,
          name: newSource.name || undefined,
          latitude: newSource.latitude,
          longitude: newSource.longitude,
        });
      }
      setIsAddDialogOpen(false);
      fetchData();
    } catch (error) {
      setAddError(error instanceof Error ? error.message : 'Failed to add source');
    }
  };

  const handleDeleteSource = async (sourceId: number) => {
    try { await streamsApi.removeSource(sourceId); fetchData(); } catch (error) { console.error(error); }
  };
  const handlePlay = async () => { await streamsApi.play(); fetchData(); };
  const handlePause = async () => { await streamsApi.pause(); fetchData(); };
  const handleStop = async () => { await streamsApi.stop(); fetchData(); };

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
    } catch (error) { console.error("Search failed", error); } finally { setIsSearching(false); }
  };

  const formatTimeAgo = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return 'OLD';
  };

  return (
    <div className="flex flex-col h-[calc(100vh-theme(spacing.14))] bg-black font-mono overflow-hidden">
      <Header title="LIVE_TRACK" />

      {/* Main Content Area (Video/Map) */}
      <div className="flex-1 flex flex-col overflow-hidden border-t border-[#262626]">

        {/* Top Toolbar / Diagnostics */}
        <div className="h-10 border-b border-[#262626] flex items-center justify-between bg-[#050505] px-2 text-[10px] tracking-widest text-[#666] shrink-0">
          <div className="flex gap-4">
            <span className="flex items-center gap-1">WS:<span className={wsStatus === 'connected' ? 'text-green-500' : 'text-red-500'}>●</span></span>
            <span>FRAMES: {frameCount}</span>
            <span>STATE: {status.state.toUpperCase()}</span>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              className="hidden"
              onChange={(e) => setSearchFile(e.target.files?.[0] || null)}
            />
            <Button
              size="sm"
              className="h-6 gap-2 text-[10px] rounded-none border border-[#333] bg-black text-[#888] hover:bg-[#111] hover:text-white uppercase px-3"
              onClick={() => fileInputRef.current?.click()}
            >
              <FolderOpen className="w-3 h-3" />
              {searchFile ? (searchFile.name.length > 15 ? searchFile.name.substring(0, 12) + '...' : searchFile.name) : 'SELECT_IMAGE'}
            </Button>

            <div className="w-px h-4 bg-[#262626]" />

            <Button
              size="sm"
              className="h-6 text-[10px] rounded-none border border-[#333] bg-[#111] hover:bg-white hover:text-black uppercase px-4"
              onClick={handleSearch}
              disabled={!searchFile || isSearching}
            >
              {isSearching ? 'SCANNING...' : 'START_SEARCH'}
            </Button>
          </div>
        </div>

        {/* Viewport Content */}
        <div className="flex-1 overflow-hidden relative bg-[#000]">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full w-full flex flex-col">
            <div className="absolute top-4 left-4 z-[9999]">
              <TabsList className="bg-black border border-[#262626] p-0 h-8 rounded-none shadow-xl">
                <TabsTrigger value="feeds" className="h-8 rounded-none border-r border-[#262626] data-[state=active]:bg-white data-[state=active]:text-black text-[10px] uppercase tracking-wider w-24 transition-colors">Video</TabsTrigger>
                <TabsTrigger value="map" className="h-8 rounded-none data-[state=active]:bg-white data-[state=active]:text-black text-[10px] uppercase tracking-wider w-24 transition-colors">Global_Map</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="feeds" className="h-full m-0 p-0 flex flex-col relative data-[state=active]:flex">
              <div className="absolute top-4 right-4 z-[9999] w-fit">
                <StreamControls
                  onPlay={handlePlay}
                  onPause={handlePause}
                  onStop={handleStop}
                  state={status.state as any}
                  sourceCount={sources.length}
                />
              </div>

              <div className="flex-1 p-4 grid gap-px bg-[#111] border-[#222] grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 overflow-y-auto content-start">
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
                <div className="aspect-video min-h-[150px] border border-[#262626] bg-black flex flex-col items-center justify-center cursor-pointer hover:bg-[#111] transition-colors group" onClick={() => setIsAddDialogOpen(true)}>
                  <Plus className="h-8 w-8 text-[#333] group-hover:text-white transition-colors" />
                  <span className="mt-2 text-[10px] uppercase tracking-widest text-[#444] group-hover:text-white">Add_Source</span>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="map" className="h-full m-0 p-0 relative data-[state=active]:block">
              <MultiTrackMap
                sources={sources}
                activeTracks={selectedResult ? [{
                  globalId: selectedResult.track.id,
                  pathPoints: selectedResult.path_points || [],
                }] : []}
                selectedTrackId={selectedResult?.track.id}
                onMapClick={(lat, lng) => {
                  const nextId = sources.length > 0 ? Math.max(...sources.map(s => s.camera_id)) + 1 : 1;
                  setNewSource(prev => ({ ...prev, latitude: lat, longitude: lng, camera_id: nextId }));
                  setIsAddDialogOpen(true);
                }}
              />
              {searchResults.length > 0 && (
                <div className="absolute top-4 right-4 z-[9999] w-80 max-h-[calc(100%-2rem)] flex flex-col">
                  <SearchResultsPanel
                    results={searchResults}
                    selectedResult={selectedResult}
                    onSelect={setSelectedResult}
                    onClose={() => { setSearchResults([]); setSelectedResult(null); }}
                  />
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Bottom: Person Gallery (Fixed Height, Horizontal Scroll) */}
      <div className="h-48 shrink-0 border-t border-[#262626] bg-[#050505] flex flex-col">
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#1a1a1a]">
          <span className="text-[10px] uppercase tracking-[0.2em] font-medium text-[#888] flex items-center gap-2">
            <Users className="h-3 w-3" />
            IDENTITY_LOG ({persons.length})
          </span>
          <div className="flex gap-1">
            <Button variant="ghost" size="icon" className="h-6 w-6 rounded-none text-[#666] hover:text-white" onClick={fetchGallery}>
              <RefreshCw className={`h-3 w-3 ${galleryLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6 rounded-none text-[#666] hover:text-red-500" onClick={clearGallery}>
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>

        <ScrollArea className="flex-1 w-full">
          <div className="flex gap-2 p-2 h-full">
            {galleryError && <div className="text-[10px] text-red-500 font-mono self-center px-4">{galleryError}</div>}
            {persons.length === 0 && !galleryError ? (
              <div className="flex items-center justify-center w-full text-[#333]">
                <p className="text-[10px] uppercase tracking-widest">Awaiting Detection...</p>
              </div>
            ) : (
              persons.map((person) => (
                <div
                  key={person.global_id}
                  className="group relative bg-black border border-[#262626] cursor-pointer hover:border-white transition-colors shrink-0 w-28"
                >
                  <div className="aspect-[3/4] overflow-hidden grayscale group-hover:grayscale-0 transition-all bg-[#111]">
                    {person.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${person.thumbnail}`}
                        alt={person.global_id}
                        className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Users className="h-6 w-6 text-[#333]" />
                      </div>
                    )}
                  </div>
                  <div className="p-1.5 border-t border-[#262626] bg-[#050505]">
                    <div className="flex justify-between items-center">
                      <span className="text-[8px] font-bold text-white bg-[#222] px-1 font-mono">{person.global_id.slice(0, 6)}</span>
                      <span className="text-[8px] font-mono text-[#666]">{formatTimeAgo(person.last_seen)}</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </div>

      {/* Add Source Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="border border-[#262626] bg-black p-0 gap-0 sm:max-w-[425px]">
          <DialogHeader className="p-4 border-b border-[#262626]">
            <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono">Input_Configuration</DialogTitle>
          </DialogHeader>
          <div className="p-4 space-y-4">
            <Tabs defaultValue="file" className="w-full">
              <TabsList className="w-full grid grid-cols-3 h-8 bg-[#111] rounded-none p-0">
                <TabsTrigger value="file" className="text-[10px] uppercase rounded-none data-[state=active]:bg-white data-[state=active]:text-black" onClick={() => setNewSource(s => ({ ...s, source_type: 'video_file' }))}>File</TabsTrigger>
                <TabsTrigger value="webcam" className="text-[10px] uppercase rounded-none data-[state=active]:bg-white data-[state=active]:text-black" onClick={() => setNewSource(s => ({ ...s, source_type: 'webcam' }))}>Cam</TabsTrigger>
                <TabsTrigger value="rtsp" className="text-[10px] uppercase rounded-none data-[state=active]:bg-white data-[state=active]:text-black" onClick={() => setNewSource(s => ({ ...s, source_type: 'rtsp' }))}>RTSP</TabsTrigger>
              </TabsList>

              <div className="mt-4 space-y-4">
                {newSource.source_type === 'video_file' && (
                  <Input type="file" className="rounded-none border-[#333] bg-[#050505] text-xs h-9" onChange={(e) => setNewSource({ ...newSource, source_path: '', file: e.target.files?.[0] || null })} />
                )}
                {newSource.source_type !== 'video_file' && (
                  <Input
                    placeholder={newSource.source_type === 'webcam' ? "Device Index (0)" : "RTSP://..."}
                    className="rounded-none border-[#333] bg-[#050505] text-xs h-9"
                    value={newSource.source_path}
                    onChange={(e) => setNewSource({ ...newSource, source_path: e.target.value })}
                  />
                )}
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">CAM_ID</label>
                    <Input
                      type="number"
                      className="rounded-none border-[#333] bg-[#050505] text-xs h-8"
                      value={newSource.camera_id}
                      onChange={(e) => setNewSource({ ...newSource, camera_id: parseInt(e.target.value) || 1 })}
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">Label</label>
                    <Input
                      className="rounded-none border-[#333] bg-[#050505] text-xs h-8"
                      value={newSource.name}
                      onChange={(e) => setNewSource({ ...newSource, name: e.target.value })}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">LAT</label>
                    <Input
                      type="number"
                      className="rounded-none border-[#333] bg-[#050505] text-xs h-8"
                      value={newSource.latitude}
                      onChange={(e) => setNewSource({ ...newSource, latitude: parseFloat(e.target.value) || 0 })}
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">LNG</label>
                    <Input
                      type="number"
                      className="rounded-none border-[#333] bg-[#050505] text-xs h-8"
                      value={newSource.longitude}
                      onChange={(e) => setNewSource({ ...newSource, longitude: parseFloat(e.target.value) || 0 })}
                    />
                  </div>
                </div>
              </div>
            </Tabs>
            {addError && <p className="text-red-500 text-[10px]">{addError}</p>}
            <Button onClick={handleAddSource} className="w-full bg-white text-black hover:bg-[#ccc] rounded-none uppercase tracking-widest text-xs h-9">
              Initialize_Source
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
