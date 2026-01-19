"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Header } from "@/components/layout/header";
import { VideoFeed } from "@/components/tracking/video-feed";
import { StreamControls } from "@/components/tracking/stream-controls";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
// import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'; // Removed for custom layout control
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import {
  Plus,
  FolderOpen,
  Users,
  RefreshCw,
  Trash2,
  Library,
} from "lucide-react";
import {
  streamsApi,
  tracksApi,
  type StreamSource,
  type PlaybackStatus,
  type SearchResult,
  type VideoMetadata,
} from "@/lib/api";
import { VideoLibraryDialog } from "@/components/tracking/video-library-dialog";
// import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import {
  LayoutGrid,
  RectangleHorizontal,
  Rows,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import dynamic from "next/dynamic";

const MultiTrackMap = dynamic(
  () => import("@/components/tracking/MultiTrackMap"),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full bg-[#111] animate-pulse flex items-center justify-center text-xs text-[#666] uppercase tracking-widest">
        Loading Map Module...
      </div>
    ),
  },
);

const SearchResultsPanel = dynamic(
  () => import("@/components/tracking/SearchResultsPanel"),
  {
    ssr: false,
  },
);

// Inline Person Gallery for this layout
interface PersonEntry {
  global_id: string;
  last_camera_id: number;
  last_seen: string;
  appearance_count: number;
  thumbnail: string | null;
}

interface CaptureEntry {
  image_b64: string;
  quality_score: number;
  pose: string;
  sharpness: number;
  timestamp: string | null;
}

export default function TrackingPage() {
  const [sources, setSources] = useState<StreamSource[]>([]);
  const [status, setStatus] = useState<PlaybackStatus>({
    state: "stopped",
    source_count: 0,
    target_fps: 30,
    queue_size: 0,
    sources: [],
  });
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newSource, setNewSource] = useState({
    camera_id: 1,
    source_path: "",
    source_type: "video_file" as "video_file" | "webcam" | "rtsp",
    name: "",
    file: null as File | null,
    latitude: 0.0,
    longitude: 0.0,
  });

  const [activeTab, setActiveTab] = useState("feeds");
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(
    null,
  );
  const [focusedSource, setFocusedSource] = useState<StreamSource | null>(null);
  const [isLibraryDialogOpen, setIsLibraryDialogOpen] = useState(false);
  const [selectedLibraryVideo, setSelectedLibraryVideo] =
    useState<VideoMetadata | null>(null);

  const [layoutMode, setLayoutMode] = useState<"auto" | "1" | "2">("auto");

  // Custom Overlay Panel State
  const [panelHeight, setPanelHeight] = useState(200);
  const [isResizing, setIsResizing] = useState(false);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newHeight = window.innerHeight - e.clientY;
      // Clamp between 30px (header) and 80% screen
      const clamped = Math.min(
        Math.max(newHeight, 30),
        window.innerHeight * 0.8,
      );
      setPanelHeight(clamped);
      if (clamped > 40) setIsPanelCollapsed(false);
    };

    const handleMouseUp = () => setIsResizing(false);

    if (isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Gallery state
  const [persons, setPersons] = useState<PersonEntry[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [galleryError, setGalleryError] = useState<string | null>(null);
  const apiUrl =
    (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") +
    "/api/v1/streams";

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

  // Popup State
  const [selectedPerson, setSelectedPerson] = useState<string | null>(null);
  const [captures, setCaptures] = useState<CaptureEntry[]>([]);
  const [loadingCaptures, setLoadingCaptures] = useState(false);

  const fetchCaptures = async (globalId: string) => {
    try {
      setLoadingCaptures(true);
      const response = await fetch(`${apiUrl}/gallery/${globalId}/captures`);
      if (!response.ok) throw new Error("Failed to fetch captures");
      const data = await response.json();
      setCaptures(data.captures || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingCaptures(false);
    }
  };

  const handlePersonClick = (globalId: string) => {
    setSelectedPerson(globalId);
    fetchCaptures(globalId);
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
      console.error("Failed to fetch stream data:", error);
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
  const [wsStatus, setWsStatus] = useState<
    "connecting" | "connected" | "disconnected"
  >("connecting");
  const [frameCount, setFrameCount] = useState(0);
  useEffect(() => {
    // Import getWsUrl dynamically or use logic? Using helper from api.ts
    // Since getWsUrl uses the same env var logic, it's safer.
    // However, getWsUrl is not imported yet. I should add import first or use direct logic.
    // Let's use direct logic for now to match the style, or just string replace if I import it.
    // Easier to just duplicate the logic briefly or do a multi-replace to add import.
    // I will stick to replacement for now and assume I can add import later or just use inline logic.

    // Actually, let's use the helper. I need to add import to top of file first.
    // But since I'm doing single replacement here, I'll just use the env var logic directly for WS.
    const wsBase = (
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    ).replace(/^http/, "ws");
    const ws = new WebSocket(`${wsBase}/api/v1/ws/tracks`);
    ws.onopen = () => setWsStatus("connected");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
          setFrameCount((c) => c + 1);
          window.dispatchEvent(
            new CustomEvent("video-frame", { detail: data }),
          );
        }
      } catch (e) {
        console.error("WS Error", e);
      }
    };
    ws.onclose = () => setWsStatus("disconnected");
    return () => ws.close();
  }, []);

  const [addError, setAddError] = useState<string | null>(null);

  const handleAddSource = async () => {
    setAddError(null);
    try {
      if (selectedLibraryVideo) {
        // Create source from library video
        await streamsApi.createSourceFromLibrary({
          camera_id: newSource.camera_id,
          video_id: selectedLibraryVideo.id,
          name: newSource.name || selectedLibraryVideo.original_filename,
          latitude: newSource.latitude,
          longitude: newSource.longitude,
        });
        setSelectedLibraryVideo(null);
      } else if (newSource.source_type === "video_file" && newSource.file) {
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
      setAddError(
        error instanceof Error ? error.message : "Failed to add source",
      );
    }
  };

  const handleLibraryVideoSelect = (video: VideoMetadata) => {
    setSelectedLibraryVideo(video);
    setNewSource((prev) => ({
      ...prev,
      name: video.original_filename,
      source_type: "video_file",
    }));
  };

  const handleDeleteSource = async (sourceId: number) => {
    try {
      await streamsApi.removeSource(sourceId);
      fetchData();
    } catch (error) {
      console.error(error);
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

  const handleSearch = async () => {
    if (!searchFile) return;
    setIsSearching(true);
    try {
      const results = await tracksApi.searchByImage(searchFile);
      setSearchResults(results);
      if (results.length > 0) {
        setSelectedResult(results[0]);
        setActiveTab("map");
      }
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setIsSearching(false);
    }
  };

  const formatTimeAgo = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return "OLD";
  };

  return (
    <div className="flex flex-col h-[calc(100vh-theme(spacing.14))] bg-black font-mono overflow-hidden">
      <Header title="LIVE_TRACK" />

      <div className="flex-1 relative border-t border-[#262626] flex flex-col overflow-hidden">
        {/* Main Content Area (Video/Map) - Takes full space behind overlay */}
        <div className="absolute inset-0 flex flex-col overflow-hidden pb-8">
          {/* Top Toolbar / Diagnostics */}
          <div className="h-10 border-b border-[#262626] flex items-center justify-between bg-[#050505] px-2 text-[10px] tracking-widest text-[#666] shrink-0">
            <div className="flex gap-4">
              <span className="flex items-center gap-1">
                WS:
                <span
                  className={
                    wsStatus === "connected" ? "text-green-500" : "text-red-500"
                  }
                >
                  ●
                </span>
              </span>
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
                className="h-6 gap-2 text-[10px] rounded-none border border-red-800 bg-black text-[#888] hover:bg-[#111] hover:text-white uppercase px-3"
                onClick={() => fileInputRef.current?.click()}
              >
                <FolderOpen className="w-3 h-3" />
                {searchFile
                  ? searchFile.name.length > 15
                    ? searchFile.name.substring(0, 12) + "..."
                    : searchFile.name
                  : "SELECT_IMAGE"}
              </Button>

              <div className="w-px h-4 bg-[#262626]" />

              <div className="flex items-center border border-red-800 rounded-none overflow-hidden">
                <Button
                  size="icon"
                  className={`h-6 w-6 rounded-none ${layoutMode === "auto" ? "bg-white text-black" : "bg-black text-[#888] hover:text-white"}`}
                  onClick={() => setLayoutMode("auto")}
                  title="Auto Grid"
                >
                  <LayoutGrid className="w-3 h-3" />
                </Button>
                <Button
                  size="icon"
                  className={`h-6 w-6 rounded-none ${layoutMode === "2" ? "bg-white text-black" : "bg-black text-[#888] hover:text-white"}`}
                  onClick={() => setLayoutMode("2")}
                  title="2 per row"
                >
                  <RectangleHorizontal className="w-3 h-3" />
                </Button>
                <Button
                  size="icon"
                  className={`h-6 w-6 rounded-none ${layoutMode === "1" ? "bg-white text-black" : "bg-black text-[#888] hover:text-white"}`}
                  onClick={() => setLayoutMode("1")}
                  title="1 per row"
                >
                  <Rows className="w-3 h-3" />
                </Button>
              </div>

              <div className="w-px h-4 bg-[#262626]" />

              <Button
                size="sm"
                className="h-6 text-[10px] rounded-none border border-red-800 bg-[#111] hover:bg-white hover:text-black uppercase px-4"
                onClick={handleSearch}
                disabled={!searchFile || isSearching}
              >
                {isSearching ? "SCANNING..." : "START_SEARCH"}
              </Button>
            </div>
          </div>

          {/* Viewport Content */}
          <div className="flex-1 overflow-hidden relative bg-[#000]">
            {/* Custom Tab Header - Absolute positioned on top left */}
            <div className="absolute top-4 left-4 z-[9999]">
              <div className="flex bg-black border border-[#262626] h-8 shadow-xl">
                <button
                  onClick={() => setActiveTab("feeds")}
                  className={`h-full px-4 text-[10px] uppercase tracking-wider border-r border-[#262626] transition-colors ${activeTab === "feeds" ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                >
                  Video
                </button>
                <button
                  onClick={() => setActiveTab("map")}
                  className={`h-full px-4 text-[10px] uppercase tracking-wider transition-colors ${activeTab === "map" ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                >
                  Global_Map
                </button>
              </div>
            </div>

            {/* VIDEO LAYER - Always mounted, hidden via visibility/z-index */}
            <div
              className={`absolute inset-0 flex flex-col ${activeTab === "feeds" ? "z-10 opacity-100 pointer-events-auto" : "z-0 opacity-0 pointer-events-none"}`}
            >
              <div className="absolute top-4 right-4 z-[9999] w-fit">
                <StreamControls
                  onPlay={handlePlay}
                  onPause={handlePause}
                  onStop={handleStop}
                  state={status.state as any}
                  sourceCount={sources.length}
                />
              </div>

              {/* Scrollable Video Grid Area covering everything */}
              <div className="absolute inset-0 pt-0 pb-0 flex flex-col overflow-y-auto bg-black/40">
                {/* Spacer for top controls to avoid overlap if needed, or rely on padding */}
                <div className="min-h-[60px] w-full shrink-0" />{" "}
                {/* Top padding for tabs/controls */}
                <div
                  className={`p-4 grid gap-2 content-start ${
                    layoutMode === "1"
                      ? "grid-cols-1"
                      : layoutMode === "2"
                        ? "grid-cols-2"
                        : "grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                  }`}
                >
                  {sources.map((source) => (
                    <VideoFeed
                      key={source.id}
                      cameraId={source.camera_id}
                      sourceName={source.name}
                      isActive={
                        source.is_active &&
                        (status.state === "playing" ||
                          status.state === "paused")
                      }
                      onDelete={() => handleDeleteSource(source.id)}
                      onMaximize={() => setFocusedSource(source)}
                    />
                  ))}
                  <div
                    className="aspect-video min-h-[150px] border border-[#262626] bg-black flex flex-col items-center justify-center cursor-pointer hover:bg-[#111] transition-colors group"
                    onClick={() => setIsAddDialogOpen(true)}
                  >
                    <Plus className="h-8 w-8 text-red-800 group-hover:text-white transition-colors" />
                    <span className="mt-2 text-[10px] uppercase tracking-widest text-[#444] group-hover:text-white">
                      Add_Source
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* MAP LAYER - Always mounted, hidden via visibility/z-index */}
            <div
              className={`absolute inset-0 ${activeTab === "map" ? "z-10 opacity-100 pointer-events-auto" : "z-0 opacity-0 pointer-events-none"}`}
            >
              <MultiTrackMap
                sources={sources}
                activeTracks={
                  selectedResult
                    ? [
                        {
                          globalId: selectedResult.track.id,
                          pathPoints: selectedResult.path_points || [],
                        },
                      ]
                    : []
                }
                selectedTrackId={selectedResult?.track.id}
                onMapClick={(lat, lng) => {
                  const nextId =
                    sources.length > 0
                      ? Math.max(...sources.map((s) => s.camera_id)) + 1
                      : 1;
                  setNewSource((prev) => ({
                    ...prev,
                    latitude: lat,
                    longitude: lng,
                    camera_id: nextId,
                  }));
                  setIsAddDialogOpen(true);
                }}
              />
            </div>

            {/* SEARCH OVERLAY - Global, appearing over everything */}
            {searchResults.length > 0 && (
              <SearchResultsPanel
                results={searchResults}
                selectedResult={selectedResult}
                onSelect={setSelectedResult}
                onClose={() => {
                  setSearchResults([]);
                  setSelectedResult(null);
                }}
              />
            )}
          </div>
        </div>

        {/* Overlay Bottom Panel */}
        <div
          style={{
            height: isPanelCollapsed ? "32px" : `${panelHeight}px`,
            transition: isResizing ? "none" : "height 0.2s ease",
          }}
          className="absolute bottom-0 left-0 right-0 bg-black/95 z-50 border-t border-[#262626] flex flex-col shadow-[0_-5px_20px_rgba(0,0,0,0.5)]"
        >
          {/* Resize Handle / Header */}
          <div
            className="h-1 bg-transparent hover:bg-red-500/50 cursor-ns-resize w-full absolute top-0 left-0 right-0 z-50 -translate-y-1/2"
            onMouseDown={(e) => {
              e.preventDefault();
              setIsResizing(true);
            }}
          />

          <div
            className="flex items-center justify-between px-3 py-1 border-b border-[#1a1a1a] h-8 bg-[#050505] shrink-0 select-none cursor-ns-resize"
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) setIsResizing(true);
            }}
          >
            <div
              className="flex items-center gap-2 cursor-pointer"
              onClick={() => setIsPanelCollapsed(!isPanelCollapsed)}
            >
              {isPanelCollapsed ? (
                <ChevronUp className="w-3 h-3 text-[#666]" />
              ) : (
                <ChevronDown className="w-3 h-3 text-[#666]" />
              )}
              <span className="text-[10px] uppercase tracking-[0.2em] font-medium text-[#888]">
                IDENTITY_LOG ({persons.length})
              </span>
            </div>

            <div className="flex gap-1 z-10">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 rounded-none text-[#666] hover:text-white"
                onClick={fetchGallery}
              >
                <RefreshCw
                  className={`h-3 w-3 ${galleryLoading ? "animate-spin" : ""}`}
                />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 rounded-none text-[#666] hover:text-red-500"
                onClick={clearGallery}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 bg-[#050505] flex flex-col overflow-hidden">
            <ScrollArea className="flex-1 w-full">
              <div className="flex gap-2 p-2 h-full">
                {galleryError && (
                  <div className="text-[10px] text-red-500 font-mono self-center px-4">
                    {galleryError}
                  </div>
                )}
                {persons.length === 0 && !galleryError ? (
                  <div className="flex items-center justify-center w-full text-red-800">
                    <p className="text-[10px] uppercase tracking-widest">
                      Awaiting Detection...
                    </p>
                  </div>
                ) : (
                  persons.map((person) => (
                    <div
                      key={person.global_id}
                      className="group relative bg-black border border-[#262626] cursor-pointer hover:border-white transition-colors shrink-0 w-28"
                      onClick={() => handlePersonClick(person.global_id)}
                    >
                      <div className="aspect-[3/4] overflow-hidden transition-all bg-[#111]">
                        {person.thumbnail ? (
                          <img
                            src={`data:image/jpeg;base64,${person.thumbnail}`}
                            alt={person.global_id}
                            className="w-full h-full object-cover group-hover:opacity-100 transition-opacity"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <Users className="h-6 w-6 text-red-800" />
                          </div>
                        )}
                      </div>
                      <div className="p-1.5 border-t border-[#262626] bg-[#050505]">
                        <div className="flex justify-between items-center">
                          <span className="text-[8px] font-bold text-white bg-[#222] px-1 font-mono">
                            {person.global_id.slice(0, 6)}
                          </span>
                          <span className="text-[8px] font-mono text-[#666]">
                            {formatTimeAgo(person.last_seen)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </div>
        </div>
      </div>

      {/* Add Source Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="border border-[#262626] bg-black p-0 gap-0 sm:max-w-[425px]">
          <DialogHeader className="p-4 border-b border-[#262626]">
            <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono">
              Input_Configuration
            </DialogTitle>
          </DialogHeader>
          <div className="p-4 space-y-4">
            <div className="w-full">
              <div className="w-full grid grid-cols-4 h-8 bg-[#111] border border-red-800">
                <button
                  className={`text-[10px] uppercase transition-colors ${newSource.source_type === "video_file" ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                  onClick={() =>
                    setNewSource((s) => ({ ...s, source_type: "video_file" }))
                  }
                >
                  File
                </button>
                <button
                  className={`text-[10px] uppercase transition-colors ${newSource.source_type === "webcam" ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                  onClick={() =>
                    setNewSource((s) => ({ ...s, source_type: "webcam" }))
                  }
                >
                  Cam
                </button>
                <button
                  className={`text-[10px] uppercase transition-colors ${newSource.source_type === "rtsp" && !selectedLibraryVideo ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                  onClick={() => {
                    setNewSource((s) => ({ ...s, source_type: "rtsp" }));
                    setSelectedLibraryVideo(null);
                  }}
                >
                  RTSP
                </button>
                <button
                  className={`text-[10px] uppercase transition-colors ${selectedLibraryVideo ? "bg-white text-black" : "text-[#888] hover:text-white"}`}
                  onClick={() => setIsLibraryDialogOpen(true)}
                >
                  Library
                </button>
              </div>

              <div className="mt-4 space-y-4">
                {selectedLibraryVideo ? (
                  <div className="p-3 bg-[#111] border border-red-800 space-y-1">
                    <div className="text-xs font-mono text-white">
                      {selectedLibraryVideo.original_filename}
                    </div>
                    <div className="text-[10px] text-[#666] font-mono">
                      {selectedLibraryVideo.width}x{selectedLibraryVideo.height}{" "}
                      • {Math.floor(selectedLibraryVideo.duration)}s •{" "}
                      {(selectedLibraryVideo.file_size / (1024 * 1024)).toFixed(
                        1,
                      )}{" "}
                      MB
                    </div>
                    <button
                      className="text-[10px] text-red-500 hover:text-red-400 mt-2"
                      onClick={() => setSelectedLibraryVideo(null)}
                    >
                      Clear Selection
                    </button>
                  </div>
                ) : newSource.source_type === "video_file" ? (
                  <Input
                    type="file"
                    className="rounded-none border-red-800 bg-[#050505] text-xs h-9"
                    onChange={(e) =>
                      setNewSource({
                        ...newSource,
                        source_path: "",
                        file: e.target.files?.[0] || null,
                      })
                    }
                  />
                ) : (
                  <Input
                    placeholder={
                      newSource.source_type === "webcam"
                        ? "Device Index (0)"
                        : "RTSP://..."
                    }
                    className="rounded-none border-red-800 bg-[#050505] text-xs h-9"
                    value={newSource.source_path || ""}
                    onChange={(e) =>
                      setNewSource({
                        ...newSource,
                        source_path: e.target.value,
                      })
                    }
                  />
                )}
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">
                      CAM_ID
                    </label>
                    <Input
                      type="number"
                      className="rounded-none border-red-800 bg-[#050505] text-xs h-8"
                      value={newSource.camera_id}
                      onChange={(e) =>
                        setNewSource({
                          ...newSource,
                          camera_id: parseInt(e.target.value) || 1,
                        })
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">
                      Label
                    </label>
                    <Input
                      className="rounded-none border-red-800 bg-[#050505] text-xs h-8"
                      value={newSource.name}
                      onChange={(e) =>
                        setNewSource({ ...newSource, name: e.target.value })
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">
                      LAT
                    </label>
                    <Input
                      type="number"
                      className="rounded-none border-red-800 bg-[#050505] text-xs h-8"
                      value={newSource.latitude}
                      onChange={(e) =>
                        setNewSource({
                          ...newSource,
                          latitude: parseFloat(e.target.value) || 0,
                        })
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] uppercase tracking-wider text-[#666]">
                      LNG
                    </label>
                    <Input
                      type="number"
                      className="rounded-none border-red-800 bg-[#050505] text-xs h-8"
                      value={newSource.longitude}
                      onChange={(e) =>
                        setNewSource({
                          ...newSource,
                          longitude: parseFloat(e.target.value) || 0,
                        })
                      }
                    />
                  </div>
                </div>
              </div>
            </div>
            {addError && <p className="text-red-500 text-[10px]">{addError}</p>}
            <Button
              onClick={handleAddSource}
              className="w-full bg-white text-black hover:bg-[#ccc] rounded-none uppercase tracking-widest text-xs h-9"
            >
              Initialize_Source
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Identity Details Popup */}
      <Dialog
        open={!!selectedPerson}
        onOpenChange={(open) => !open && setSelectedPerson(null)}
      >
        <DialogContent className="border border-[#262626] bg-black p-0 max-w-4xl max-h-[80vh] overflow-hidden">
          <DialogHeader className="p-4 border-b border-[#262626]">
            <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono">
              Archive :: {selectedPerson?.slice(0, 8)}
            </DialogTitle>
          </DialogHeader>
          <div className="p-4 overflow-y-auto max-h-[70vh] bg-[#050505]">
            {loadingCaptures ? (
              <div className="flex justify-center p-8">
                <RefreshCw className="h-6 w-6 animate-spin text-red-800" />
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-2">
                {captures.map((cap, idx) => (
                  <div key={idx} className="bg-black border border-[#262626]">
                    <div className="aspect-[3/4] relative">
                      <img
                        src={`data:image/jpeg;base64,${cap.image_b64}`}
                        className="w-full h-full object-contain opacity-80"
                        alt={`Capture ${idx}`}
                      />
                      <span className="absolute top-1 right-1 bg-black/80 text-white text-[9px] px-1 font-mono">
                        Q:{cap.quality_score.toFixed(0)}
                      </span>
                    </div>
                    <div className="p-1 border-t border-[#262626] text-center">
                      <span className="text-[9px] text-[#555] font-mono">
                        {cap.pose}
                      </span>
                    </div>
                  </div>
                ))}
                {captures.length === 0 && (
                  <div className="col-span-full text-center text-[#444] text-xs py-8">
                    No high-quality captures found.
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Video Library Dialog */}
      <VideoLibraryDialog
        open={isLibraryDialogOpen}
        onClose={() => setIsLibraryDialogOpen(false)}
        onSelectVideo={handleLibraryVideoSelect}
      />
    </div>
  );
}
