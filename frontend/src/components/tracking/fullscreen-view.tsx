"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X, Clock, Zap } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface FullscreenViewProps {
  source: any;
  onClose: () => void;
}

export function FullscreenView({ source, onClose }: FullscreenViewProps) {
  const [frameSrc, setFrameSrc] = useState<string>("");
  const [tracks, setTracks] = useState<any[]>([]);
  const [fps, setFps] = useState(0);

  const progress =
    source.total_frames > 0
      ? (source.current_frame / source.total_frames) * 100
      : 0;

  useEffect(() => {
    const handleFrame = (event: CustomEvent) => {
      const data = event.detail;
      if (data.camera_id !== source.camera_id) return;
      if (data.frame_data)
        setFrameSrc(`data:image/jpeg;base64,${data.frame_data}`);
      if (data.tracks) setTracks(data.tracks);
      setFps(Math.round(data.fps) || 0);
    };
    window.addEventListener("video-frame" as any, handleFrame);
    return () => window.removeEventListener("video-frame" as any, handleFrame);
  }, [source.camera_id]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black text-white font-mono">
      <div className="flex flex-1 overflow-hidden">
        {/* Main Video Area */}
        <div className="flex-1 flex flex-col relative border-r border-[#262626]">
          {/* Minimal Header */}
          <div className="absolute top-0 left-0 right-0 p-4 z-10 flex justify-between items-start pointer-events-none">
            <div className="bg-black/50 backdrop-blur px-3 py-1 border border-red-800 pointer-events-auto">
              <h2 className="text-xs uppercase tracking-[0.2em]">
                {source.name}
              </h2>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="pointer-events-auto rounded-none bg-black hover:bg-white hover:text-black border border-red-800"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex-1 bg-[#050505] flex items-center justify-center relative">
            {frameSrc ? (
              <img
                src={frameSrc}
                className="h-full w-full object-contain"
                alt={source.name}
              />
            ) : (
              <div className="text-red-800 uppercase text-xs tracking-widest">
                NO DATA STREAM
              </div>
            )}
            {/* HUD Overlay */}
            <div className="absolute inset-0 pointer-events-none p-8">
              <div className="absolute top-8 left-8 w-8 h-8 border-l border-t border-white/30" />
              <div className="absolute top-8 right-8 w-8 h-8 border-r border-t border-white/30" />
              <div className="absolute bottom-8 left-8 w-8 h-8 border-l border-b border-white/30" />
              <div className="absolute bottom-8 right-8 w-8 h-8 border-r border-b border-white/30" />
            </div>

            {/* Bottom Bar overlay */}
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent">
              {source.source_type === "video_file" && (
                <div className="mb-2">
                  <Progress
                    value={progress}
                    className="h-0.5 rounded-none bg-red-800 [&>div]:bg-white"
                  />
                </div>
              )}
              <div className="flex gap-6 text-[10px] uppercase tracking-widest text-[#888]">
                <span className="flex items-center gap-2">
                  <Zap className="w-3 h-3" /> {fps} FPS
                </span>
                <span className="flex items-center gap-2">
                  <Clock className="w-3 h-3" /> F:{source.current_frame}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar - Tracking Matrix */}
        <div className="w-[300px] bg-black flex flex-col">
          <div className="p-3 border-b border-[#262626] flex justify-between items-center">
            <span className="text-xs uppercase tracking-widest text-[#666]">
              Objects
            </span>
            <span className="text-xs font-bold">{tracks.length}</span>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {tracks.map((track: any) => (
              <Card
                key={track.track_id}
                className="rounded-none bg-black border border-[#262626] hover:border-white/50 transition-colors"
              >
                <CardContent className="p-2">
                  <div className="flex justify-between items-start mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold text-white bg-red-800 px-1">
                        #{track.track_id}
                      </span>
                      <span className="text-[10px] uppercase text-[#888]">
                        {track.class_name}
                      </span>
                    </div>
                    <span className="text-[9px] font-mono text-green-500">
                      {Math.round(track.confidence * 100)}%
                    </span>
                  </div>
                  {track.global_id && (
                    <div className="mt-1 pt-1 border-t border-[#222]">
                      <span className="text-[8px] uppercase tracking-widest text-blue-400">
                        GID: {track.global_id.slice(0, 8)}
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
