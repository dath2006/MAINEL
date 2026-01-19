"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Trash2, RefreshCw, Clock, HardDrive, Film } from "lucide-react";
import { VideoMetadata, videoLibraryApi } from "@/lib/api";

interface VideoLibraryDialogProps {
  open: boolean;
  onClose: () => void;
  onSelectVideo: (video: VideoMetadata) => void;
}

export function VideoLibraryDialog({
  open,
  onClose,
  onSelectVideo,
}: VideoLibraryDialogProps) {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);

  const fetchVideos = async () => {
    try {
      setLoading(true);
      const data = await videoLibraryApi.listVideos();
      setVideos(data);
    } catch (error) {
      console.error("Failed to load videos:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchVideos();
    }
  }, [open]);

  const filteredVideos = videos.filter(
    (v) =>
      v.original_filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.description?.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const handleSelect = () => {
    const video = videos.find((v) => v.id === selectedVideoId);
    if (video) {
      onSelectVideo(video);
      onClose();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="border border-[#262626] bg-black p-0 max-w-5xl max-h-[80vh] overflow-hidden">
        <DialogHeader className="p-4 border-b border-[#262626]">
          <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono flex items-center gap-2">
            <Film className="h-4 w-4" />
            Video_Library :: Select Source
          </DialogTitle>
        </DialogHeader>

        <div className="p-4 space-y-4">
          {/* Search */}
          <div className="flex gap-2">
            <Input
              placeholder="Search videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 rounded-none border-[#262626] bg-[#050505] text-xs h-9 px-3 focus-visible:ring-0 focus-visible:border-white"
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchVideos}
              className="h-9 w-9 rounded-none text-[#666] hover:text-white border border-[#262626]"
            >
              <RefreshCw
                className={`h-4 w-4 ${loading ? "animate-spin" : ""}`}
              />
            </Button>
          </div>

          {/* Video Grid */}
          <ScrollArea className="h-[50vh]">
            {loading ? (
              <div className="flex justify-center items-center h-32">
                <RefreshCw className="h-6 w-6 animate-spin text-[#666]" />
              </div>
            ) : filteredVideos.length === 0 ? (
              <div className="text-center text-[#666] py-12 text-xs uppercase tracking-widest">
                {searchQuery ? "No matching videos" : "No videos in library"}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 pr-2">
                {filteredVideos.map((video) => (
                  <div
                    key={video.id}
                    onClick={() => setSelectedVideoId(video.id)}
                    className={`
                      group cursor-pointer border transition-all
                      ${
                        selectedVideoId === video.id
                          ? "border-white bg-[#111]"
                          : "border-[#262626] hover:border-[#444] bg-black"
                      }
                    `}
                  >
                    {/* Thumbnail Placeholder */}
                    <div className="aspect-video bg-[#0a0a0a] flex items-center justify-center border-b border-[#262626] relative">
                      <Film className="h-8 w-8 text-red-800" />
                      {selectedVideoId === video.id && (
                        <div className="absolute inset-0 border-2 border-white pointer-events-none" />
                      )}
                    </div>

                    {/* Info */}
                    <div className="p-2 space-y-1">
                      <div className="text-[10px] font-mono text-white truncate">
                        {video.original_filename}
                      </div>
                      <div className="flex items-center gap-2 text-[9px] text-[#666] font-mono">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDuration(video.duration)}
                        </span>
                        <span>•</span>
                        <span>
                          {video.width}x{video.height}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[9px] text-[#666] font-mono">
                        <HardDrive className="h-3 w-3" />
                        {formatFileSize(video.file_size)}
                      </div>
                      {video.use_count > 0 && (
                        <div className="text-[9px] text-green-500 font-mono">
                          IN USE: {video.use_count}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>

          {/* Actions */}
          <div className="flex justify-between items-center pt-2 border-t border-[#262626]">
            <div className="text-[10px] text-[#666] font-mono">
              {filteredVideos.length} video(s)
            </div>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                onClick={onClose}
                className="rounded-none text-xs h-9 px-4 border border-[#262626] text-[#888] hover:text-white"
              >
                Cancel
              </Button>
              <Button
                onClick={handleSelect}
                disabled={!selectedVideoId}
                className="rounded-none bg-white text-black hover:bg-[#ccc] text-xs h-9 px-6 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Select_Video
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
