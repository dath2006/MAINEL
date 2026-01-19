"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Trash2,
  Upload,
  RefreshCw,
  Film,
  Clock,
  HardDrive,
  AlertCircle,
} from "lucide-react";
import { VideoMetadata, videoLibraryApi } from "@/lib/api";

export function VideoUploadManager() {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState<VideoMetadata | null>(
    null,
  );

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadDescription, setUploadDescription] = useState("");
  const [uploadTags, setUploadTags] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    fetchVideos();
  }, []);

  const handleUpload = async () => {
    if (!uploadFile) return;

    try {
      setUploading(true);
      await videoLibraryApi.uploadToLibrary(
        uploadFile,
        uploadDescription,
        uploadTags,
      );
      setUploadDialogOpen(false);
      setUploadFile(null);
      setUploadDescription("");
      setUploadTags("");
      await fetchVideos();
    } catch (error) {
      console.error("Upload failed:", error);
      alert(
        `Upload failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async () => {
    if (!videoToDelete) return;

    try {
      await videoLibraryApi.deleteVideo(videoToDelete.id);
      setDeleteDialogOpen(false);
      setVideoToDelete(null);
      await fetchVideos();
    } catch (error) {
      console.error("Delete failed:", error);
      alert(
        `Delete failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  return (
    <Card className="border-[#262626] bg-black">
      <CardHeader className="border-b border-[#262626] pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-[0.2em] font-mono flex items-center gap-2">
            <Film className="h-4 w-4" />
            Video_Library :: Management
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchVideos}
              className="h-7 w-7 rounded-none text-[#666] hover:text-white"
            >
              <RefreshCw
                className={`h-3 w-3 ${loading ? "animate-spin" : ""}`}
              />
            </Button>
            <Button
              onClick={() => setUploadDialogOpen(true)}
              className="rounded-none bg-white text-black hover:bg-[#ccc] text-[10px] h-7 px-4 flex items-center gap-2"
            >
              <Upload className="h-3 w-3" />
              Upload
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <ScrollArea className="h-[400px]">
          {loading ? (
            <div className="flex justify-center items-center h-32">
              <RefreshCw className="h-6 w-6 animate-spin text-[#666]" />
            </div>
          ) : videos.length === 0 ? (
            <div className="text-center text-[#666] py-12 text-xs uppercase tracking-widest">
              No videos uploaded
            </div>
          ) : (
            <div className="divide-y divide-[#262626]">
              {videos.map((video) => (
                <div
                  key={video.id}
                  className="p-4 hover:bg-[#0a0a0a] transition-colors flex items-center gap-4"
                >
                  {/* Thumbnail */}
                  <div className="w-32 h-18 bg-[#111] border border-[#262626] flex items-center justify-center flex-shrink-0">
                    <Film className="h-6 w-6 text-red-800" />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="text-xs font-mono text-white truncate">
                      {video.original_filename}
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-[#666] font-mono">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatDuration(video.duration)}
                      </span>
                      <span>•</span>
                      <span>
                        {video.width}x{video.height} @ {video.fps.toFixed(1)}{" "}
                        fps
                      </span>
                      <span>•</span>
                      <span className="flex items-center gap-1">
                        <HardDrive className="h-3 w-3" />
                        {formatFileSize(video.file_size)}
                      </span>
                    </div>
                    <div className="text-[9px] text-[#555] font-mono">
                      Uploaded: {formatDate(video.uploaded_at)}
                    </div>
                    {video.description && (
                      <div className="text-[10px] text-[#888] italic">
                        {video.description}
                      </div>
                    )}
                    {video.use_count > 0 && (
                      <div className="text-[10px] text-green-500 font-mono flex items-center gap-1">
                        <AlertCircle className="h-3 w-3" />
                        Currently used by {video.use_count} source(s)
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setVideoToDelete(video);
                      setDeleteDialogOpen(true);
                    }}
                    disabled={video.use_count > 0}
                    className="h-8 w-8 rounded-none text-[#666] hover:text-red-500 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>

      {/* Upload Dialog */}
      <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
        <DialogContent className="border border-[#262626] bg-black p-0 max-w-md">
          <DialogHeader className="p-4 border-b border-[#262626]">
            <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono">
              Upload_Video :: Library
            </DialogTitle>
          </DialogHeader>
          <div className="p-4 space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-[#666]">
                Video File
              </label>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="w-full rounded-none border-[#262626] bg-[#050505] text-xs h-10 hover:bg-[#111] hover:border-white"
              >
                {uploadFile ? uploadFile.name : "Choose File..."}
              </Button>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-[#666]">
                Description (Optional)
              </label>
              <Textarea
                value={uploadDescription}
                onChange={(e) => setUploadDescription(e.target.value)}
                placeholder="Brief description..."
                className="rounded-none border-[#262626] bg-[#050505] text-xs min-h-[60px] focus-visible:ring-0 focus-visible:border-white resize-none"
              />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-[#666]">
                Tags (Optional, comma-separated)
              </label>
              <Input
                value={uploadTags}
                onChange={(e) => setUploadTags(e.target.value)}
                placeholder="indoor, test, cam1"
                className="rounded-none border-[#262626] bg-[#050505] text-xs h-9 focus-visible:ring-0 focus-visible:border-white"
              />
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                variant="ghost"
                onClick={() => setUploadDialogOpen(false)}
                className="flex-1 rounded-none text-xs h-9 border border-[#262626] text-[#888] hover:text-white"
              >
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                disabled={!uploadFile || uploading}
                className="flex-1 rounded-none bg-white text-black hover:bg-[#ccc] text-xs h-9 disabled:opacity-50"
              >
                {uploading ? "Uploading..." : "Upload"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="border border-[#262626] bg-black p-0 max-w-md">
          <DialogHeader className="p-4 border-b border-[#262626]">
            <DialogTitle className="text-xs uppercase tracking-[0.2em] font-mono text-red-500">
              Confirm_Deletion
            </DialogTitle>
          </DialogHeader>
          <div className="p-4 space-y-4">
            <div className="text-xs text-[#ccc]">
              Are you sure you want to delete this video?
            </div>
            {videoToDelete && (
              <div className="p-3 bg-[#111] border border-[#262626] space-y-1">
                <div className="text-xs font-mono text-white">
                  {videoToDelete.original_filename}
                </div>
                <div className="text-[10px] text-[#666] font-mono">
                  {formatFileSize(videoToDelete.file_size)} •{" "}
                  {videoToDelete.width}x{videoToDelete.height}
                </div>
              </div>
            )}
            <div className="text-[10px] text-[#888]">
              This action cannot be undone. The video file will be permanently
              deleted.
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                variant="ghost"
                onClick={() => setDeleteDialogOpen(false)}
                className="flex-1 rounded-none text-xs h-9 border border-[#262626] text-[#888] hover:text-white"
              >
                Cancel
              </Button>
              <Button
                onClick={handleDelete}
                className="flex-1 rounded-none bg-red-600 text-white hover:bg-red-700 text-xs h-9"
              >
                Delete_Forever
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
