"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Users, Trash2, RefreshCw, Camera } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const USE_MOCK = false;

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

interface PersonGalleryProps {
  apiUrl?: string;
  refreshInterval?: number;
}

export function PersonGallery({
  apiUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") +
    "/api/v1/streams",
  refreshInterval = 5000,
}: PersonGalleryProps) {
  const [persons, setPersons] = useState<PersonEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPerson, setSelectedPerson] = useState<string | null>(null);
  const [captures, setCaptures] = useState<CaptureEntry[]>([]);
  const [loadingCaptures, setLoadingCaptures] = useState(false);

  const fetchCaptures = async (globalId: string) => {
    if (USE_MOCK) return;
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

  const fetchGallery = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/gallery`);
      if (!response.ok) throw new Error("Failed to fetch gallery");
      const data = await response.json();
      setPersons(data.persons || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const clearGallery = async () => {
    try {
      const response = await fetch(`${apiUrl}/gallery`, { method: "DELETE" });
      if (!response.ok) throw new Error("Failed to clear gallery");
      setPersons([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  useEffect(() => {
    fetchGallery();
    const interval = setInterval(fetchGallery, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

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
    <div className="flex flex-col h-full bg-[#050505]">
      <div className="flex items-center justify-between p-3 border-b border-[#262626]">
        <span className="text-[10px] uppercase tracking-[0.2em] font-medium text-[#888] flex items-center gap-2">
          <Users className="h-3 w-3" />
          IDENTITY_LOG ({persons.length})
        </span>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 rounded-none text-[#666] hover:text-white"
            onClick={fetchGallery}
          >
            <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
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

      <div className="flex-1 overflow-hidden p-2">
        {error && (
          <div className="text-[10px] text-red-500 mb-2 font-mono">{error}</div>
        )}

        <ScrollArea className="h-full pr-2">
          {persons.length === 0 ? (
            <div className="text-center py-8 text-red-800">
              <p className="text-[10px] uppercase tracking-widest">
                Awaiting Detection...
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2">
              {persons.map((person) => (
                <div
                  key={person.global_id}
                  className="group relative bg-black border border-[#262626] cursor-pointer hover:border-white transition-colors"
                  onClick={() => handlePersonClick(person.global_id)}
                >
                  <div className="aspect-[3/4] overflow-hidden grayscale group-hover:grayscale-0 transition-all bg-[#111]">
                    {person.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${person.thumbnail}`}
                        className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Users className="h-6 w-6 text-red-800" />
                      </div>
                    )}
                    {/* Corner Markers */}
                    <div className="absolute top-0 left-0 w-2 h-2 border-l border-t border-white/30" />
                    <div className="absolute top-0 right-0 w-2 h-2 border-r border-t border-white/30" />
                    <div className="absolute bottom-0 left-0 w-2 h-2 border-l border-b border-white/30" />
                    <div className="absolute bottom-0 right-0 w-2 h-2 border-r border-b border-white/30" />
                  </div>

                  <div className="p-2 border-t border-[#262626] bg-[#050505]">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[9px] font-bold text-white bg-[#222] px-1 font-mono">
                        {person.global_id.slice(0, 6)}
                      </span>
                      <span className="text-[9px] font-mono text-[#666]">
                        {formatTimeAgo(person.last_seen)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-[9px] text-[#555] font-mono uppercase">
                      <span>CAM_{person.last_camera_id}</span>
                      <span>CNT:{person.appearance_count}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

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
              <div className="grid grid-cols-4 gap-2">
                {captures.map((cap, idx) => (
                  <div key={idx} className="bg-black border border-[#262626]">
                    <div className="aspect-[3/4] relative">
                      <img
                        src={`data:image/jpeg;base64,${cap.image_b64}`}
                        className="w-full h-full object-contain opacity-80"
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
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
