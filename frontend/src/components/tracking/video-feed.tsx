import { useEffect, useRef, useState } from "react";
import { Maximize2, Trash2, Crosshair, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface VideoFeedProps {
  cameraId: number;
  sourceName: string;
  isActive: boolean;
  onDelete?: () => void;
  onMaximize?: () => void;
}

export function VideoFeed({
  cameraId,
  sourceName,
  isActive,
  onDelete,
  onMaximize
}: VideoFeedProps) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const lastFrameTime = useRef(performance.now());
  const frameCount = useRef(0);

  useEffect(() => {
    const handleFrame = (e: CustomEvent) => {
      if (e.detail.camera_id === cameraId) {
        setImgSrc(`data:image/jpeg;base64,${e.detail.frame}`);

        // FPS Calc
        frameCount.current++;
        const now = performance.now();
        if (now - lastFrameTime.current >= 1000) {
          setFps(frameCount.current);
          frameCount.current = 0;
          lastFrameTime.current = now;
        }
      }
    };
    window.addEventListener('video-frame', handleFrame as EventListener);
    return () => window.removeEventListener('video-frame', handleFrame as EventListener);
  }, [cameraId]);

  return (
    <div className="relative aspect-video bg-black border border-[#262626] group overflow-hidden">
      {/* Feed Content */}
      {isActive ? (
        imgSrc ? (
          <img src={imgSrc} className="w-full h-full object-contain" alt={sourceName} />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-[#050505]">
            <div className="flex flex-col items-center gap-2 animate-pulse">
              <span className="text-xs font-mono text-[#333] uppercase">Connecting...</span>
            </div>
          </div>
        )
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-dot-pattern opacity-50">
          <div className="flex flex-col items-center gap-2">
            <span className="text-[10px] font-mono text-[#333] uppercase tracking-widest">Feed_Terminated</span>
          </div>
        </div>
      )}

      {/* Surveillance Overlay (Always visible) */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top Info Bar */}
        <div className="absolute top-0 left-0 right-0 p-2 flex justify-between items-start bg-gradient-to-b from-black/80 to-transparent">
          <div className="flex flex-col">
            <span className="text-white text-[10px] font-bold font-mono tracking-wider">{sourceName}</span>
            <span className="text-[#666] text-[8px] font-mono uppercase">CAM_{cameraId.toString().padStart(2, '0')}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-[#444] text-[9px] font-mono">{fps} FPS</span>
            {isActive && <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse" />}
          </div>
        </div>

        {/* Crosshair Center */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-20 group-hover:opacity-50 transition-opacity">
          <Crosshair className="w-8 h-8 text-white stroke-[1]" />
        </div>

        {/* Corner Brackets */}
        <div className="absolute top-2 left-2 w-4 h-4 border-l border-t border-white/50" />
        <div className="absolute top-2 right-2 w-4 h-4 border-r border-t border-white/50" />
        <div className="absolute bottom-2 left-2 w-4 h-4 border-l border-b border-white/50" />
        <div className="absolute bottom-2 right-2 w-4 h-4 border-r border-b border-white/50" />
      </div>

      {/* Hover Controls */}
      <div className="absolute bottom-0 left-0 right-0 p-2 bg-black/90 translate-y-full group-hover:translate-y-0 transition-transform duration-150 flex justify-between items-center border-t border-[#333]">
        <span className="text-[9px] text-[#444] font-mono uppercase tracking-widest">SECURE_LINK</span>
        <div className="flex gap-1">
          <button onClick={onMaximize} className="p-1.5 hover:bg-white hover:text-black text-[#888] transition-colors"><Maximize2 className="w-3 h-3" /></button>
          <button onClick={onDelete} className="p-1.5 hover:bg-red-900 hover:text-white text-[#888] transition-colors"><Trash2 className="w-3 h-3" /></button>
        </div>
      </div>
    </div>
  );
}
