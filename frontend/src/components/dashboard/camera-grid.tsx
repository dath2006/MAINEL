import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CameraGridProps {
  cameras?: { id: number; name: string; is_active: boolean }[];
  onAddCamera?: () => void;
}

export function CameraGrid({ cameras = [], onAddCamera }: CameraGridProps) {
  return (
    <div className="h-full flex flex-col border border-[#262626] bg-[#050505]">
      <div className="p-3 border-b border-[#262626] flex items-center justify-between bg-black">
        <span className="text-[10px] uppercase tracking-[0.2em] text-[#888]">Active_Feeds</span>
        <span className="text-[10px] text-[#444] font-mono">GRID_VIEW_4X4</span>
      </div>

      <div className="flex-1 p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {cameras.map((cam) => (
          <div key={cam.id} className="aspect-video bg-black border border-[#262626] relative group">
            {/* Camera Placeholder / Feed Area */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-[#222] text-[10px] uppercase tracking-widest font-mono">Signal_Offline</div>
            </div>

            {/* Overlays */}
            <div className="absolute top-2 left-2 flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${cam.is_active ? 'bg-green-600' : 'bg-red-900'}`} />
              <span className="bg-black/80 text-white text-[9px] px-1 font-mono">{cam.name}</span>
            </div>

            {/* Corner Markers */}
            <div className="absolute top-0 left-0 w-2 h-2 border-l border-t border-[#444]" />
            <div className="absolute top-0 right-0 w-2 h-2 border-r border-t border-[#444]" />
            <div className="absolute bottom-0 left-0 w-2 h-2 border-l border-b border-[#444]" />
            <div className="absolute bottom-0 right-0 w-2 h-2 border-r border-b border-[#444]" />
          </div>
        ))}

        {/* Add Button */}
        <button
          onClick={onAddCamera}
          className="aspect-video border border-dashed border-[#262626] hover:border-[#444] hover:bg-[#0a0a0a] transition-colors flex flex-col items-center justify-center gap-2 group"
        >
          <div className="w-8 h-8 flex items-center justify-center rounded-full bg-[#111] group-hover:bg-[#222]">
            <Plus className="w-4 h-4 text-[#444] group-hover:text-white" />
          </div>
          <span className="text-[9px] uppercase tracking-widest text-[#444] group-hover:text-[#666]">Connect_Source</span>
        </button>
      </div>
    </div>
  );
}
