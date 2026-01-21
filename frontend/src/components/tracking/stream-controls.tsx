import { Play, Pause, Square, Activity, Settings2, Loader2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

interface StreamControlsProps {
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  state: 'playing' | 'paused' | 'stopped' | 'loading';
  sourceCount: number;

}

export function StreamControls({
  onPlay,
  onPause,
  onStop,
  state,
  sourceCount,
}: StreamControlsProps) {
  return (
    <div className="bg-[#050505] border border-[#262626] p-1 flex items-center gap-1 shadow-2xl">
      {/* Playback Group */}
      <div className="flex items-center">
        <Button
          variant="ghost"
          size="icon"
          onClick={onPlay}
          disabled={state === 'playing' || state === 'loading'}
          className="h-8 w-8 rounded-none hover:bg-white hover:text-black disabled:opacity-30 text-[#888]"
        >
          <Play className="h-3 w-3 fill-current" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onPause}
          disabled={state === 'paused' || state === 'stopped'}
          className="h-8 w-8 rounded-none hover:bg-white hover:text-black disabled:opacity-30 text-[#888]"
        >
          <Pause className="h-3 w-3 fill-current" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onStop}
          className="h-8 w-8 rounded-none hover:bg-red-900 hover:text-white text-[#888]"
        >
          <Square className="h-3 w-3 fill-current" />
        </Button>
      </div>

      <div className="w-px h-4 bg-[#262626] mx-1" />

      {/* Stats Group */}
      <div className="flex items-center gap-3 px-2">
        <div className="flex flex-col items-center">
          <span className="text-[8px] uppercase text-[#444] tracking-wider">Sources</span>
          <span className="text-xs font-mono font-bold text-white">{sourceCount}</span>
        </div>

      </div>

      {/* Status Indicator */}
      <div className={`
            ml-2 h-8 px-3 flex items-center gap-2 border-l border-[#262626]
            ${state === 'playing' ? 'bg-[#0a200a]' : state === 'loading' ? 'bg-[#202000]' : 'bg-[#0a0a0a]'}
       `}>
        <div className={`w-1.5 h-1.5 rounded-full ${state === 'playing' ? 'bg-green-500 animate-pulse' : state === 'loading' ? 'bg-yellow-500 animate-spin' : 'bg-red-900'}`} />
        <span className={`text-[9px] uppercase tracking-widest font-mono ${state === 'playing' ? 'text-green-500' : state === 'loading' ? 'text-yellow-500' : 'text-red-900'}`}>
          {state}
        </span>
      </div>
    </div>
  );
}
