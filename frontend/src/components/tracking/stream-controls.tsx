'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, Square, RefreshCw } from 'lucide-react';

interface StreamControlsProps {
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  state: 'stopped' | 'playing' | 'paused';
  sourceCount: number;
  fps?: number;
  onFpsChange?: (fps: number) => void;
}

export function StreamControls({
  onPlay,
  onPause,
  onStop,
  state,
  sourceCount,
  fps = 30,
  onFpsChange,
}: StreamControlsProps) {
  const [targetFps, setTargetFps] = useState(fps);

  const handleFpsChange = (value: number[]) => {
    const newFps = value[0];
    setTargetFps(newFps);
    onFpsChange?.(newFps);
  };

  return (

      <div className="flex flex-wrap items-center justify-between gap-4 border-2 p-2 rounded-lg">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          {state === 'playing' ? (
            <Button variant="outline" size="sm" onClick={onPause}>
              <Pause className="mr-2 h-4 w-4 " />
              Pause
            </Button>
          ) : (
            <Button size="sm" onClick={onPlay} disabled={sourceCount === 0}>
              <Play className="mr-2 h-4 w-4" />
              Play
            </Button>
          )}
          
          <Button
            variant="destructive"
            size="sm"
            onClick={onStop}
            disabled={state === 'stopped'}
          >
            <Square className="mr-2 h-4 w-4" />
            Stop
          </Button>
        </div>

        {/* Status */}
        <div className="flex items-center gap-3">
          <Badge variant={state === 'playing' ? 'default' : 'secondary'}>
            {state.charAt(0).toUpperCase() + state.slice(1)}
          </Badge>
          <span className="text-sm text-muted-foreground">
            {sourceCount} source{sourceCount !== 1 ? 's' : ''} active
          </span>
        </div>

        {/* FPS Control */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">FPS:</span>
          <div className="w-32">
            <Slider
              value={[targetFps]}
              min={1}
              max={60}
              step={1}
              onValueChange={handleFpsChange}
            />
          </div>
          <span className="w-8 text-sm font-medium">{targetFps}</span>
        </div>
      </div>

  );
}
