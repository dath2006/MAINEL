import { Users, Video, Activity, Eye } from "lucide-react";

interface StatsCardsProps {
  activeTracks?: number;
  totalCameras?: number;
  activeCameras?: number;
  detectionsPerSecond?: number;
}

export function StatsCards({
  activeTracks = 0,
  totalCameras = 0,
  activeCameras = 0,
  detectionsPerSecond = 0,
}: StatsCardsProps) {
  const stats = [
    {
      label: 'TARGETS_ACTIVE',
      value: activeTracks,
      sub: 'Tracking',
      highlight: true
    },
    {
      label: 'CAMERA_NODES',
      value: `${activeCameras}/${totalCameras}`,
      sub: 'Online',
      highlight: false
    },
    {
      label: 'PROCESS_RATE',
      value: detectionsPerSecond.toFixed(1),
      sub: 'Items/Sec',
      highlight: false
    },
    {
      label: 'SYS_UPTIME',
      value: '04:22:19',
      sub: 'Stable',
      highlight: false
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 border-y border-[#262626] bg-black">
      {stats.map((stat, i) => (
        <div
          key={i}
          className={`
                p-6 border-r border-[#262626] last:border-r-0 
                flex flex-col justify-between h-32
                hover:bg-[#050505] transition-colors group
            `}
        >
          <div className="flex justify-between items-start">
            <span className="text-[10px] uppercase tracking-[0.2em] text-[#666] group-hover:text-[#888]">{stat.label}</span>
            {stat.highlight && <div className="w-1.5 h-1.5 bg-white animate-pulse" />}
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-3xl font-light font-mono text-white tracking-tighter">
              {stat.value}
            </span>
            <span className="text-[10px] uppercase text-[#444] tracking-widest font-mono group-hover:text-[#666]">
              [{stat.sub}]
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
