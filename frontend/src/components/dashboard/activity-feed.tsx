import { ScrollArea } from "@/components/ui/scroll-area";
import { Terminal } from "lucide-react";

interface ActivityItem {
  type: string;
  data: any;
  timestamp: Date;
}

interface ActivityFeedProps {
  events?: ActivityItem[];
}

export function ActivityFeed({ events = [] }: ActivityFeedProps) {
  return (
    <div className="h-full flex flex-col border border-[#262626] bg-[#050505]">
      {/* Header */}
      <div className="p-3 border-b border-[#262626] flex items-center justify-between bg-black">
        <div className="flex items-center gap-2">
          <Terminal className="w-3 h-3 text-[#666]" />
          <span className="text-[10px] uppercase tracking-[0.2em] text-[#888]">Event_Log</span>
        </div>
        <div className="flex gap-1">
          <div className="w-1 h-1 bg-[#444]" />
          <div className="w-1 h-1 bg-[#444]" />
          <div className="w-1 h-1 bg-[#444]" />
        </div>
      </div>

      <ScrollArea className="flex-1 p-0">
        <div className="flex flex-col font-mono text-xs">
          {events.length === 0 ? (
            <div className="p-4 text-[#444] text-[10px] uppercase">No_Recent_Activity</div>
          ) : events.map((item, i) => (
            <div
              key={i}
              className={`
                        flex gap-3 p-3 border-b border-[#111] 
                        hover:bg-[#0a0a0a] transition-colors
                        text-[#888]
                    `}
            >
              <span className="text-[10px] text-[#444] min-w-[60px]">
                {new Date(item.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}
              </span>

              <div className="flex-1 flex flex-col gap-0.5">
                <div className="flex items-center gap-2">
                  <span className="uppercase font-bold tracking-wider text-[10px] text-[#555]">
                    [{item.type}]
                  </span>
                  <span className="text-[#ccc] text-[10px]">
                    {typeof item.data === 'string' ? item.data : JSON.stringify(item.data)}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {/* Typing cursor effect */}
          <div className="p-3 flex items-center gap-2">
            <span className="text-[10px] text-[#444]">{new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}</span>
            <div className="w-2 h-4 bg-white animate-pulse" />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
