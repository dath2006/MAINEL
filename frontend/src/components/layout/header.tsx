"use client";

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

interface HeaderProps {
  title: string;
  className?: string;
}

export function Header({ title, className }: HeaderProps) {
  const [time, setTime] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header
      className={cn(
        "flex h-14 items-center justify-between border-b border-[#262626] bg-[#050505] px-6 select-none",
        className,
      )}
    >
      <div className="flex items-center gap-6">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase text-[#444] tracking-[0.2em] leading-none mb-1">
            System_View
          </span>
          <h1 className="text-sm font-bold uppercase tracking-[0.1em] text-white leading-none">
            {title}
          </h1>
        </div>

        {/* Decorative Grid Fragment */}
        <div className="hidden md:flex gap-0.5">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={cn("w-1 h-3", i === 2 ? "bg-red-800" : "bg-[#111]")}
            />
          ))}
        </div>
      </div>

      <div className="flex items-center gap-8">
        <div className="hidden md:flex flex-col items-end">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#666] font-mono">MEM_USAGE</span>
            <div className="w-16 h-1.5 bg-[#111] overflow-hidden">
              <div className="h-full bg-white w-[40%]" />
            </div>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] text-[#666] font-mono">CPU_LOAD</span>
            <div className="w-16 h-1.5 bg-[#111] overflow-hidden">
              <div className="h-full bg-white w-[65%]" />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 border-l border-[#262626] pl-6 h-8">
          <span className="text-xs font-mono text-[#888] tracking-widest">
            {time}
          </span>
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 bg-white animate-pulse" />
            <span className="text-[10px] uppercase font-bold tracking-widest">
              Connected
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
