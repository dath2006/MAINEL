"use client";

import {
  Users,
  LayoutDashboard,
  Map,
  Settings,
  Camera,
  Radio,
  BarChart,
  HardDrive,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { useEffect, useState } from "react";

const navigation = [
  { name: "Live Track", href: "/tracking", icon: Map },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function AppSidebar() {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [wsState, setWsState] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Mock FS state check
    const interval = setInterval(() => setWsState((prev) => !prev), 5000);
    return () => clearInterval(interval);
  }, []);

  if (!mounted) return null;

  return (
    <Sidebar
      className="border-r border-[#262626] bg-[#000000] w-[64px]"
      collapsible="none"
    >
      {/* Heavy Top Slab */}
      <SidebarHeader className="h-14 border-b border-[#262626] flex items-center justify-center bg-black">
        <div className="w-8 h-8 bg-white flex items-center justify-center">
          <div className="w-4 h-4 bg-black rounded-full" />
        </div>
      </SidebarHeader>

      <SidebarContent className="bg-black py-4">
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu className="gap-2 px-2">
              {navigation.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <SidebarMenuItem key={item.name}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      className={cn(
                        "h-10 w-10 justify-center rounded-none transition-none border border-transparent",
                        isActive
                          ? "bg-white text-black hover:bg-white hover:text-black"
                          : "bg-[#111] text-[#666] hover:bg-[#222] hover:text-[#888] hover:border-red-800",
                      )}
                      tooltip={item.name}
                    >
                      <Link href={item.href}>
                        <item.icon className="h-4 w-4" />
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-[#262626] p-0 bg-black">
        <div className="flex flex-col items-center gap-2 py-4">
          {/* Status Indicators */}
          <div className="w-10 h-10 border border-[#262626] bg-[#050505] flex flex-col items-center justify-center gap-1">
            <div
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                wsState ? "bg-white" : "bg-red-800",
              )}
            />
            <span className="text-[8px] text-[#444] font-mono">NET</span>
          </div>
          <div className="w-10 h-10 border border-[#262626] bg-[#050505] flex flex-col items-center justify-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-red-800" />
            <span className="text-[8px] text-[#444] font-mono">REC</span>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
