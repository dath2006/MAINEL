import type { Metadata } from "next";
import "./globals.css";

import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";

export const metadata: Metadata = {
  title: "MCMT-ReID | MONITOR",
  description: "Multi-Camera Multi-Target Person Re-Identification System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full w-full" suppressHydrationWarning>
      <body className="antialiased h-full w-full bg-black text-foreground selection:bg-white selection:text-black">
        <SidebarProvider defaultOpen={true}>
          <AppSidebar />
          <SidebarInset className="bg-black border-l border-[#262626]">
            {children}
          </SidebarInset>
        </SidebarProvider>
      </body>
    </html>
  );
}
