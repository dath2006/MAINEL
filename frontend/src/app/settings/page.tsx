"use client";

import { Header } from "@/components/layout/header";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { VideoUploadManager } from "@/components/tracking/video-upload-manager";

export default function SettingsPage() {
  return (
    <div className="flex flex-1 flex-col bg-black text-white font-mono h-screen overflow-hidden">
      <Header title="SYSTEM_CONFIG" />

      <main className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full space-y-8">
        {/* Video Library Management */}
        <section>
          <div className="flex items-center gap-4 mb-4">
            <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">
              Video_Library
            </h2>
            <div className="h-px bg-[#262626] flex-1" />
          </div>
          <VideoUploadManager />
        </section>

        {/* Network Configuration */}
        <section>
          <div className="flex items-center gap-4 mb-4">
            <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">
              Network_Interface
            </h2>
            <div className="h-px bg-[#262626] flex-1" />
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="group space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-[#666] group-focus-within:text-white transition-colors">
                Backend Endpoint
              </label>
              <Input
                defaultValue={
                  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
                }
                className="rounded-none border-[#262626] bg-[#050505] text-xs h-10 px-4 focus-visible:ring-0 focus-visible:border-white transition-colors"
              />
            </div>
            <div className="group space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-[#666] group-focus-within:text-white transition-colors">
                WebSocket Stream
              </label>
              <Input
                defaultValue={
                  (
                    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
                  ).replace(/^http/, "ws") + "/api/v1/ws/tracks"
                }
                className="rounded-none border-[#262626] bg-[#050505] text-xs h-10 px-4 focus-visible:ring-0 focus-visible:border-white transition-colors"
              />
            </div>
          </div>
        </section>

        {/* Detection Parameters */}
        <section>
          <div className="flex items-center gap-4 mb-4">
            <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">
              Inference_Engine
            </h2>
            <div className="h-px bg-[#262626] flex-1" />
          </div>

          <div className="bg-[#050505] border border-[#262626] p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label className="text-xs font-bold uppercase tracking-wider block">
                  Confidence Threshold
                </label>
                <p className="text-[10px] text-[#666] max-w-[300px]">
                  Minimum probability required for object classification
                  acceptance.
                </p>
              </div>
              <div className="w-24">
                <Input
                  type="number"
                  step="0.05"
                  defaultValue="0.5"
                  className="text-right rounded-none border-b border-t-0 border-l-0 border-r-0 border-red-800 bg-transparent focus-visible:ring-0 focus-visible:border-white h-auto py-1 px-0 font-mono text-sm"
                />
              </div>
            </div>

            <div className="h-px bg-[#111]" />

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label className="text-xs font-bold uppercase tracking-wider block">
                  IOU Threshold
                </label>
                <p className="text-[10px] text-[#666] max-w-[300px]">
                  Intersection over Union threshold for non-maximum suppression.
                </p>
              </div>
              <div className="w-24">
                <Input
                  type="number"
                  step="0.05"
                  defaultValue="0.45"
                  className="text-right rounded-none border-b border-t-0 border-l-0 border-r-0 border-red-800 bg-transparent focus-visible:ring-0 focus-visible:border-white h-auto py-1 px-0 font-mono text-sm"
                />
              </div>
            </div>
          </div>
        </section>

        <div className="pt-4 flex justify-end">
          <Button className="rounded-none bg-white text-black hover:bg-[#ccc] uppercase tracking-widest text-xs h-10 px-8">
            Apply_Configuration
          </Button>
        </div>
      </main>
    </div>
  );
}
