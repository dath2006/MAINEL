'use client';

import { Header } from '@/components/layout/header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Activity, GitCommit } from 'lucide-react';

export default function AnalyticsPage() {
  return (
    <div className="flex flex-1 flex-col bg-black text-white font-mono h-screen">
      <Header title="DATA_ANALYSIS" />

      <main className="flex-1 space-y-4 p-8 max-w-6xl mx-auto w-full">

        <div className="flex items-center gap-4 mb-4">
          <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">System_Metrics</h2>
          <div className="h-px bg-[#262626] flex-1" />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="bg-[#050505] border border-[#262626] rounded-none">
            <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-[#262626]">
              <CardTitle className="text-[10px] uppercase tracking-widest text-[#666]">Track_Volume</CardTitle>
              <BarChart className="h-4 w-4 text-[#444]" />
            </CardHeader>
            <CardContent className="pt-8">
              <div className="flex h-64 items-center justify-center text-[#333] border border-dashed border-[#222]">
                <span className="text-xs uppercase tracking-widest">NO_DATA_SOURCE</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-[#050505] border border-[#262626] rounded-none">
            <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-[#262626]">
              <CardTitle className="text-[10px] uppercase tracking-widest text-[#666]">Node_Transitions</CardTitle>
              <Activity className="h-4 w-4 text-[#444]" />
            </CardHeader>
            <CardContent className="pt-8">
              <div className="flex h-64 items-center justify-center text-[#333] border border-dashed border-[#222]">
                <span className="text-xs uppercase tracking-widest">AWAITING_INPUT</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex items-center gap-4 mt-8 mb-4">
          <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">Identification_Logs</h2>
          <div className="h-px bg-[#262626] flex-1" />
        </div>

        <Card className="bg-[#050505] border border-[#262626] rounded-none">
          <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-[#262626]">
            <CardTitle className="text-[10px] uppercase tracking-widest text-[#666]">ReID_History</CardTitle>
            <GitCommit className="h-4 w-4 text-[#444]" />
          </CardHeader>
          <CardContent className="pt-8">
            <div className="flex h-40 items-center justify-center text-[#333] border border-dashed border-[#222]">
              <span className="text-xs uppercase tracking-widest">LOG_BUFFER_EMPTY</span>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
