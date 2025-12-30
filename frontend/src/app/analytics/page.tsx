'use client';

import { Header } from '@/components/layout/header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function AnalyticsPage() {
  return (
    <div className="flex flex-1 flex-col">
      <Header title="Analytics" />
      
      <main className="flex-1 space-y-6 p-6">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Track Count Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex h-64 items-center justify-center text-muted-foreground">
                Chart placeholder - Connect to backend for real data
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Camera Transitions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex h-64 items-center justify-center text-muted-foreground">
                Heatmap placeholder - Shows camera-to-camera transitions
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">ReID Match History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex h-40 items-center justify-center text-muted-foreground">
              Table of recent identity matches across cameras
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
