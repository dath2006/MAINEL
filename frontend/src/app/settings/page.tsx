'use client';

import { Header } from '@/components/layout/header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

export default function SettingsPage() {
  return (
    <div className="flex flex-1 flex-col">
      <Header title="Settings" />
      
      <main className="flex-1 space-y-6 p-6">
        <Card>
          <CardHeader>
            <CardTitle>API Configuration</CardTitle>
            <CardDescription>Configure backend connection settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Backend URL</label>
              <Input defaultValue="http://localhost:8000" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">WebSocket URL</label>
              <Input defaultValue="ws://localhost:8000/api/v1/ws/tracks" />
            </div>
            <Button>Save Configuration</Button>
          </CardContent>
        </Card>

        <Separator />

        <Card>
          <CardHeader>
            <CardTitle>Detection Settings</CardTitle>
            <CardDescription>Configure YOLO detection parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Confidence Threshold</label>
              <Input type="number" step="0.1" min="0" max="1" defaultValue="0.5" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">IOU Threshold</label>
              <Input type="number" step="0.1" min="0" max="1" defaultValue="0.45" />
            </div>
            <Button>Update Settings</Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
