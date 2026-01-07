'use client';

import { useEffect, useState } from 'react';
import { Header } from '@/components/layout/header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import api, { type Camera } from '@/lib/api';
import { Plus, Trash2, Power, PowerOff } from 'lucide-react';

export default function CamerasPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newCamera, setNewCamera] = useState({ name: '', latitude: '', longitude: '' });

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const data = await api.cameras.list();
      setCameras(data);
    } catch (error) {
      console.error('Failed to fetch cameras:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateCamera = async () => {
    try {
      await api.cameras.create({
        name: newCamera.name,
        latitude: parseFloat(newCamera.latitude),
        longitude: parseFloat(newCamera.longitude),
      });
      setIsDialogOpen(false);
      setNewCamera({ name: '', latitude: '', longitude: '' });
      fetchCameras();
    } catch (error) {
      console.error('Failed to create camera:', error);
    }
  };

  const handleToggleCamera = async (camera: Camera) => {
    try {
      if (camera.is_active) {
        await api.cameras.deactivate(camera.id);
      } else {
        await api.cameras.activate(camera.id);
      }
      fetchCameras();
    } catch (error) {
      console.error('Failed to toggle camera:', error);
    }
  };

  const handleDeleteCamera = async (id: number) => {
    try {
      await api.cameras.delete(id);
      fetchCameras();
    } catch (error) {
      console.error('Failed to delete camera:', error);
    }
  };

  return (
    <div className="flex flex-1 flex-col">
      <Header title="Cameras" />
      
      <main className="flex-1 p-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Camera Management</CardTitle>
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Camera
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New Camera</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <Input
                    placeholder="Camera Name"
                    value={newCamera.name}
                    onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
                  />
                  <Input
                    placeholder="Latitude"
                    type="number"
                    step="0.0001"
                    value={newCamera.latitude}
                    onChange={(e) => setNewCamera({ ...newCamera, latitude: e.target.value })}
                  />
                  <Input
                    placeholder="Longitude"
                    type="number"
                    step="0.0001"
                    value={newCamera.longitude}
                    onChange={(e) => setNewCamera({ ...newCamera, longitude: e.target.value })}
                  />
                  <Button onClick={handleCreateCamera} className="w-full">
                    Create Camera
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Location</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center">Loading...</TableCell>
                  </TableRow>
                ) : cameras.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-muted-foreground">
                      No cameras configured
                    </TableCell>
                  </TableRow>
                ) : (
                  cameras.map((camera) => (
                    <TableRow key={camera.id}>
                      <TableCell className="font-mono">{camera.id}</TableCell>
                      <TableCell className="font-medium">{camera.name}</TableCell>
                      <TableCell className="font-mono text-xs">
                        {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}
                      </TableCell>
                      <TableCell>
                        <Badge variant={camera.is_active ? 'default' : 'secondary'}>
                          {camera.is_active ? 'Active' : 'Inactive'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => handleToggleCamera(camera)}
                          >
                            {camera.is_active ? (
                              <PowerOff className="h-4 w-4" />
                            ) : (
                              <Power className="h-4 w-4" />
                            )}
                          </Button>
                          <Button
                            variant="destructive"
                            size="icon"
                            onClick={() => handleDeleteCamera(camera.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
