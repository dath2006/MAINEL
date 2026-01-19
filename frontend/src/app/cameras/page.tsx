"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/header";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import api, { type Camera } from "@/lib/api";
import { Plus, Trash2, Power, PowerOff } from "lucide-react";

export default function CamerasPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newCamera, setNewCamera] = useState({
    name: "",
    latitude: "",
    longitude: "",
  });

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const data = await api.cameras.list();
      setCameras(data);
    } catch (error) {
      console.error(error);
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
      setNewCamera({ name: "", latitude: "", longitude: "" });
      fetchCameras();
    } catch (error) {
      console.error(error);
    }
  };

  const handleToggleCamera = async (camera: Camera) => {
    try {
      if (camera.is_active) await api.cameras.deactivate(camera.id);
      else await api.cameras.activate(camera.id);
      fetchCameras();
    } catch (error) {
      console.error(error);
    }
  };

  const handleDeleteCamera = async (id: number) => {
    try {
      await api.cameras.delete(id);
      fetchCameras();
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="flex flex-1 flex-col bg-black text-white font-mono h-screen">
      <Header title="NODE_MANAGEMENT" />

      <main className="flex-1 p-8 max-w-5xl mx-auto w-full">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-xs uppercase tracking-[0.2em] text-[#888]">
            Registered_Nodes
          </h2>

          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogTrigger asChild>
              <Button
                variant="outline"
                className="rounded-none border-red-800 hover:bg-white hover:text-black uppercase text-[10px] tracking-widest h-8"
              >
                <Plus className="mr-2 h-3 w-3" />
                Link_Node
              </Button>
            </DialogTrigger>
            <DialogContent className="rounded-none border-[#262626] bg-black">
              <DialogHeader className="border-b border-[#262626] p-4">
                <DialogTitle className="text-xs uppercase tracking-[0.2em]">
                  New_Connection
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-4 p-4">
                <Input
                  placeholder="NODE_ALIAS"
                  value={newCamera.name}
                  onChange={(e) =>
                    setNewCamera({ ...newCamera, name: e.target.value })
                  }
                  className="rounded-none border-red-800 bg-[#050505] text-xs"
                />
                <div className="grid grid-cols-2 gap-4">
                  <Input
                    placeholder="LAT"
                    type="number"
                    step="0.0001"
                    value={newCamera.latitude}
                    onChange={(e) =>
                      setNewCamera({ ...newCamera, latitude: e.target.value })
                    }
                    className="rounded-none border-red-800 bg-[#050505] text-xs"
                  />
                  <Input
                    placeholder="LNG"
                    type="number"
                    step="0.0001"
                    value={newCamera.longitude}
                    onChange={(e) =>
                      setNewCamera({ ...newCamera, longitude: e.target.value })
                    }
                    className="rounded-none border-red-800 bg-[#050505] text-xs"
                  />
                </div>
                <Button
                  onClick={handleCreateCamera}
                  className="w-full rounded-none bg-white text-black hover:bg-[#ccc] uppercase text-[10px] tracking-widest"
                >
                  Initialize
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        <div className="border border-[#262626] bg-[#050505]">
          {/* Header */}
          <div className="grid grid-cols-12 bg-[#111] p-3 text-[10px] uppercase tracking-widest text-[#666] border-b border-[#262626]">
            <div className="col-span-1">ID</div>
            <div className="col-span-4">Node_Label</div>
            <div className="col-span-4">Coordinates</div>
            <div className="col-span-2">State</div>
            <div className="col-span-1 text-right">CMD</div>
          </div>

          {/* List */}
          {isLoading ? (
            <div className="p-8 text-center text-xs text-[#444] animate-pulse">
              Scanning...
            </div>
          ) : cameras.length === 0 ? (
            <div className="p-8 text-center text-xs text-[#444]">
              No nodes found on network.
            </div>
          ) : (
            cameras.map((camera) => (
              <div
                key={camera.id}
                className="grid grid-cols-12 p-3 text-xs border-b border-[#111] items-center hover:bg-[#0a0a0a] transition-colors group"
              >
                <div className="col-span-1 font-mono text-[#444]">
                  #{camera.id}
                </div>
                <div className="col-span-4 font-bold">{camera.name}</div>
                <div className="col-span-4 font-mono text-[#666] text-[10px]">
                  {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}
                </div>
                <div className="col-span-2">
                  <span
                    className={`inline-flex items-center gap-1.5 px-2 py-0.5 border ${camera.is_active ? "border-green-900 bg-green-900/10 text-green-500" : "border-red-800 bg-[#111] text-[#666]"} text-[9px] uppercase tracking-wider`}
                  >
                    <span
                      className={`w-1 h-1 rounded-full ${camera.is_active ? "bg-green-500 animate-pulse" : "bg-[#444]"}`}
                    />
                    {camera.is_active ? "Online" : "Standby"}
                  </span>
                </div>
                <div className="col-span-1 flex justify-end gap-2 opacity-50 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => handleToggleCamera(camera)}
                    className="hover:text-white text-[#666]"
                  >
                    {camera.is_active ? (
                      <PowerOff className="h-4 w-4" />
                    ) : (
                      <Power className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    onClick={() => handleDeleteCamera(camera.id)}
                    className="hover:text-red-500 text-[#666]"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}
