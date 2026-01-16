"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Users, Trash2, RefreshCw, X, Camera } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";


const USE_MOCK = false; // for mock data: set to false later

interface PersonEntry {
    global_id: string;
    last_camera_id: number;
    last_seen: string;
    appearance_count: number;
    thumbnail: string | null;
}

interface CaptureEntry {
    image_b64: string;
    quality_score: number;
    pose: string;
    sharpness: number;
    timestamp: string | null;
}

const MOCK_PERSONS: PersonEntry[] = [ //for mock data
    {
        global_id: "mock-001",
        last_camera_id: 1,
        last_seen: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
        appearance_count: 4,
        thumbnail: null,
    },
    {
        global_id: "mock-002",
        last_camera_id: 3,
        last_seen: new Date(Date.now() - 8 * 60 * 1000).toISOString(),
        appearance_count: 2,
        thumbnail: null,
    },
    {
        global_id: "mock-003",
        last_camera_id: 2,
        last_seen: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
        appearance_count: 7,
        thumbnail: null,
    },
];


interface PersonGalleryProps {
    apiUrl?: string;
    refreshInterval?: number;
}

export function PersonGallery({ 
    apiUrl = "http://localhost:8000/api/v1/streams",
    refreshInterval = 5000 
}: PersonGalleryProps) {
    const [persons, setPersons] = useState<PersonEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Popup state
    const [selectedPerson, setSelectedPerson] = useState<string | null>(null);
    const [captures, setCaptures] = useState<CaptureEntry[]>([]);
    const [loadingCaptures, setLoadingCaptures] = useState(false);

    // Fetch captures for a specific person
    const fetchCaptures = async (globalId: string) => {
        if (USE_MOCK) return; 
        try {
            setLoadingCaptures(true);
            const response = await fetch(`${apiUrl}/gallery/${globalId}/captures`);
            if (!response.ok) throw new Error("Failed to fetch captures");
            const data = await response.json();
            setCaptures(data.captures || []);
        } catch (err) {
            console.error(err);
        } finally {
            setLoadingCaptures(false);
        }
    };

    const handlePersonClick = (globalId: string) => {
        setSelectedPerson(globalId);
        fetchCaptures(globalId);
    };

    const fetchGallery = async () => {

        if(USE_MOCK){  // for mock data
            setPersons(MOCK_PERSONS);
            setError(null);
            return;
        }
        try {
            setLoading(true);
            const response = await fetch(`${apiUrl}/gallery`);
            if (!response.ok) throw new Error("Failed to fetch gallery");
            const data = await response.json();
            setPersons(data.persons || []);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setLoading(false);
        }
    };

    const clearGallery = async () => {
        try {
            const response = await fetch(`${apiUrl}/gallery`, { method: "DELETE" });
            if (!response.ok) throw new Error("Failed to clear gallery");
            setPersons([]);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        }
    };

    useEffect(() => {
        fetchGallery();

        if(USE_MOCK) return; // for mock data
        const interval = setInterval(fetchGallery, refreshInterval);
        return () => clearInterval(interval);
    }, [refreshInterval]);

    const formatTimeAgo = (isoString: string) => {
        const date = new Date(isoString);
        const now = new Date();
        const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
        
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return date.toLocaleDateString();
    };

    return (
        <Card className="h-full">
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg">
                        <Users className="h-5 w-5" />
                        Person Gallery
                        <Badge variant="secondary">{persons.length}</Badge>

                        {USE_MOCK && ( // for mock data
                        <Badge variant="secondary" className="ml-2">
                            Mock
                        </Badge>
                        )}
                    </CardTitle>
                    <div className="flex gap-2">
                        <Button 
                            variant="outline" 
                            size="sm" 
                            onClick={fetchGallery}
                            disabled={loading}
                        >
                            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                        </Button>
                        <Button 
                            variant="destructive" 
                            size="sm" 
                            onClick={clearGallery}
                            disabled={persons.length === 0}
                        >
                            <Trash2 className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                {error && (
                    <div className="text-sm text-destructive mb-2">{error}</div>
                )}
                <ScrollArea className="h-[400px]">
                    {persons.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                            No persons detected yet
                        </div>
                    ) : (
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3">
                            {persons.map((person) => (
                                <div 
                                    key={person.global_id}
                                    className="border bg-card rounded-xl p-3 pt-6 flex flex-col items-center gap-2  drop-shadow-md hover:shadow-lg hover:translate-y-0.5 hover:bg-accent/20 hover:border-gray-400 transition-all "
                                >
                                    {/* Thumbnail */}
                                    <div 
                                        className="w-full aspect-[9/16] ring-1 ring-border bg-muted rounded hover:ring-primary cursor-pointer overflow-hidden relative group"
                                        onClick={() => handlePersonClick(person.global_id)}
                                    >
                                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors z-10 flex items-center justify-center opacity-0 group-hover:opacity-100">
                                            <Camera className="text-white w-8 h-8 drop-shadow-md" />
                                        </div>
                                        {person.thumbnail ? (
                                            <img 
                                                src={`data:image/jpeg;base64,${person.thumbnail}`}
                                                alt={`Person ${person.global_id.slice(0, 8)}`}
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                                                <Users className="h-8 w-8" />
                                            </div>
                                        )}
                                    </div>
                                    
                                    {/* Info */}
                                    <div className="text-center w-full">
                                        <Badge  className="text-xs font-medium mt-1 mb-1">
                                            {person.global_id.slice(0, 8)}
                                        </Badge>
                                        <div className="text-xs text-foreground">
                                            Cam {person.last_camera_id}
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {formatTimeAgo(person.last_seen)}<br/>
                                            {person.appearance_count} obs
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </ScrollArea>
            </CardContent>

            {/* Captures Modal */}
            <Dialog open={!!selectedPerson} onOpenChange={(open) => !open && setSelectedPerson(null)}>
                <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <Users className="h-5 w-5" />
                            Identity: {selectedPerson?.slice(0, 8)}
                        </DialogTitle>
                    </DialogHeader>
                    
                    {loadingCaptures ? (
                        <div className="flex justify-center p-8">
                            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                        </div>
                    ) : captures.length === 0 ? (
                        <div className="text-center text-muted-foreground p-8">
                            No high-quality captures found for this identity.
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
                            {captures.map((cap, idx) => (
                                <div key={idx} className="bg-muted rounded-lg overflow-hidden border">
                                    <div className="aspect-[9/16] relative bg-black/5">
                                        <img 
                                            src={`data:image/jpeg;base64,${cap.image_b64}`}
                                            alt={`Capture ${idx}`}
                                            className="w-full h-full object-contain"
                                        />
                                        <div className="absolute top-2 right-2 flex flex-col items-end gap-1">
                                            <Badge variant={cap.quality_score > 70 ? "default" : "secondary"}>
                                                Q: {cap.quality_score.toFixed(0)}
                                            </Badge>
                                            <Badge variant="outline" className="bg-background/80">
                                                {cap.pose}
                                            </Badge>
                                        </div>
                                    </div>
                                    <div className="p-2 text-xs text-center text-muted-foreground">
                                        {cap.timestamp ? new Date(cap.timestamp).toLocaleTimeString() : 'Unknown time'}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </Card>
    );
}
