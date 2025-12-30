"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Users, Trash2, RefreshCw } from "lucide-react";

interface PersonEntry {
    global_id: string;
    last_camera_id: number;
    last_seen: string;
    appearance_count: number;
    thumbnail: string | null;
}

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

    const fetchGallery = async () => {
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
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                            {persons.map((person) => (
                                <div 
                                    key={person.global_id}
                                    className="border rounded-lg p-2 flex flex-col items-center gap-2 bg-card hover:bg-accent transition-colors"
                                >
                                    {/* Thumbnail */}
                                    <div className="w-16 h-32 bg-muted rounded overflow-hidden">
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
                                        <Badge variant="outline" className="text-xs mb-1">
                                            {person.global_id.slice(0, 8)}
                                        </Badge>
                                        <div className="text-xs text-muted-foreground">
                                            Cam {person.last_camera_id}
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {formatTimeAgo(person.last_seen)}
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {person.appearance_count} obs
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
