'use client';

import { useState, useEffect, useCallback } from 'react';

interface PathPoint {
    camera_id: number;
    latitude: number;
    longitude: number;
    name: string;
}

interface TrackPathUpdate {
    global_track_id: string;
    from_camera_id: number;
    to_camera_id: number;
    camera_sequence: number[];
    path_points: PathPoint[];
}

interface UseTrackPathOptions {
    wsUrl?: string;
    globalId?: string | null;
}

/**
 * Hook for real-time track path updates via WebSocket.
 * 
 * Subscribes to track_path_update events and maintains path state.
 */
export function useTrackPath({ wsUrl = 'ws://localhost:8000/api/v1/ws/tracks', globalId = null }: UseTrackPathOptions = {}) {
    const [pathPoints, setPathPoints] = useState<PathPoint[]>([]);
    const [cameraSequence, setCameraSequence] = useState<number[]>([]);
    const [allTracks, setAllTracks] = useState<Map<string, TrackPathUpdate>>(new Map());
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('Track path WebSocket connected');
            setConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                if (message.type === 'track_path_update') {
                    const update: TrackPathUpdate = message.data;
                    
                    // Update all tracks map
                    setAllTracks(prev => {
                        const newMap = new Map(prev);
                        newMap.set(update.global_track_id, update);
                        return newMap;
                    });
                    
                    // If watching specific track, update path
                    if (globalId && update.global_track_id === globalId) {
                        setPathPoints(update.path_points);
                        setCameraSequence(update.camera_sequence);
                    }
                }
            } catch (e) {
                console.error('WebSocket message parse error:', e);
            }
        };

        ws.onclose = () => {
            console.log('Track path WebSocket disconnected');
            setConnected(false);
        };

        ws.onerror = (error) => {
            console.error('Track path WebSocket error:', error);
        };

        return () => {
            ws.close();
        };
    }, [wsUrl, globalId]);

    // Manually set path for a specific track (e.g., from search results)
    const setTrackPath = useCallback((trackId: string, points: PathPoint[]) => {
        setPathPoints(points);
    }, []);

    // Get path for any tracked person
    const getTrackPath = useCallback((trackId: string): TrackPathUpdate | undefined => {
        return allTracks.get(trackId);
    }, [allTracks]);

    return {
        pathPoints,
        cameraSequence,
        allTracks,
        connected,
        setTrackPath,
        getTrackPath,
    };
}
