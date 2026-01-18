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

// Construct WebSocket URL from environment variable
const getDefaultWsUrl = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    // Convert HTTP(S) to WS(S)
    const wsProtocol = apiUrl.startsWith('https://') ? 'wss://' : 'ws://';
    const baseUrl = apiUrl.replace(/^https?:\/\//, '');
    return `${wsProtocol}${baseUrl}/api/v1/ws/tracks`;
};

/**
 * Hook for real-time track path updates via WebSocket.
 * 
 * Subscribes to track_path_update events and maintains path state.
 */
export function useTrackPath({ wsUrl = getDefaultWsUrl(), globalId = null }: UseTrackPathOptions = {}) {
    const [pathPoints, setPathPoints] = useState<PathPoint[]>([]);
    const [cameraSequence, setCameraSequence] = useState<number[]>([]);
    const [allTracks, setAllTracks] = useState<Map<string, TrackPathUpdate>>(new Map());
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectTimeout: ReturnType<typeof setTimeout>;
        let isMounted = true;
        let retryCount = 0;
        const MAX_RETRIES = 5;
        const BASE_DELAY = 1000;

        const connect = () => {
            if (!isMounted) return;

            try {
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    if (!isMounted) return;
                    console.log('Track path WebSocket connected');
                    setConnected(true);
                    retryCount = 0; // Reset retries on successful connection
                };

                ws.onmessage = (event) => {
                    if (!isMounted) return;
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

                ws.onclose = (event) => {
                    if (!isMounted) return;
                    setConnected(false);
                    
                    // Only reconnect if not closed cleanly
                    if (!event.wasClean) {
                        const delay = Math.min(BASE_DELAY * Math.pow(2, retryCount), 10000);
                        console.log(`WebSocket disconnected. Reconnecting in ${delay}ms... (Attempt ${retryCount + 1}/${MAX_RETRIES})`);
                        
                        if (retryCount < MAX_RETRIES) {
                            retryCount++;
                            reconnectTimeout = setTimeout(connect, delay);
                        } else {
                            console.error('Max WebSocket reconnection attempts reached');
                        }
                    }
                };

                ws.onerror = (error) => {
                    // Start suppressing verbose errors after first failure to reduce console noise
                    if (retryCount === 0) {
                        console.warn('Track path WebSocket connection error - will attempt to reconnect');
                    }
                    // Do not log the full error object as it is usually empty in browsers
                };

            } catch (err) {
                console.error('WebSocket connection initialization error:', err);
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    reconnectTimeout = setTimeout(connect, 3000);
                }
            }
        };

        connect();

        return () => {
            isMounted = false;
            if (ws) {
                ws.onclose = null; // Prevent reconnection trigger on cleanup
                ws.close();
            }
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }
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
