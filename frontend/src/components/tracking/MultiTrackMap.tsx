'use client';

import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useTrackPath } from '@/hooks/useTrackPath';

// Generate distinct colors for multiple tracks
const TRACK_COLORS = [
    '#3b82f6', // Blue
    '#ef4444', // Red  
    '#22c55e', // Green
    '#f59e0b', // Amber
    '#8b5cf6', // Purple
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#f97316', // Orange
];

const createCustomIcon = (color: string, number?: number | string) => {
    return L.divIcon({
        className: 'custom-map-marker',
        html: `
            <div style="
                background-color: ${color};
                width: 30px;
                height: 30px;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 12px;
            ">
                ${number || ''}
            </div>
        `,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
    });
};

const cameraIcon = (isActive: boolean) => createCustomIcon(isActive ? '#22c55e' : '#ef4444');

interface PathPoint {
    camera_id: number;
    latitude: number;
    longitude: number;
    name: string;
}

interface TrackPath {
    globalId: string;
    pathPoints: PathPoint[];
    color?: string;
}

interface MultiTrackMapProps {
    sources: any[];
    activeTracks?: TrackPath[];
    selectedTrackId?: string | null;
    center?: [number, number];
    zoom?: number;
    onMapClick?: (lat: number, lng: number) => void;
    onTrackSelect?: (globalId: string) => void;
}

// Route cache - persists routes between camera pairs to avoid repeated API calls
const routeCache = new Map<string, [number, number][]>();

// Generate cache key for a route segment
const getCacheKey = (start: [number, number], end: [number, number]): string => {
    return `${start[0].toFixed(5)},${start[1].toFixed(5)}-${end[0].toFixed(5)},${end[1].toFixed(5)}`;
};

// Load cached routes from localStorage on startup
const loadCachedRoutes = () => {
    try {
        const cached = localStorage.getItem('route_cache');
        if (cached) {
            const entries = JSON.parse(cached);
            entries.forEach(([key, value]: [string, [number, number][]]) => {
                routeCache.set(key, value);
            });
            console.log(`Loaded ${routeCache.size} cached routes`);
        }
    } catch (e) {
        console.warn('Failed to load route cache:', e);
    }
};

// Save routes to localStorage
const saveCachedRoutes = () => {
    try {
        const entries = Array.from(routeCache.entries());
        localStorage.setItem('route_cache', JSON.stringify(entries));
    } catch (e) {
        console.warn('Failed to save route cache:', e);
    }
};

// Initialize cache
if (typeof window !== 'undefined') {
    loadCachedRoutes();
}

// Fast route fetching with cache and fallback
const fetchRoute = async (start: [number, number], end: [number, number]): Promise<[number, number][]> => {
    const cacheKey = getCacheKey(start, end);
    
    // Check cache first
    if (routeCache.has(cacheKey)) {
        return routeCache.get(cacheKey)!;
    }
    
    // Try FOSSGIS OSRM instance first (Often faster/less rate-limited than main demo)
    try {
        const response = await fetch(
            `https://routing.openstreetmap.de/routed-foot/route/v1/foot/${start[1]},${start[0]};${end[1]},${end[0]}?overview=full&geometries=geojson`,
            { signal: AbortSignal.timeout(3000) }
        );
        if (response.ok) {
            const data = await response.json();
            if (data.routes?.[0]?.geometry?.coordinates) {
                const route = data.routes[0].geometry.coordinates.map((coord: number[]) => 
                    [coord[1], coord[0]] as [number, number]
                );
                routeCache.set(cacheKey, route);
                saveCachedRoutes();
                return route;
            }
        }
    } catch (e) {
        // FOSSGIS failed, try main OSRM
    }
    
    // Fallback to Main OSRM Demo Server
    try {
        const response = await fetch(
            `https://router.project-osrm.org/route/v1/foot/${start[1]},${start[0]};${end[1]},${end[0]}?overview=full&geometries=geojson`,
            { signal: AbortSignal.timeout(5000) }
        );
        const data = await response.json();
        if (data.routes?.[0]?.geometry?.coordinates) {
            const route = data.routes[0].geometry.coordinates.map((coord: number[]) => 
                [coord[1], coord[0]] as [number, number]
            );
            routeCache.set(cacheKey, route);
            saveCachedRoutes();
            return route;
        }
    } catch (error) {
        console.warn("All route fetch providers failed, using fallback");
    }
    
    // Final fallback: curved line (Bezier approximation)
    const midLat = (start[0] + end[0]) / 2;
    const midLng = (start[1] + end[1]) / 2;
    const offset = Math.min(Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1])) * 0.3;
    return [
        start,
        [midLat + offset, midLng],
        end
    ];
};

function MapEvents({ onMapClick }: { onMapClick?: (lat: number, lng: number) => void }) {
    useMapEvents({
        click(e) {
            if (onMapClick) {
                onMapClick(e.latlng.lat, e.latlng.lng);
            }
        },
    });
    return null;
}

/**
 * Multi-person tracking map with color-coded paths.
 * 
 * Real-time updates via WebSocket when tracks move between cameras.
 */
export default function MultiTrackMap({ 
    sources, 
    activeTracks = [], 
    selectedTrackId,
    center = [12.9716, 77.5946], 
    zoom = 13, 
    onMapClick,
    onTrackSelect 
}: MultiTrackMapProps) {
    const [interpolatedPaths, setInterpolatedPaths] = useState<Map<string, [number, number][]>>(new Map());
    const [isLoading, setIsLoading] = useState(false);
    const { allTracks, connected } = useTrackPath({});

    // Merge WebSocket updates with provided tracks
    const mergedTracks = [...activeTracks];
    allTracks.forEach((update, globalId) => {
        if (!mergedTracks.find(t => t.globalId === globalId)) {
            mergedTracks.push({
                globalId,
                pathPoints: update.path_points,
            });
        }
    });

    // Assign colors to tracks
    const tracksWithColors = mergedTracks.map((track, idx) => ({
        ...track,
        color: track.color || TRACK_COLORS[idx % TRACK_COLORS.length],
    }));

    // Interpolate routes for each track - PARALLEL fetching for speed
    useEffect(() => {
        const interpolateAll = async () => {
            if (tracksWithColors.length === 0) return;
            
            setIsLoading(true);
            const newPaths = new Map<string, [number, number][]>();
            
            // Fetch all routes in parallel
            await Promise.all(tracksWithColors.map(async (track) => {
                if (track.pathPoints.length < 2) {
                    if (track.pathPoints.length === 1) {
                        newPaths.set(track.globalId, [[track.pathPoints[0].latitude, track.pathPoints[0].longitude]]);
                    }
                    return;
                }

                // Fetch all segments in parallel
                const segmentPromises = [];
                for (let i = 0; i < track.pathPoints.length - 1; i++) {
                    const start = track.pathPoints[i];
                    const end = track.pathPoints[i + 1];
                    segmentPromises.push(
                        fetchRoute([start.latitude, start.longitude], [end.latitude, end.longitude])
                    );
                }
                
                const segments = await Promise.all(segmentPromises);
                const fullPath = segments.flat();
                newPaths.set(track.globalId, fullPath);
            }));
            
            setInterpolatedPaths(newPaths);
            setIsLoading(false);
        };

        interpolateAll();
    }, [tracksWithColors.length]);

    return (
        <div className="relative h-full w-full">
            {/* Connection status indicator */}
            <div className={`absolute top-3 right-3 z-[1000] flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                connected 
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                    : 'bg-zinc-700/50 text-zinc-400 border border-zinc-600/30'
            }`}>
                <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-zinc-500'}`} />
                {connected ? 'Live' : 'Offline'}
            </div>

            {/* Track legend */}
            {tracksWithColors.length > 0 && (
                <div className="absolute bottom-3 left-3 z-[1000] bg-zinc-900/90 backdrop-blur-sm rounded-lg border border-zinc-700/50 p-3 max-w-xs">
                    <h4 className="text-xs font-semibold text-zinc-400 mb-2">Tracked Persons</h4>
                    <div className="space-y-1.5 max-h-32 overflow-y-auto">
                        {tracksWithColors.map(track => (
                            <div 
                                key={track.globalId}
                                className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer transition-colors ${
                                    selectedTrackId === track.globalId 
                                        ? 'bg-zinc-700/50' 
                                        : 'hover:bg-zinc-800/50'
                                }`}
                                onClick={() => onTrackSelect?.(track.globalId)}
                            >
                                <div 
                                    className="w-3 h-3 rounded-full border border-white/30"
                                    style={{ backgroundColor: track.color }}
                                />
                                <span className="text-xs text-white font-mono">
                                    {track.globalId.slice(0, 8)}...
                                </span>
                                <span className="text-xs text-zinc-500">
                                    ({track.pathPoints.length} pts)
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} className="h-full w-full rounded-lg" style={{ minHeight: '400px' }}>
                <MapEvents onMapClick={onMapClick} />
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                
                {/* Camera Markers */}
                {sources.map(source => (
                    source.latitude && source.longitude ? (
                        <Marker 
                            key={source.id} 
                            position={[source.latitude, source.longitude]}
                            icon={cameraIcon(source.is_active)}
                        >
                            <Popup>
                                <strong>{source.name}</strong><br />
                                Type: {source.source_type}<br />
                                Status: {source.is_active ? 'Active' : 'Inactive'}
                            </Popup>
                        </Marker>
                    ) : null
                ))}

                {/* Track Paths */}
                {tracksWithColors.map(track => {
                    const path = interpolatedPaths.get(track.globalId) || [];
                    if (path.length === 0) return null;

                    const isSelected = selectedTrackId === track.globalId;
                    
                    return (
                        <Polyline 
                            key={track.globalId}
                            positions={path}
                            color={track.color}
                            weight={isSelected ? 6 : 4}
                            opacity={isSelected ? 1 : 0.7}
                            dashArray={isSelected ? undefined : '5, 10'}
                        />
                    );
                })}

                {/* Path Point Markers */}
                {tracksWithColors.map(track => 
                    track.pathPoints.map((point, index) => (
                        <Marker 
                            key={`${track.globalId}-${index}`}
                            position={[point.latitude, point.longitude]}
                            icon={createCustomIcon(track.color!, index + 1)}
                            zIndexOffset={selectedTrackId === track.globalId ? 2000 : 1000}
                        >
                            <Popup>
                                <strong>Step {index + 1}</strong><br/>
                                {point.name}<br/>
                                Camera ID: {point.camera_id}
                            </Popup>
                        </Marker>
                    ))
                )}
            </MapContainer>
        </div>
    );
}
