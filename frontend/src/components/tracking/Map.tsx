'use client';

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

const createCustomIcon = (color: string, number?: number) => {
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
                font-size: 14px;
            ">
                ${number || ''}
            </div>
        `,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
    });
};

const activeIcon = createCustomIcon('#22c55e'); // Green
const inactiveIcon = createCustomIcon('#ef4444'); // Red


interface MapProps {
    sources: any[];
    pathPoints?: any[]; // Points for the tracked path
    center?: [number, number];
    zoom?: number;
    onMapClick?: (lat: number, lng: number) => void;
}

function ChangeView({ center, zoom, pathPoints }: { center: [number, number], zoom: number, pathPoints?: any[] }) {
    const map = useMap();
    const [lat, lng] = center;

    useEffect(() => {
        if (pathPoints && pathPoints.length > 0) {
            const bounds = L.latLngBounds(pathPoints.map(p => [p.latitude, p.longitude]));
            map.fitBounds(bounds, { padding: [50, 50] });
        } else {
            map.setView([lat, lng], zoom);
        }
    }, [lat, lng, zoom, map, pathPoints]); 
    return null;
}

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

const fetchRoute = async (start: [number, number], end: [number, number]) => {
    try {
        const response = await fetch(
            `http://router.project-osrm.org/route/v1/foot/${start[1]},${start[0]};${end[1]},${end[0]}?overview=full&geometries=geojson`
        );
        const data = await response.json();
        if (data.routes && data.routes.length > 0) {
            return data.routes[0].geometry.coordinates.map((coord: number[]) => [coord[1], coord[0]]); // Swap to [lat, lng]
        }
    } catch (error) {
        console.error("Error fetching route:", error);
    }
    return [start, end]; // Fallback to straight line
};

export default function Map({ sources, pathPoints, center = [12.9716, 77.5946], zoom = 13, onMapClick }: MapProps) {
    const [interpolatedPath, setInterpolatedPath] = useState<[number, number][]>([]);

    useEffect(() => {
        const updatePath = async () => {
            if (!pathPoints || pathPoints.length < 2) {
                setInterpolatedPath([]);
                return;
            }

            let fullPath: [number, number][] = [];
            
            for (let i = 0; i < pathPoints.length - 1; i++) {
                const start = pathPoints[i];
                const end = pathPoints[i+1];
                
                // Add start point
                if (i === 0) fullPath.push([start.latitude, start.longitude]);

                const segment = await fetchRoute(
                    [start.latitude, start.longitude],
                    [end.latitude, end.longitude]
                );

                // Add segment points (excluding first since it's same as last)
                // Actually OSRM returns start->end inclusive.
                // We should append all except the first one to avoid dupe, or just append all and let Polyline handle it.
                // Better: Append all.
                fullPath = [...fullPath, ...segment];
            }
            
            setInterpolatedPath(fullPath);
        };

        updatePath();
    }, [pathPoints]); // Re-run when pathPoints changes

    return (
        <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} className="h-full w-full rounded-lg" style={{ minHeight: '400px' }}>
            <ChangeView center={center} zoom={zoom} pathPoints={pathPoints} />
            <MapEvents onMapClick={onMapClick} />
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            
            {/* Camera Markers */}
            {sources.map(source => (
                source.latitude && source.longitude ? (
                    <Marker 
                        key={source.id} 
                        position={[source.latitude, source.longitude]}
                        icon={source.is_active ? activeIcon : inactiveIcon}
                    >
                        <Popup>
                            <strong>{source.name}</strong><br />
                            Type: {source.source_type}<br />
                            Status: {source.is_active ? 'Active' : 'Inactive'}
                        </Popup>
                    </Marker>
                ) : null
            ))}

            {/* Track Path (Interpolated) */}
            {interpolatedPath.length > 0 && (
                <Polyline 
                    positions={interpolatedPath} 
                    color="#3b82f6" 
                    weight={6} 
                    opacity={0.8}
                />
            )}
            
            {/* Path Points (Sequence Markers) */}
            {pathPoints && pathPoints.map((point, index) => (
                <Marker 
                    key={`path-${index}`}
                    position={[point.latitude, point.longitude]}
                    icon={createCustomIcon('#3b82f6', index + 1)} // Blue with number
                    zIndexOffset={1000 + index} // Ensure path markers are on top
                >
                    <Popup>
                        <strong>Step {index + 1}</strong><br/>
                        {point.name}<br/>
                        Camera ID: {point.camera_id}
                    </Popup>
                </Marker>
            ))}

        </MapContainer>
    );
}
