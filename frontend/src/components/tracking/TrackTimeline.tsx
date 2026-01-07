'use client';

import { useState, useEffect } from 'react';
import { Clock, Camera, MapPin, ChevronRight } from 'lucide-react';

interface TimelineEvent {
    camera_id: number;
    camera_name: string;
    timestamp: string;
    thumbnail?: string;
    latitude?: number;
    longitude?: number;
}

interface TrackTimelineProps {
    globalId: string;
    events?: TimelineEvent[];
    onEventClick?: (event: TimelineEvent, index: number) => void;
}

/**
 * Premium Track Timeline component for visualizing camera transitions.
 * 
 * Shows a vertical timeline of camera appearances with timestamps and thumbnails.
 */
export default function TrackTimeline({ globalId, events = [], onEventClick }: TrackTimelineProps) {
    const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

    if (events.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-48 text-zinc-400">
                <Clock className="w-12 h-12 mb-3 opacity-50" />
                <p className="text-sm">No camera transitions recorded</p>
                <p className="text-xs opacity-75">Track ID: {globalId}</p>
            </div>
        );
    }

    return (
        <div className="relative w-full">
            {/* Header */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-zinc-700/50">
                <div>
                    <h3 className="text-lg font-semibold text-white">Track Timeline</h3>
                    <p className="text-xs text-zinc-400">ID: {globalId.slice(0, 8)}...</p>
                </div>
                <span className="px-3 py-1 bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30 rounded-full text-xs text-blue-300">
                    {events.length} camera{events.length !== 1 ? 's' : ''} visited
                </span>
            </div>

            {/* Timeline */}
            <div className="relative pl-8">
                {/* Vertical line */}
                <div className="absolute left-3 top-2 bottom-2 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500 rounded-full" />

                {events.map((event, index) => (
                    <div
                        key={`${event.camera_id}-${index}`}
                        className="relative mb-6 last:mb-0 group"
                    >
                        {/* Node dot */}
                        <div className={`absolute -left-5 w-4 h-4 rounded-full border-2 transition-all duration-300 ${
                            index === 0 
                                ? 'bg-green-500 border-green-400 shadow-lg shadow-green-500/50' 
                                : index === events.length - 1 
                                    ? 'bg-pink-500 border-pink-400 shadow-lg shadow-pink-500/50'
                                    : 'bg-blue-500 border-blue-400'
                        }`} />

                        {/* Event card */}
                        <div 
                            className={`ml-4 p-4 rounded-xl transition-all duration-300 cursor-pointer
                                bg-gradient-to-br from-zinc-800/80 to-zinc-900/80 
                                border border-zinc-700/50 hover:border-blue-500/50
                                hover:shadow-lg hover:shadow-blue-500/10
                                ${expandedIndex === index ? 'border-blue-500/50 shadow-lg shadow-blue-500/10' : ''}
                            `}
                            onClick={() => {
                                setExpandedIndex(expandedIndex === index ? null : index);
                                onEventClick?.(event, index);
                            }}
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex-1">
                                    {/* Camera info */}
                                    <div className="flex items-center gap-2 mb-2">
                                        <Camera className="w-4 h-4 text-blue-400" />
                                        <span className="font-medium text-white">
                                            {event.camera_name || `Camera ${event.camera_id}`}
                                        </span>
                                        {index === 0 && (
                                            <span className="px-2 py-0.5 text-[10px] bg-green-500/20 text-green-400 rounded-full">
                                                ENTRY
                                            </span>
                                        )}
                                        {index === events.length - 1 && index !== 0 && (
                                            <span className="px-2 py-0.5 text-[10px] bg-pink-500/20 text-pink-400 rounded-full">
                                                LAST SEEN
                                            </span>
                                        )}
                                    </div>

                                    {/* Timestamp */}
                                    <div className="flex items-center gap-2 text-sm text-zinc-400">
                                        <Clock className="w-3 h-3" />
                                        <span>{new Date(event.timestamp).toLocaleString()}</span>
                                    </div>

                                    {/* Location (expanded) */}
                                    {expandedIndex === index && event.latitude && event.longitude && (
                                        <div className="flex items-center gap-2 mt-2 text-sm text-zinc-400">
                                            <MapPin className="w-3 h-3" />
                                            <span>{event.latitude.toFixed(4)}, {event.longitude.toFixed(4)}</span>
                                        </div>
                                    )}
                                </div>

                                {/* Thumbnail */}
                                {event.thumbnail && (
                                    <div className="ml-4 w-16 h-20 rounded-lg overflow-hidden border border-zinc-600/50 flex-shrink-0">
                                        <img 
                                            src={`data:image/jpeg;base64,${event.thumbnail}`}
                                            alt={`Camera ${event.camera_id}`}
                                            className="w-full h-full object-cover"
                                        />
                                    </div>
                                )}

                                {/* Expand indicator */}
                                <ChevronRight className={`w-4 h-4 text-zinc-500 transition-transform ml-2 ${
                                    expandedIndex === index ? 'rotate-90' : ''
                                }`} />
                            </div>

                            {/* Transition indicator */}
                            {index < events.length - 1 && (
                                <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 text-[10px] text-zinc-500">
                                    ↓
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
