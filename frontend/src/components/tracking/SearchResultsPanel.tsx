'use client';

import { useState } from 'react';
import { 
    MapPin, 
    Clock, 
    Camera, 
    ChevronDown, 
    ChevronRight,
    Route,
    Timer,
    User,
    X,
    Maximize2
} from 'lucide-react';
import type { SearchResult, TrackPathPoint, CameraSequenceItem, Transition } from '@/lib/api';

interface SearchResultsPanelProps {
    results: SearchResult[];
    selectedResult: SearchResult | null;
    onSelect: (result: SearchResult) => void;
    onClose?: () => void;
    onMaximize?: () => void;
}

// Format time duration in human readable format
const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.round((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
};

// Format timestamp to readable time
const formatTime = (isoString: string): string => {
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit',
            hour12: true 
        });
    } catch {
        return 'Unknown';
    }
};

// Format date
const formatDate = (isoString: string): string => {
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            year: 'numeric'
        });
    } catch {
        return '';
    }
};

export default function SearchResultsPanel({
    results,
    selectedResult,
    onSelect,
    onClose,
    onMaximize
}: SearchResultsPanelProps) {
    const [expandedSection, setExpandedSection] = useState<'timeline' | 'transitions' | null>('timeline');

    if (results.length === 0) return null;

    const toggleSection = (section: 'timeline' | 'transitions') => {
        setExpandedSection(expandedSection === section ? null : section);
    };

    return (
        <div className="absolute top-4 right-4 z-[5000] w-80 bg-zinc-900/95 backdrop-blur-xl shadow-2xl rounded-2xl border border-zinc-700/50 max-h-[calc(100%-2rem)] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="p-4 bg-gradient-to-r from-blue-600/20 to-purple-600/20 border-b border-zinc-700/50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                            <User className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white">Search Results</h3>
                            <p className="text-xs text-zinc-400">{results.length} match{results.length !== 1 ? 'es' : ''} found</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-1">
                        {onMaximize && (
                            <button 
                                onClick={onMaximize}
                                className="p-2 rounded-lg hover:bg-zinc-700/50 transition-colors text-zinc-400 hover:text-white"
                            >
                                <Maximize2 className="w-4 h-4" />
                            </button>
                        )}
                        {onClose && (
                            <button 
                                onClick={onClose}
                                className="p-2 rounded-lg hover:bg-zinc-700/50 transition-colors text-zinc-400 hover:text-white"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Results List */}
            <div className="p-3 border-b border-zinc-700/50 max-h-40 overflow-y-auto">
                <div className="space-y-2">
                    {results.map((result, idx) => (
                        <button
                            key={result.track.id}
                            onClick={() => onSelect(result)}
                            className={`w-full p-3 rounded-xl text-left transition-all ${
                                selectedResult?.track.id === result.track.id
                                    ? 'bg-gradient-to-r from-blue-600/30 to-purple-600/30 border border-blue-500/50 shadow-lg shadow-blue-500/10'
                                    : 'bg-zinc-800/50 hover:bg-zinc-700/50 border border-transparent'
                            }`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2">
                                    <span className="text-xs font-medium text-zinc-400">#{idx + 1}</span>
                                    <span className="font-mono text-sm text-white">
                                        {result.track.id.slice(0, 8)}...
                                    </span>
                                </div>
                                <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                                    result.score >= 0.7 
                                        ? 'bg-green-500/20 text-green-400' 
                                        : result.score >= 0.5 
                                            ? 'bg-yellow-500/20 text-yellow-400'
                                            : 'bg-zinc-600/50 text-zinc-400'
                                }`}>
                                    {(result.score * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-zinc-400">
                                <span className="flex items-center gap-1">
                                    <Camera className="w-3 h-3" />
                                    {result.path_points?.length || 0} cameras
                                </span>
                                <span className="flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {formatTime(result.track.last_seen)}
                                </span>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Selected Result Details */}
            {selectedResult && (
                <div className="flex-1 overflow-y-auto">
                    {/* Quick Stats */}
                    <div className="p-4 grid grid-cols-2 gap-3">
                        <div className="bg-zinc-800/50 rounded-xl p-3">
                            <div className="text-xs text-zinc-400 mb-1">First Seen</div>
                            <div className="text-sm font-medium text-white">
                                {selectedResult.track.first_seen 
                                    ? formatTime(selectedResult.track.first_seen)
                                    : 'N/A'}
                            </div>
                            <div className="text-xs text-zinc-500">
                                {selectedResult.track.first_seen && formatDate(selectedResult.track.first_seen)}
                            </div>
                        </div>
                        <div className="bg-zinc-800/50 rounded-xl p-3">
                            <div className="text-xs text-zinc-400 mb-1">Last Seen</div>
                            <div className="text-sm font-medium text-white">
                                {formatTime(selectedResult.track.last_seen)}
                            </div>
                            <div className="text-xs text-zinc-500">
                                {formatDate(selectedResult.track.last_seen)}
                            </div>
                        </div>
                    </div>

                    {/* Camera Journey Section */}
                    <div className="px-4 pb-4">
                        <button
                            onClick={() => toggleSection('timeline')}
                            className="w-full flex items-center justify-between p-3 bg-zinc-800/50 rounded-xl hover:bg-zinc-700/50 transition-colors mb-2"
                        >
                            <div className="flex items-center gap-2">
                                <Route className="w-4 h-4 text-blue-400" />
                                <span className="font-medium text-white">Camera Journey</span>
                            </div>
                            {expandedSection === 'timeline' 
                                ? <ChevronDown className="w-4 h-4 text-zinc-400" />
                                : <ChevronRight className="w-4 h-4 text-zinc-400" />
                            }
                        </button>
                        
                        {expandedSection === 'timeline' && selectedResult.path_points?.length > 0 && (
                            <div className="relative pl-6 py-2">
                                {/* Vertical line */}
                                <div className="absolute left-3 top-4 bottom-4 w-0.5 bg-gradient-to-b from-green-500 via-blue-500 to-pink-500 rounded-full" />
                                
                                {selectedResult.path_points.map((point, idx) => (
                                    <div key={`${point.camera_id}-${idx}`} className="relative mb-4 last:mb-0">
                                        {/* Node */}
                                        <div className={`absolute -left-3 w-3 h-3 rounded-full border-2 ${
                                            idx === 0 
                                                ? 'bg-green-500 border-green-400' 
                                                : idx === selectedResult.path_points.length - 1 
                                                    ? 'bg-pink-500 border-pink-400'
                                                    : 'bg-blue-500 border-blue-400'
                                        }`} />
                                        
                                        {/* Card */}
                                        <div className="ml-4 p-3 bg-zinc-800/80 rounded-xl border border-zinc-700/50">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="font-medium text-white text-sm">
                                                    {point.name || `Camera ${point.camera_id}`}
                                                </span>
                                                {idx === 0 && (
                                                    <span className="text-[10px] px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full">
                                                        ENTRY
                                                    </span>
                                                )}
                                                {idx === selectedResult.path_points.length - 1 && idx !== 0 && (
                                                    <span className="text-[10px] px-2 py-0.5 bg-pink-500/20 text-pink-400 rounded-full">
                                                        LAST
                                                    </span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2 text-xs text-zinc-400">
                                                <MapPin className="w-3 h-3" />
                                                <span>{point.latitude.toFixed(4)}, {point.longitude.toFixed(4)}</span>
                                            </div>
                                        </div>
                                        
                                        {/* Transition indicator */}
                                        {idx < selectedResult.path_points.length - 1 && selectedResult.transitions?.[idx] && (
                                            <div className="ml-4 mt-2 mb-2 flex items-center gap-2 text-xs text-zinc-500">
                                                <Timer className="w-3 h-3" />
                                                <span>
                                                    {formatDuration(selectedResult.transitions[idx].time_at_from)} at camera
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Transitions Section */}
                    {selectedResult.transitions && selectedResult.transitions.length > 0 && (
                        <div className="px-4 pb-4">
                            <button
                                onClick={() => toggleSection('transitions')}
                                className="w-full flex items-center justify-between p-3 bg-zinc-800/50 rounded-xl hover:bg-zinc-700/50 transition-colors mb-2"
                            >
                                <div className="flex items-center gap-2">
                                    <Timer className="w-4 h-4 text-purple-400" />
                                    <span className="font-medium text-white">Transition Details</span>
                                    <span className="text-xs text-zinc-500">
                                        ({selectedResult.transitions.length})
                                    </span>
                                </div>
                                {expandedSection === 'transitions' 
                                    ? <ChevronDown className="w-4 h-4 text-zinc-400" />
                                    : <ChevronRight className="w-4 h-4 text-zinc-400" />
                                }
                            </button>
                            
                            {expandedSection === 'transitions' && (
                                <div className="space-y-2">
                                    {selectedResult.transitions.map((t, idx) => (
                                        <div 
                                            key={idx}
                                            className="p-3 bg-zinc-800/50 rounded-xl border border-zinc-700/50"
                                        >
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="flex items-center gap-2">
                                                    <span className="w-6 h-6 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-xs font-medium">
                                                        {t.from_camera}
                                                    </span>
                                                    <ChevronRight className="w-4 h-4 text-zinc-500" />
                                                    <span className="w-6 h-6 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-medium">
                                                        {t.to_camera}
                                                    </span>
                                                </div>
                                                <span className="text-xs text-zinc-400">
                                                    {formatTime(t.transition_time)}
                                                </span>
                                            </div>
                                            <div className="text-xs text-zinc-500">
                                                Time at Camera {t.from_camera}: {formatDuration(t.time_at_from)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
