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
import type { SearchResult } from '@/lib/api';

interface SearchResultsPanelProps {
    results: SearchResult[];
    selectedResult: SearchResult | null;
    onSelect: (result: SearchResult) => void;
    onClose?: () => void;
    onMaximize?: () => void;
}

const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.round((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
};

const formatTime = (isoString: string): string => {
    try { return new Date(isoString).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }); } catch { return 'Unknown'; }
};

const formatDate = (isoString: string): string => {
    try { return new Date(isoString).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }); } catch { return ''; }
};

export default function SearchResultsPanel({
    results,
    selectedResult,
    onSelect,
    onClose,
    onMaximize
}: SearchResultsPanelProps) {
    const [expandedSection, setExpandedSection] = useState<'timeline' | 'transitions' | null>('timeline');
    const toggleSection = (section: 'timeline' | 'transitions') => setExpandedSection(expandedSection === section ? null : section);

    if (results.length === 0) return null;

    return (
        <div className="absolute top-4 right-4 z-[5000] w-80 bg-black border border-[#333] shadow-2xl max-h-[calc(100%-2rem)] flex flex-col font-mono">
            {/* Header */}
            <div className="p-3 border-b border-[#333] bg-[#050505]">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 flex items-center justify-center bg-[#111] border border-[#333]">
                            <User className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h3 className="text-[10px] uppercase tracking-widest text-[#888]">SEARCH_RESULTS</h3>
                            <p className="text-xs font-bold text-white">{results.length} MATCH{results.length !== 1 ? 'ES' : ''}</p>
                        </div>
                    </div>
                    <div className="flex gap-1">
                        {onMaximize && (
                            <button onClick={onMaximize} className="p-1 hover:bg-white hover:text-black text-[#666] transition-colors border border-transparent hover:border-[#333]">
                                <Maximize2 className="w-4 h-4" />
                            </button>
                        )}
                        {onClose && (
                            <button onClick={onClose} className="p-1 hover:bg-red-900 hover:text-white text-[#666] transition-colors">
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Results List */}
            <div className="max-h-40 overflow-y-auto border-b border-[#333] bg-black">
                {results.map((result, idx) => (
                    <div
                        key={result.track.id}
                        onClick={() => onSelect(result)}
                        className={`p-3 cursor-pointer border-l-2 text-left transition-colors ${selectedResult?.track.id === result.track.id
                                ? 'bg-[#111] border-white'
                                : 'bg-black border-transparent hover:bg-[#0a0a0a]'
                            }`}
                    >
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] uppercase text-[#666]">{result.track.id.slice(0, 8)}...</span>
                            <span className={`text-[9px] px-1 font-bold ${result.score > 0.7 ? 'text-green-500' : 'text-yellow-500'}`}>
                                {(result.score * 100).toFixed(0)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-[#888]">
                            <span className="flex items-center gap-1"><Camera className="w-3 h-3" /> {result.path_points?.length || 0}</span>
                            <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {formatTime(result.track.last_seen)}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Details */}
            {selectedResult && (
                <div className="flex-1 overflow-y-auto bg-[#080808]">
                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-px bg-[#333] border-b border-[#333]">
                        <div className="bg-black p-3">
                            <div className="text-[9px] uppercase tracking-wider text-[#666] mb-1">First_Seen</div>
                            <div className="text-xs text-white">{selectedResult.track.first_seen ? formatTime(selectedResult.track.first_seen) : 'N/A'}</div>
                        </div>
                        <div className="bg-black p-3">
                            <div className="text-[9px] uppercase tracking-wider text-[#666] mb-1">Last_Seen</div>
                            <div className="text-xs text-white">{formatTime(selectedResult.track.last_seen)}</div>
                        </div>
                    </div>

                    {/* Timeline */}
                    <div className="p-3">
                        <button onClick={() => toggleSection('timeline')} className="w-full flex items-center justify-between py-2 border-b border-[#333] mb-2 hover:text-white text-[#888] transition-colors">
                            <span className="text-[10px] uppercase tracking-widest flex items-center gap-2"><Route className="w-3 h-3" /> Trajectory</span>
                            {expandedSection === 'timeline' ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                        </button>

                        {expandedSection === 'timeline' && selectedResult.path_points?.length > 0 && (
                            <div className="pl-4 border-l border-[#333] ml-2 space-y-4 py-2">
                                {selectedResult.path_points.map((point, idx) => (
                                    <div key={idx} className="relative">
                                        <div className="absolute -left-[21px] top-1 w-2.5 h-2.5 bg-black border border-white rounded-full z-10" />
                                        <div className="text-xs text-white mb-0.5">{point.name || `CAM_${point.camera_id}`}</div>
                                        <div className="text-[10px] text-[#666] flex items-center gap-1">
                                            <MapPin className="w-3 h-3" /> {point.latitude.toFixed(4)}, {point.longitude.toFixed(4)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
