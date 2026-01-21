'use client';

import { useState, useEffect } from 'react';
import {
    MapPin,
    Clock,
    Camera,
    ChevronDown,
    ChevronRight,
    Route,
    User,
    X,
    Maximize2,
    Activity,
    Calendar,
    ArrowRight
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

const formatTime = (isoString?: string): string => {
    if (!isoString) return 'N/A';
    try { return new Date(isoString).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }); } catch { return 'Unknown'; }
};

const formatDate = (isoString?: string): string => {
    if (!isoString) return 'N/A';
    try { return new Date(isoString).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }); } catch { return ''; }
};

// Circular Progress Component for Match Score
const ScoreRing = ({ score }: { score: number }) => {
    const radius = 18;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score * circumference);
    const color = score > 0.8 ? 'text-emerald-500' : score > 0.6 ? 'text-yellow-500' : 'text-red-500';

    return (
        <div className="relative flex items-center justify-center w-12 h-12">
            <svg className="w-full h-full transform -rotate-90">
                <circle
                    cx="24"
                    cy="24"
                    r={radius}
                    stroke="currentColor"
                    strokeWidth="3"
                    fill="transparent"
                    className="text-white/10"
                />
                <circle
                    cx="24"
                    cy="24"
                    r={radius}
                    stroke="currentColor"
                    strokeWidth="3"
                    fill="transparent"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    className={`transition-all duration-1000 ease-out ${color}`}
                />
            </svg>
            <span className={`absolute text-[10px] font-bold ${color}`}>{(score * 100).toFixed(0)}%</span>
        </div>
    );
};

export default function SearchResultsPanel({
    results,
    selectedResult,
    onSelect,
    onClose,
    onMaximize
}: SearchResultsPanelProps) {
    const [activeTab, setActiveTab] = useState<'details' | 'timeline'>('details');

    // Auto-select first result if only one match
    useEffect(() => {
        if (results.length === 1 && !selectedResult) {
            onSelect(results[0]);
        }
    }, [results, selectedResult, onSelect]);

    if (results.length === 0) return null;

    // If no result is selected, show the list
    if (!selectedResult && results.length > 0) {
        return (
            <div className="absolute top-4 right-4 z-[10000] w-80 bg-black/95 backdrop-blur-md border border-white/10 shadow-2xl flex flex-col font-mono animate-in slide-in-from-right-10 duration-300">
                <div className="p-4 border-b border-white/10 bg-white/5 flex justify-between items-center">
                    <div>
                        <h3 className="text-xs font-bold text-white tracking-widest uppercase">Select Match</h3>
                        <p className="text-[10px] text-white/50">{results.length} Identity Matches Found</p>
                    </div>
                     <button onClick={onClose} className="p-1 hover:bg-white/10 rounded transition-colors text-white/70">
                        <X className="w-4 h-4" />
                    </button>
                </div>
                <div className="max-h-[60vh] overflow-y-auto p-2 space-y-2">
                    {results.map((result) => (
                        <div
                            key={result.track.id}
                            onClick={() => onSelect(result)}
                            className="bg-black border border-white/10 p-3 cursor-pointer hover:border-white/40 hover:bg-white/5 transition-all group relative overflow-hidden"
                        >
                            <div className="absolute top-0 right-0 p-2 opacity-50 text-[10px] font-bold tracking-widest text-white/50 group-hover:opacity-100 transition-opacity">
                                {(result.score * 100).toFixed(0)}% MATCH
                            </div>
                            <div className="text-xs text-white font-bold mb-1 font-mono tracking-wider">{result.track.id.split('-')[0]}...</div>
                            <div className="flex items-center gap-4 text-[10px] text-white/60">
                                <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {formatTime(result.track.last_seen)}</span>
                                <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> {result.path_points?.length || 0} Locs</span>
                            </div>
                            {/* Similarity Bar */}
                            <div className="mt-2 h-0.5 w-full bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-white/80" style={{ width: `${result.score * 100}%` }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (!selectedResult) return null;

    return (
        <div className="absolute top-4 right-4 z-[10000] w-96 bg-[#09090b] border border-[#27272a] shadow-2xl max-h-[calc(100%-2rem)] flex flex-col font-mono animate-in fade-in duration-300 zoom-in-95">
            {/* Header: Match Summary */}
            <div className="relative p-0 overflow-hidden">
                 <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 via-purple-500/5 to-transparent pointer-events-none" />
                 
                 <div className="relative p-4 flex items-start justify-between border-b border-[#27272a] bg-[#121214]">
                    <div className="flex items-start gap-3">
                        <ScoreRing score={selectedResult.score} />
                        <div>
                            <div className="flex items-center gap-2 mb-1">
                                <h2 className="text-sm font-bold text-white tracking-widest uppercase">ID: {selectedResult.track.id.split('-')[0]}</h2>
                                <span className="px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-[9px] font-bold uppercase tracking-wider border border-emerald-500/30">
                                    {selectedResult.track.status}
                                </span>
                            </div>
                            <div className="text-[10px] text-[#a1a1aa] flex flex-col gap-0.5">
                                <span className="flex items-center gap-1.5"><Calendar className="w-3 h-3" /> {formatDate(selectedResult.track.first_seen)}</span>
                                <span className="flex items-center gap-1.5"><Route className="w-3 h-3" /> Tracked across {new Set(selectedResult.track.camera_sequence).size} Cameras</span>
                            </div>
                        </div>
                    </div>
                    <div className="flex gap-1">
                         <button onClick={() => onSelect(null as any)} className="p-1.5 hover:bg-white/10 rounded text-[#a1a1aa] hover:text-white transition-colors" title="Back to results">
                            <ArrowRight className="w-4 h-4 rotate-180" />
                        </button>
                        {onMaximize && (
                            <button onClick={onMaximize} className="p-1.5 hover:bg-white/10 rounded text-[#a1a1aa] hover:text-white transition-colors" title="Focus View">
                                <Maximize2 className="w-4 h-4" />
                            </button>
                        )}
                        {onClose && (
                            <button onClick={onClose} className="p-1.5 hover:bg-red-500/20 hover:text-red-400 rounded text-[#a1a1aa] transition-colors" title="Close">
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Navigation Tabs */}
                <div className="flex border-b border-[#27272a] bg-[#09090b]">
                    <button 
                        onClick={() => setActiveTab('details')}
                        className={`flex-1 py-3 text-[10px] uppercase tracking-widest font-bold transition-colors relative ${activeTab === 'details' ? 'text-white' : 'text-[#52525b] hover:text-[#a1a1aa]'}`}
                    >
                        Overview
                        {activeTab === 'details' && <div className="absolute bottom-0 left-0 w-full h-[2px] bg-white" />}
                    </button>
                    <button 
                        onClick={() => setActiveTab('timeline')}
                        className={`flex-1 py-3 text-[10px] uppercase tracking-widest font-bold transition-colors relative ${activeTab === 'timeline' ? 'text-white' : 'text-[#52525b] hover:text-[#a1a1aa]'}`}
                    >
                        Journey Log
                        {activeTab === 'timeline' && <div className="absolute bottom-0 left-0 w-full h-[2px] bg-white" />}
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto bg-black/50">
                
                {/* DETAILS TAB */}
                {activeTab === 'details' && (
                    <div className="p-4 space-y-4 animate-in slide-in-from-left-4 duration-300">
                        <div className="grid grid-cols-2 gap-2">
                             <div className="bg-[#18181b] border border-[#27272a] p-3 rounded-sm">
                                <div className="text-[#a1a1aa] text-[9px] uppercase tracking-wider mb-1 flex items-center gap-1.5"><Clock className="w-3 h-3" /> First Sighting</div>
                                <div className="text-white text-xs font-mono">{formatTime(selectedResult.track.first_seen)}</div>
                            </div>
                            <div className="bg-[#18181b] border border-[#27272a] p-3 rounded-sm">
                                <div className="text-[#a1a1aa] text-[9px] uppercase tracking-wider mb-1 flex items-center gap-1.5"><Activity className="w-3 h-3" /> Last Active</div>
                                <div className="text-white text-xs font-mono">{formatTime(selectedResult.track.last_seen)}</div>
                            </div>
                        </div>
                        
                        {/* Camera Sequence Badge List */}
                        <div className="bg-[#18181b] border border-[#27272a] p-3 rounded-sm">
                             <div className="text-[#a1a1aa] text-[9px] uppercase tracking-wider mb-3 flex items-center gap-1.5"><Camera className="w-3 h-3" /> Camera Sequence</div>
                             <div className="flex flex-wrap gap-2">
                                {selectedResult.track.camera_sequence.map((camId, idx) => (
                                    <div key={idx} className="flex items-center">
                                        <div className="h-6 px-2 flex items-center justify-center bg-[#27272a] border border-[#3f3f46] text-white text-[10px] font-mono rounded">
                                            CAM {camId}
                                        </div>
                                        {idx < selectedResult.track.camera_sequence.length - 1 && (
                                            <div className="w-4 h-[1px] bg-[#3f3f46] mx-1" />
                                        )}
                                    </div>
                                ))}
                             </div>
                        </div>

                        {/* Person Thumbnail or Placeholder */}
                        {selectedResult.thumbnail ? (
                            <div className="relative aspect-[3/4] bg-[#18181b] border border-[#27272a] rounded-sm overflow-hidden group">
                                <img 
                                    src={`data:image/jpeg;base64,${selectedResult.thumbnail}`} 
                                    alt={`ID ${selectedResult.track.id}`}
                                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                                    <p className="text-[10px] text-white font-mono uppercase tracking-widest">Global Identity Capture</p>
                                </div>
                            </div>
                        ) : (
                            <div className="p-4 border border-dashed border-[#27272a] rounded text-center">
                                <User className="w-8 h-8 text-[#27272a] mx-auto mb-2" />
                                <p className="text-[10px] text-[#52525b] uppercase tracking-widest">Re-ID Feature Attributes Unavailable</p>
                            </div>
                        )}
                    </div>
                )}

                {/* TIMELINE TAB */}
                {activeTab === 'timeline' && (
                    <div className="p-4 animate-in slide-in-from-right-4 duration-300">
                         <div className="relative pl-4 border-l border-[#27272a] ml-2 space-y-6">
                            {selectedResult.path_points?.map((point, idx) => (
                                <div key={idx} className="relative group">
                                     {/* Timeline Node */}
                                    <div className="absolute -left-[21px] top-1.5 w-2.5 h-2.5 bg-[#09090b] border-2 border-[#52525b] rounded-full group-hover:border-white group-hover:scale-125 transition-all z-10 box-content" />
                                    
                                    {/* Content Card */}
                                    <div className="bg-[#18181b] border border-[#27272a] p-3 rounded-sm group-hover:border-white/30 transition-colors">
                                        <div className="flex justify-between items-start mb-1">
                                            <span className="text-xs font-bold text-white flex items-center gap-2">
                                                {point.name || `Camera ${point.camera_id}`}
                                            </span>
                                            <span className="text-[9px] font-mono text-[#a1a1aa] bg-[#27272a] px-1.5 py-0.5 rounded">
                                                {/* Mock time offset for demo since path_points might not have indiv timestamps yet */}
                                                SEQ: {idx + 1}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2 text-[10px] text-[#71717a] font-mono">
                                            <MapPin className="w-3 h-3" />
                                            {point.latitude.toFixed(6)}, {point.longitude.toFixed(6)}
                                        </div>
                                    </div>
                                </div>
                            ))}
                         </div>
                    </div>
                )}
            </div>
        </div>
    );
}
