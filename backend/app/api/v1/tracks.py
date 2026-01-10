"""
Track Query API Endpoints

Query and search for person tracks across cameras.
"""

from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from loguru import logger

from app.services.reid_service import get_reid_service
from app.api.v1.streams import SourceResponse
from app.services.stream_manager import get_stream_manager

from app.schemas.track import (
    TrackStatus,
    GlobalTrackResponse,
    GlobalTrackDetail,
    TrackletResponse,
    TrackSearchQuery,
    TransitEvent,
)


router = APIRouter()

# In-memory storage (replace with database in production)
from app.services.track_store import get_track_store

router = APIRouter()



@router.get("/active", response_model=List[GlobalTrackResponse])
async def get_active_tracks(
    camera_id: Optional[int] = Query(None, description="Filter by last camera"),
    limit: int = Query(100, le=1000, description="Max results"),
):
    """
    Get all currently active global tracks.
    
    Active tracks are identities currently being tracked in the system.
    """
    tracks = [
        t for t in get_track_store().get_all_tracks()
        if t["status"] == TrackStatus.ACTIVE
    ]
    
    if camera_id is not None:
        tracks = [t for t in tracks if camera_id in t.get("camera_sequence", [])]
    
    tracks = tracks[:limit]
    return [GlobalTrackResponse(**t) for t in tracks]


@router.get("/{track_id}", response_model=GlobalTrackDetail)
async def get_track(track_id: UUID):
    """Get full trajectory history for a track."""
    track_id_str = str(track_id)
    
    store = get_track_store()
    track_data = store.get_track(track_id_str)
    
    if not track_data:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    track_data = track_data.copy()
    
    # Get associated tracklets
    tracklet_ids = track_data.get("tracklet_ids", [])
    tracklets = [
        TrackletResponse(**_tracklets[tid])
        for tid in tracklet_ids
        if tid in _tracklets
    ]
    
    track_data["tracklets"] = tracklets
    
    return GlobalTrackDetail(**track_data)


@router.post("/search", response_model=List[GlobalTrackResponse])
async def search_tracks(query: TrackSearchQuery):
    """
    Search tracks by various criteria.
    
    Supports filtering by camera, time range, status, and duration.
    """
    tracks = get_track_store().get_all_tracks()
    
    # Apply filters
    if query.camera_ids:
        tracks = [
            t for t in tracks
            if any(cid in t.get("camera_sequence", []) for cid in query.camera_ids)
        ]
    
    if query.status:
        tracks = [t for t in tracks if t["status"] == query.status]
    
    if query.start_time:
        tracks = [t for t in tracks if t["first_seen"] >= query.start_time]
    
    if query.end_time:
        tracks = [t for t in tracks if t["last_seen"] <= query.end_time]
    
    # Pagination
    tracks = tracks[query.offset:query.offset + query.limit]
    
    return [GlobalTrackResponse(**t) for t in tracks]


@router.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = 5,
    threshold: Optional[float] = Query(None, description="Similarity threshold (defaults to server config)"),
    mode: str = Query("auto", enum=["auto", "face", "body"], description="Search mode: 'auto' (fusion), 'face' (face-only), 'body' (body-only)")
):
    """
    Search for a person by image.
    
    Uploads an image, extracts features, and finds matching global tracks.
    Returns matches with their tracking history and camera locations.
    """
    reid_service = get_reid_service()
    stream_manager = get_stream_manager()
    
    try:
        content = await file.read()
        matches = reid_service.search_by_image(content, top_k=limit, threshold=threshold, mode=mode)
        
        # Enrich results with GlobalTrack info and Camera locations
        results = []
        for match in matches:
             global_id = match["global_track_id"]
             score = match["score"]
             
             store = get_track_store()
             track_data = store.get_track(global_id)

             if track_data:
                 
                 # Enrich camera sequence with locations
                 camera_seq = track_data.get("camera_sequence", [])
                 logger.info(f"Enriching track {global_id} with camera seq: {camera_seq}")
                 with open("debug_tracks.log", "a") as f:
                     f.write(f"Enriching track {global_id} with camera seq: {camera_seq}\n")
                 
                 path_points = []
                 for cam_id in camera_seq:
                     # Find source for this camera to get location
                     found_source = False
                     for source in stream_manager.sources:
                         if source.camera_id == cam_id:
                             path_points.append({
                                 "camera_id": cam_id,
                                 "latitude": source.latitude,
                                 "longitude": source.longitude,
                                 "name": source.name
                             })
                             found_source = True
                             break
                     if not found_source:
                         logger.warning(f"No source found for camera_id {cam_id} in track {global_id}")

                 if not path_points:
                     logger.warning(f"No path points generated from sequence {camera_seq}")
                     with open("debug_tracks.log", "a") as f:
                         f.write(f"No path points generated from sequence {camera_seq}\n")
                 else:
                     logger.info(f"Generated {len(path_points)} path points")
                     with open("debug_tracks.log", "a") as f:
                         f.write(f"Generated {len(path_points)} path points: {path_points}\n")
                 
                 results.append({
                     "track": GlobalTrackResponse(**track_data),
                     "score": float(score),
                     "path_points": path_points
                 })
                 
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")


@router.get("/{track_id}/interpolate")
async def get_interpolated_path(
    track_id: UUID,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
):
    """
    Get OSRM-interpolated path for a track's trajectory.
    
    Returns the estimated path between camera observations.
    """
    track_id_str = str(track_id)
    
    if track_id_str not in _global_tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    # TODO: Implement OSRM interpolation
    return {
        "track_id": track_id,
        "from_time": from_time,
        "to_time": to_time,
        "path": [],  # GeoJSON LineString coordinates
        "message": "OSRM interpolation not yet implemented",
    }


@router.get("/{track_id}/transits", response_model=List[TransitEvent])
async def get_track_transits(track_id: UUID):
    """Get all cross-camera transition events for a track."""
    track_id_str = str(track_id)
    
    if track_id_str not in _global_tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    # TODO: Retrieve transit events from database
    return []


@router.post("/{track_id}/finish", response_model=GlobalTrackResponse)
async def finish_track(track_id: UUID):
    """Mark a track as finished (person left monitored area)."""
    track_id_str = str(track_id)
    
    if track_id_str not in _global_tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    _global_tracks[track_id_str]["status"] = TrackStatus.FINISHED
    logger.info(f"Finished track {track_id}")
    
    return GlobalTrackResponse(**_global_tracks[track_id_str])


@router.delete("/{track_id}", status_code=204)
async def delete_track(track_id: UUID):
    """Delete a track and its associated data."""
    track_id_str = str(track_id)
    
    if track_id_str not in _global_tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    # Delete associated tracklets
    tracklet_ids = _global_tracks[track_id_str].get("tracklet_ids", [])
    for tid in tracklet_ids:
        if tid in _tracklets:
            del _tracklets[tid]
    
    del _global_tracks[track_id_str]
    logger.info(f"Deleted track {track_id}")
