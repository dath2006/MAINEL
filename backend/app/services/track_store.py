from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from loguru import logger
from app.schemas.track import TrackStatus

class TrackStore:
    def __init__(self):
        self._global_tracks: Dict[str, dict] = {}
        self._tracklets: Dict[str, dict] = {}
        logger.info("TrackStore initialized")

    def add_or_update_track(self, global_id: str, track_data: dict):
        """Add or update a global track."""
        if global_id not in self._global_tracks:
            self._global_tracks[global_id] = {
                "id": global_id,
                "status": TrackStatus.ACTIVE,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "camera_sequence": [],
                "tracklet_count": 0,
                "tracklet_ids": []
            }
        
        track = self._global_tracks[global_id]
        
        # Update metadata
        # Merge new data if provided
        for key, value in track_data.items():
            if key in track:
                track[key] = value
                
        track["last_seen"] = datetime.utcnow()
        return track

    def get_track(self, global_id: str) -> Optional[dict]:
        return self._global_tracks.get(global_id)

    def get_all_tracks(self) -> List[dict]:
        return list(self._global_tracks.values())

    def update_camera_sequence(self, global_id: str, camera_id: int):
        if global_id in self._global_tracks:
            seq = self._global_tracks[global_id]["camera_sequence"]
            # Dedup sequential same-camera entries if needed
            if not seq or seq[-1] != camera_id:
                seq.append(camera_id)
                self._global_tracks[global_id]["last_seen"] = datetime.utcnow()

# Singleton
_track_store = TrackStore()

def get_track_store() -> TrackStore:
    return _track_store
