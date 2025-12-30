"""
Global Track Repository

Database operations for GlobalTrack entities (cross-camera identities).
"""

from typing import List, Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from loguru import logger

from app.db.models import GlobalTrack, TrackStatus, Tracklet


class GlobalTrackRepository:
    """Repository for GlobalTrack database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        first_seen: datetime,
        camera_id: int,
        avg_embedding: Optional[List[float]] = None,
    ) -> GlobalTrack:
        """Create a new global track."""
        global_track = GlobalTrack(
            first_seen=first_seen,
            last_seen=first_seen,
            camera_sequence=[camera_id],
            avg_embedding=avg_embedding,
            status=TrackStatus.ACTIVE,
        )
        
        self.session.add(global_track)
        await self.session.flush()
        await self.session.refresh(global_track)
        
        logger.info(f"Created global track {global_track.id}")
        return global_track
    
    async def get_by_id(
        self,
        track_id: UUID,
        include_tracklets: bool = False,
    ) -> Optional[GlobalTrack]:
        """Get global track by ID."""
        query = select(GlobalTrack).where(GlobalTrack.id == track_id)
        
        if include_tracklets:
            query = query.options(selectinload(GlobalTrack.tracklets))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_active(
        self,
        camera_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[GlobalTrack]:
        """Get all active global tracks."""
        query = select(GlobalTrack).where(GlobalTrack.status == TrackStatus.ACTIVE)
        
        if camera_id is not None:
            query = query.where(GlobalTrack.camera_sequence.any(camera_id))
        
        query = query.order_by(GlobalTrack.last_seen.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_status(
        self,
        status: TrackStatus,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GlobalTrack]:
        """Get tracks by status."""
        query = (
            select(GlobalTrack)
            .where(GlobalTrack.status == status)
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def search(
        self,
        camera_ids: Optional[List[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: Optional[TrackStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GlobalTrack]:
        """Search global tracks with filters."""
        query = select(GlobalTrack)
        
        if camera_ids:
            # Filter by any camera in sequence
            for cam_id in camera_ids:
                query = query.where(GlobalTrack.camera_sequence.any(cam_id))
        
        if start_time:
            query = query.where(GlobalTrack.first_seen >= start_time)
        
        if end_time:
            query = query.where(GlobalTrack.last_seen <= end_time)
        
        if status:
            query = query.where(GlobalTrack.status == status)
        
        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self,
        track_id: UUID,
        **kwargs
    ) -> Optional[GlobalTrack]:
        """Update global track fields."""
        if kwargs:
            await self.session.execute(
                update(GlobalTrack)
                .where(GlobalTrack.id == track_id)
                .values(**kwargs)
            )
        return await self.get_by_id(track_id)
    
    async def add_camera_to_sequence(
        self,
        track_id: UUID,
        camera_id: int,
        last_seen: datetime,
    ) -> Optional[GlobalTrack]:
        """Add camera to track's camera sequence."""
        track = await self.get_by_id(track_id)
        if not track:
            return None
        
        sequence = list(track.camera_sequence or [])
        if not sequence or sequence[-1] != camera_id:
            sequence.append(camera_id)
        
        return await self.update(
            track_id,
            camera_sequence=sequence,
            last_seen=last_seen,
        )
    
    async def update_embedding(
        self,
        track_id: UUID,
        embedding: List[float],
    ) -> Optional[GlobalTrack]:
        """Update average embedding for track."""
        return await self.update(track_id, avg_embedding=embedding)
    
    async def set_status(
        self,
        track_id: UUID,
        status: TrackStatus,
    ) -> Optional[GlobalTrack]:
        """Set track status."""
        return await self.update(track_id, status=status)
    
    async def mark_lost(self, track_id: UUID) -> Optional[GlobalTrack]:
        """Mark track as lost."""
        return await self.set_status(track_id, TrackStatus.LOST)
    
    async def mark_finished(self, track_id: UUID) -> Optional[GlobalTrack]:
        """Mark track as finished."""
        return await self.set_status(track_id, TrackStatus.FINISHED)
    
    async def delete(self, track_id: UUID) -> bool:
        """Delete a global track."""
        result = await self.session.execute(
            delete(GlobalTrack).where(GlobalTrack.id == track_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted global track {track_id}")
        return deleted
    
    async def get_candidates_for_matching(
        self,
        camera_ids: List[int],
        since: datetime,
        exclude_id: Optional[UUID] = None,
        limit: int = 50,
    ) -> List[GlobalTrack]:
        """Get candidate tracks for ReID matching."""
        query = (
            select(GlobalTrack)
            .where(GlobalTrack.status.in_([TrackStatus.ACTIVE, TrackStatus.LOST]))
            .where(GlobalTrack.last_seen >= since)
            .where(GlobalTrack.avg_embedding.isnot(None))
        )
        
        # Filter by camera sequence overlap
        for cam_id in camera_ids:
            query = query.where(GlobalTrack.camera_sequence.any(cam_id))
        
        if exclude_id:
            query = query.where(GlobalTrack.id != exclude_id)
        
        query = query.limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count(self, status: Optional[TrackStatus] = None) -> int:
        """Count global tracks."""
        query = select(func.count(GlobalTrack.id))
        if status:
            query = query.where(GlobalTrack.status == status)
        result = await self.session.execute(query)
        return result.scalar_one()
