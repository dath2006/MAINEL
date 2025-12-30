"""
Tracklet Repository

Database operations for Tracklet entities (single-camera tracks).
"""

from typing import List, Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.db.models import Tracklet


class TrackletRepository:
    """Repository for Tracklet database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        camera_id: int,
        start_time: datetime,
        local_track_id: Optional[int] = None,
        feature_vector: Optional[List[float]] = None,
    ) -> Tracklet:
        """Create a new tracklet."""
        tracklet = Tracklet(
            camera_id=camera_id,
            start_time=start_time,
            local_track_id=local_track_id,
            feature_vector=feature_vector,
        )
        
        self.session.add(tracklet)
        await self.session.flush()
        await self.session.refresh(tracklet)
        
        logger.debug(f"Created tracklet {tracklet.id} for camera {camera_id}")
        return tracklet
    
    async def get_by_id(self, tracklet_id: UUID) -> Optional[Tracklet]:
        """Get tracklet by ID."""
        result = await self.session.execute(
            select(Tracklet).where(Tracklet.id == tracklet_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_camera(
        self,
        camera_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Tracklet]:
        """Get tracklets for a camera with optional time filter."""
        query = select(Tracklet).where(Tracklet.camera_id == camera_id)
        
        if start_time:
            query = query.where(Tracklet.start_time >= start_time)
        if end_time:
            query = query.where(Tracklet.start_time <= end_time)
        
        query = query.order_by(Tracklet.start_time.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_unassigned(
        self,
        camera_id: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Tracklet]:
        """Get tracklets not yet assigned to a global track."""
        query = select(Tracklet).where(Tracklet.global_track_id.is_(None))
        
        if camera_id:
            query = query.where(Tracklet.camera_id == camera_id)
        if since:
            query = query.where(Tracklet.start_time >= since)
        
        query = query.order_by(Tracklet.start_time).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self,
        tracklet_id: UUID,
        **kwargs
    ) -> Optional[Tracklet]:
        """Update tracklet fields."""
        if kwargs:
            await self.session.execute(
                update(Tracklet)
                .where(Tracklet.id == tracklet_id)
                .values(**kwargs)
            )
        return await self.get_by_id(tracklet_id)
    
    async def end_tracklet(
        self,
        tracklet_id: UUID,
        end_time: datetime,
        exit_zone: Optional[str] = None,
        feature_vector: Optional[List[float]] = None,
    ) -> Optional[Tracklet]:
        """Mark tracklet as ended."""
        update_data = {"end_time": end_time}
        if exit_zone:
            update_data["exit_zone"] = exit_zone
        if feature_vector:
            update_data["feature_vector"] = feature_vector
        
        return await self.update(tracklet_id, **update_data)
    
    async def assign_to_global(
        self,
        tracklet_id: UUID,
        global_track_id: UUID,
    ) -> Optional[Tracklet]:
        """Assign tracklet to a global track."""
        return await self.update(tracklet_id, global_track_id=global_track_id)
    
    async def delete(self, tracklet_id: UUID) -> bool:
        """Delete a tracklet."""
        result = await self.session.execute(
            delete(Tracklet).where(Tracklet.id == tracklet_id)
        )
        return result.rowcount > 0
    
    async def get_recent_exits(
        self,
        camera_ids: Optional[List[int]] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Tracklet]:
        """Get recently ended tracklets (for ReID matching)."""
        query = select(Tracklet).where(Tracklet.end_time.isnot(None))
        
        if camera_ids:
            query = query.where(Tracklet.camera_id.in_(camera_ids))
        if since:
            query = query.where(Tracklet.end_time >= since)
        
        query = query.order_by(Tracklet.end_time.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_by_camera(self, camera_id: int) -> int:
        """Count tracklets for a camera."""
        result = await self.session.execute(
            select(func.count(Tracklet.id)).where(Tracklet.camera_id == camera_id)
        )
        return result.scalar_one()
