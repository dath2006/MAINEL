"""
Camera Repository

Database operations for Camera entities.
"""

from typing import List, Optional, Tuple
from datetime import datetime

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.functions import ST_MakePoint, ST_Distance, ST_DWithin
from loguru import logger

from app.db.models import Camera


class CameraRepository:
    """Repository for Camera database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        name: str,
        latitude: float,
        longitude: float,
        zone_id: Optional[int] = None,
        fov_angle: Optional[float] = None,
        stream_url: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Camera:
        """Create a new camera."""
        # Create PostGIS point from lat/lon
        location = func.ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
        
        camera = Camera(
            name=name,
            location=location,
            zone_id=zone_id,
            fov_angle=fov_angle,
            stream_url=stream_url,
            description=description,
        )
        
        self.session.add(camera)
        await self.session.flush()
        await self.session.refresh(camera)
        
        logger.info(f"Created camera {camera.id}: {name}")
        return camera
    
    async def get_by_id(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID."""
        result = await self.session.execute(
            select(Camera).where(Camera.id == camera_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        zone_id: Optional[int] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Camera]:
        """Get all cameras with optional filtering."""
        query = select(Camera)
        
        if zone_id is not None:
            query = query.where(Camera.zone_id == zone_id)
        
        if is_active is not None:
            query = query.where(Camera.is_active == is_active)
        
        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self,
        camera_id: int,
        **kwargs
    ) -> Optional[Camera]:
        """Update camera fields."""
        # Handle location update if lat/lon provided
        if 'latitude' in kwargs and 'longitude' in kwargs:
            kwargs['location'] = func.ST_SetSRID(
                ST_MakePoint(kwargs.pop('longitude'), kwargs.pop('latitude')),
                4326
            )
        elif 'latitude' in kwargs or 'longitude' in kwargs:
            # Need both lat and lon
            kwargs.pop('latitude', None)
            kwargs.pop('longitude', None)
        
        if kwargs:
            kwargs['updated_at'] = datetime.utcnow()
            await self.session.execute(
                update(Camera)
                .where(Camera.id == camera_id)
                .values(**kwargs)
            )
        
        return await self.get_by_id(camera_id)
    
    async def delete(self, camera_id: int) -> bool:
        """Delete a camera."""
        result = await self.session.execute(
            delete(Camera).where(Camera.id == camera_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted camera {camera_id}")
        return deleted
    
    async def set_active(self, camera_id: int, is_active: bool) -> Optional[Camera]:
        """Set camera active status."""
        return await self.update(camera_id, is_active=is_active)
    
    async def get_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_meters: float = 500,
        limit: int = 10,
    ) -> List[Camera]:
        """Get cameras within radius of a point."""
        point = func.ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
        
        query = (
            select(Camera)
            .where(ST_DWithin(Camera.location, point, radius_meters))
            .order_by(ST_Distance(Camera.location, point))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_distance(
        self,
        camera_id_1: int,
        camera_id_2: int,
    ) -> Optional[float]:
        """Get distance between two cameras in meters."""
        query = select(
            ST_Distance(
                select(Camera.location).where(Camera.id == camera_id_1).scalar_subquery(),
                select(Camera.location).where(Camera.id == camera_id_2).scalar_subquery(),
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_coordinates(
        self,
        camera_id: int,
    ) -> Optional[Tuple[float, float]]:
        """Get camera lat/lon coordinates."""
        query = select(
            func.ST_Y(Camera.location).label('latitude'),
            func.ST_X(Camera.location).label('longitude'),
        ).where(Camera.id == camera_id)
        
        result = await self.session.execute(query)
        row = result.first()
        if row:
            return (row.latitude, row.longitude)
        return None
    
    async def count(self, is_active: Optional[bool] = None) -> int:
        """Count cameras."""
        query = select(func.count(Camera.id))
        if is_active is not None:
            query = query.where(Camera.is_active == is_active)
        result = await self.session.execute(query)
        return result.scalar_one()
