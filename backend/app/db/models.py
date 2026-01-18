"""
Database Models

SQLAlchemy models for cameras, tracklets, and global tracks.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4
import enum

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, Enum, Index, ARRAY,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.db.session import Base


class TrackStatus(str, enum.Enum):
    """Track lifecycle status."""
    ACTIVE = "active"
    LOST = "lost"
    FINISHED = "finished"


class Camera(Base):
    """
    Camera node in the tracking network.
    
    Stores camera location and configuration.
    """
    __tablename__ = "cameras"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Geospatial location (Simple Lat/Lon)
    latitude: Mapped[float] = mapped_column(Float, default=0.0)
    longitude: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Optional zone grouping
    zone_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    
    # Camera configuration
    fov_angle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stream_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=datetime.utcnow)
    
    # Relationships
    tracklets: Mapped[List["Tracklet"]] = relationship("Tracklet", back_populates="camera")
    
    # Indexes
    __table_args__ = (
        Index('ix_cameras_zone', 'zone_id'),
        Index('ix_cameras_active', 'is_active'),
    )
    
    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.name}')>"


class Tracklet(Base):
    """
    Single-camera track (local track).
    
    Represents a person detection sequence within one camera view.
    """
    __tablename__ = "tracklets"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    
    # Track timing
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Feature embedding (stored as array)
    feature_vector: Mapped[Optional[List[float]]] = mapped_column(
        ARRAY(Float), nullable=True
    )
    
    # Exit information for transition detection
    exit_zone: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # "left", "right", "top", "bottom"
    
    # Local track ID (from DeepSORT)
    local_track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Link to global identity
    global_track_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("global_tracks.id"), nullable=True
    )
    
    # Detection count in this tracklet
    detection_count: Mapped[int] = mapped_column(Integer, default=1)
    
    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="tracklets")
    global_track: Mapped[Optional["GlobalTrack"]] = relationship(
        "GlobalTrack", back_populates="tracklets"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_tracklets_camera_time', 'camera_id', 'start_time'),
        Index('ix_tracklets_global', 'global_track_id'),
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate tracklet duration."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def __repr__(self) -> str:
        return f"<Tracklet(id={self.id}, camera={self.camera_id})>"


class GlobalTrack(Base):
    """
    Global identity track across multiple cameras.
    
    Represents a unique person tracked across the camera network.
    """
    __tablename__ = "global_tracks"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        Enum(TrackStatus), default=TrackStatus.ACTIVE
    )
    
    # Time range
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Camera sequence (ordered list of camera IDs visited)
    camera_sequence: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), default=[]
    )
    
    # Average feature embedding for this identity
    avg_embedding: Mapped[Optional[List[float]]] = mapped_column(
        ARRAY(Float), nullable=True
    )
    
    # Metadata (renamed to avoid SQLAlchemy reserved name)
    track_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, default={})
    
    # Relationships
    tracklets: Mapped[List["Tracklet"]] = relationship(
        "Tracklet", back_populates="global_track"
    )
    transit_events_from: Mapped[List["TransitEvent"]] = relationship(
        "TransitEvent",
        foreign_keys="TransitEvent.global_track_id",
        back_populates="global_track"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_global_tracks_status', 'status'),
        Index('ix_global_tracks_time', 'first_seen', 'last_seen'),
    )
    
    @property
    def tracklet_count(self) -> int:
        return len(self.tracklets) if self.tracklets else 0
    
    def __repr__(self) -> str:
        return f"<GlobalTrack(id={self.id}, status={self.status})>"


class TransitEvent(Base):
    """
    Cross-camera transition event.
    
    Records when a person moves between camera views.
    """
    __tablename__ = "transit_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    global_track_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("global_tracks.id"), nullable=False
    )
    
    # Camera transition
    from_camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"))
    to_camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"))
    
    # Tracklet references
    from_tracklet_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("tracklets.id"))
    to_tracklet_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("tracklets.id"))
    
    # Timing
    transit_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Matching scores
    visual_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    st_probability: Mapped[float] = mapped_column(Float, nullable=True)
    joint_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Interpolated path (GeoJSON)
    interpolated_path: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    global_track: Mapped["GlobalTrack"] = relationship(
        "GlobalTrack", back_populates="transit_events_from"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_transit_cameras', 'from_camera_id', 'to_camera_id'),
        Index('ix_transit_time', 'timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<TransitEvent({self.from_camera_id} -> {self.to_camera_id})>"


class CameraTransition(Base):
    """
    Camera transition statistics (learned topology).
    
    Stores transition time distributions between camera pairs.
    """
    __tablename__ = "camera_transitions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    from_camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"))
    to_camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"))
    
    # Statistics
    observation_count: Mapped[int] = mapped_column(Integer, default=0)
    mean_transit_time: Mapped[float] = mapped_column(Float, default=0.0)
    std_transit_time: Mapped[float] = mapped_column(Float, default=30.0)
    min_transit_time: Mapped[float] = mapped_column(Float, default=0.0)
    max_transit_time: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Histogram data for Parzen estimation
    time_histogram: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)
    
    # Distance (meters) - can be computed from GPS or provided
    distance_meters: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Unique constraint on camera pair
    __table_args__ = (
        Index('ix_camera_transitions_pair', 'from_camera_id', 'to_camera_id', unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<CameraTransition({self.from_camera_id} -> {self.to_camera_id})>"


class VideoMetadata(Base):
    """
    Video library metadata.
    
    Stores information about uploaded videos that can be reused
    as sources without re-uploading.
    """
    __tablename__ = "video_metadata"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    
    # File information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)  # bytes
    
    # Video properties
    duration: Mapped[float] = mapped_column(Float, default=0.0)  # seconds
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    
    # Usage tracking
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    use_count: Mapped[int] = mapped_column(Integer, default=0)  # How many sources use this video
    
    # Optional description/tags
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), default=[])
    
    # Indexes
    __table_args__ = (
        Index('ix_video_metadata_uploaded', 'uploaded_at'),
        Index('ix_video_metadata_filename', 'filename'),
    )
    
    def __repr__(self) -> str:
        return f"<VideoMetadata(id={self.id}, filename='{self.filename}')>"
