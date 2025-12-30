"""
OSRM Routing Client

Provides route interpolation for blind zones between cameras.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import httpx
from loguru import logger

from app.config import settings


@dataclass
class RouteResult:
    """Route calculation result."""
    distance_meters: float
    duration_seconds: float
    geometry: List[Tuple[float, float]]  # [(lon, lat), ...]
    legs: List[Dict]
    waypoints: List[Dict]


class OSRMClient:
    """
    OSRM (Open Source Routing Machine) client.
    
    Used to interpolate pedestrian paths between camera observations.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.osrm_url).rstrip("/")
        self.profile = "foot"  # Walking profile
        self._client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
            logger.info(f"OSRM client connected: {self.base_url}")
    
    async def disconnect(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OSRM client not connected")
        return self._client
    
    async def get_route(
        self,
        coordinates: List[Tuple[float, float]],
        geometries: str = "geojson",
        overview: str = "full",
        alternatives: bool = False,
    ) -> Optional[RouteResult]:
        """
        Calculate route between coordinates.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
            geometries: Output format (geojson, polyline, polyline6)
            overview: Geometry detail (full, simplified, false)
            alternatives: Request alternative routes
            
        Returns:
            RouteResult or None if routing fails
        """
        if len(coordinates) < 2:
            return None
        
        # Format coordinates
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
        
        url = f"{self.base_url}/route/v1/{self.profile}/{coords_str}"
        params = {
            "geometries": geometries,
            "overview": overview,
            "alternatives": str(alternatives).lower(),
            "steps": "false",
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "Ok":
                logger.warning(f"OSRM error: {data.get('message')}")
                return None
            
            route = data["routes"][0]
            
            # Extract geometry
            geometry = []
            if geometries == "geojson" and "geometry" in route:
                geometry = [
                    (coord[0], coord[1])
                    for coord in route["geometry"]["coordinates"]
                ]
            
            return RouteResult(
                distance_meters=route["distance"],
                duration_seconds=route["duration"],
                geometry=geometry,
                legs=route.get("legs", []),
                waypoints=data.get("waypoints", []),
            )
            
        except httpx.RequestError as e:
            logger.error(f"OSRM request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing OSRM response: {e}")
            return None
    
    async def get_duration(
        self,
        from_coord: Tuple[float, float],
        to_coord: Tuple[float, float],
    ) -> Optional[float]:
        """Get estimated walking duration between two points."""
        result = await self.get_route(
            [from_coord, to_coord],
            overview="false",
        )
        return result.duration_seconds if result else None
    
    async def get_distance(
        self,
        from_coord: Tuple[float, float],
        to_coord: Tuple[float, float],
    ) -> Optional[float]:
        """Get walking distance between two points."""
        result = await self.get_route(
            [from_coord, to_coord],
            overview="false",
        )
        return result.distance_meters if result else None
    
    async def interpolate_path(
        self,
        from_camera: Tuple[float, float],
        to_camera: Tuple[float, float],
    ) -> Optional[Dict]:
        """
        Get interpolated path for blind zone between cameras.
        
        Returns GeoJSON LineString.
        """
        result = await self.get_route(
            [from_camera, to_camera],
            geometries="geojson",
            overview="full",
        )
        
        if not result:
            return None
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    list(coord) for coord in result.geometry
                ],
            },
            "properties": {
                "distance_meters": result.distance_meters,
                "duration_seconds": result.duration_seconds,
            },
        }
    
    async def health_check(self) -> bool:
        """Check OSRM server health."""
        try:
            # Simple route request to verify server
            response = await self.client.get(
                f"{self.base_url}/route/v1/foot/0.0,0.0;0.001,0.001"
            )
            return response.status_code == 200
        except Exception:
            return False


# Singleton
_osrm_client: Optional[OSRMClient] = None


async def get_osrm_client() -> OSRMClient:
    """Get or create OSRM client singleton."""
    global _osrm_client
    if _osrm_client is None:
        _osrm_client = OSRMClient()
        await _osrm_client.connect()
    return _osrm_client


async def close_osrm():
    """Close OSRM connection."""
    global _osrm_client
    if _osrm_client:
        await _osrm_client.disconnect()
        _osrm_client = None
