"""
Camera Network Topology

Manages the camera network graph and transition relationships.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class CameraNode:
    """Camera node in the topology graph."""
    camera_id: int
    lat: float
    lon: float
    zone_id: Optional[int] = None
    neighbors: Set[int] = field(default_factory=set)
    entry_zones: List[str] = field(default_factory=list)  # e.g., ["left", "bottom"]
    exit_zones: List[str] = field(default_factory=list)   # e.g., ["right", "top"]


@dataclass
class TopologyEdge:
    """Edge between camera nodes."""
    from_camera: int
    to_camera: int
    distance: float  # meters
    avg_transit_time: float = 0.0  # seconds
    transition_count: int = 0
    is_bidirectional: bool = True


class CameraTopology:
    """
    Camera network topology manager.
    
    Maintains the graph structure of camera connections and
    provides methods for topology learning and querying.
    """
    
    def __init__(self, auto_connect_radius: float = 500.0):
        """
        Args:
            auto_connect_radius: Auto-connect cameras within this distance (meters)
        """
        self.auto_connect_radius = auto_connect_radius
        
        # Nodes: camera_id -> CameraNode
        self.nodes: Dict[int, CameraNode] = {}
        
        # Adjacency matrix (dynamic size)
        self._adjacency: Dict[Tuple[int, int], TopologyEdge] = {}
        
        logger.info(f"CameraTopology initialized (auto_connect={auto_connect_radius}m)")
    
    def add_camera(
        self,
        camera_id: int,
        lat: float,
        lon: float,
        zone_id: Optional[int] = None,
    ):
        """Add camera to topology."""
        node = CameraNode(
            camera_id=camera_id,
            lat=lat,
            lon=lon,
            zone_id=zone_id,
        )
        self.nodes[camera_id] = node
        
        # Auto-connect to nearby cameras
        for other_id, other_node in self.nodes.items():
            if other_id == camera_id:
                continue
            
            distance = self._haversine_distance(lat, lon, other_node.lat, other_node.lon)
            if distance <= self.auto_connect_radius:
                self.connect_cameras(camera_id, other_id, distance)
        
        logger.info(f"Added camera {camera_id} at ({lat:.4f}, {lon:.4f})")
    
    def remove_camera(self, camera_id: int):
        """Remove camera from topology."""
        if camera_id in self.nodes:
            # Remove all edges
            edges_to_remove = [
                key for key in self._adjacency
                if camera_id in key
            ]
            for key in edges_to_remove:
                del self._adjacency[key]
            
            # Remove from neighbors
            for node in self.nodes.values():
                node.neighbors.discard(camera_id)
            
            del self.nodes[camera_id]
            logger.info(f"Removed camera {camera_id}")
    
    def connect_cameras(
        self,
        cam1: int,
        cam2: int,
        distance: Optional[float] = None,
        bidirectional: bool = True,
    ):
        """
        Connect two cameras in the topology.
        
        Args:
            cam1: First camera ID
            cam2: Second camera ID
            distance: Distance in meters (computed if None)
            bidirectional: Create edge in both directions
        """
        if cam1 not in self.nodes or cam2 not in self.nodes:
            logger.warning(f"Cannot connect: camera(s) not in topology")
            return
        
        if distance is None:
            distance = self._haversine_distance(
                self.nodes[cam1].lat, self.nodes[cam1].lon,
                self.nodes[cam2].lat, self.nodes[cam2].lon,
            )
        
        # Add edge
        edge = TopologyEdge(
            from_camera=cam1,
            to_camera=cam2,
            distance=distance,
            is_bidirectional=bidirectional,
        )
        self._adjacency[(cam1, cam2)] = edge
        self.nodes[cam1].neighbors.add(cam2)
        
        if bidirectional:
            edge_rev = TopologyEdge(
                from_camera=cam2,
                to_camera=cam1,
                distance=distance,
                is_bidirectional=True,
            )
            self._adjacency[(cam2, cam1)] = edge_rev
            self.nodes[cam2].neighbors.add(cam1)
        
        logger.debug(f"Connected cameras {cam1} <-> {cam2} (distance={distance:.1f}m)")
    
    def disconnect_cameras(self, cam1: int, cam2: int):
        """Remove connection between cameras."""
        if (cam1, cam2) in self._adjacency:
            del self._adjacency[(cam1, cam2)]
            self.nodes[cam1].neighbors.discard(cam2)
        
        if (cam2, cam1) in self._adjacency:
            del self._adjacency[(cam2, cam1)]
            self.nodes[cam2].neighbors.discard(cam1)
    
    def is_connected(self, cam1: int, cam2: int) -> bool:
        """Check if direct connection exists."""
        return (cam1, cam2) in self._adjacency
    
    def get_neighbors(self, camera_id: int) -> List[int]:
        """Get neighboring cameras."""
        if camera_id not in self.nodes:
            return []
        return list(self.nodes[camera_id].neighbors)
    
    def get_reachable(
        self,
        camera_id: int,
        max_hops: int = 2,
    ) -> Set[int]:
        """
        Get all cameras reachable within max_hops.
        
        Uses BFS to find reachable cameras.
        """
        if camera_id not in self.nodes:
            return set()
        
        visited = {camera_id}
        frontier = {camera_id}
        
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.nodes[node].neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        
        visited.discard(camera_id)  # Remove source
        return visited
    
    def get_distance(self, cam1: int, cam2: int) -> Optional[float]:
        """Get distance between cameras."""
        if (cam1, cam2) in self._adjacency:
            return self._adjacency[(cam1, cam2)].distance
        
        if cam1 in self.nodes and cam2 in self.nodes:
            return self._haversine_distance(
                self.nodes[cam1].lat, self.nodes[cam1].lon,
                self.nodes[cam2].lat, self.nodes[cam2].lon,
            )
        
        return None
    
    def update_transition(
        self,
        from_camera: int,
        to_camera: int,
        transit_time: float,
    ):
        """
        Update edge statistics with observed transition.
        
        This allows the topology to learn typical transit times.
        """
        key = (from_camera, to_camera)
        
        if key not in self._adjacency:
            # Auto-create edge if transition observed
            self.connect_cameras(from_camera, to_camera)
        
        edge = self._adjacency[key]
        
        # Update running average
        n = edge.transition_count
        edge.avg_transit_time = (edge.avg_transit_time * n + transit_time) / (n + 1)
        edge.transition_count = n + 1
        
        logger.debug(f"Transition {from_camera} -> {to_camera}: {transit_time:.1f}s (avg={edge.avg_transit_time:.1f}s)")
    
    def infer_topology_from_transitions(
        self,
        transitions: List[Tuple[int, int, float]],
        min_count: int = 3,
    ):
        """
        Infer topology graph from observed transitions.
        
        Args:
            transitions: List of (from_cam, to_cam, transit_time)
            min_count: Minimum observations to create edge
        """
        # Count transitions
        counts: Dict[Tuple[int, int], List[float]] = {}
        for from_cam, to_cam, time in transitions:
            key = (from_cam, to_cam)
            if key not in counts:
                counts[key] = []
            counts[key].append(time)
        
        # Create edges for frequent transitions
        for (from_cam, to_cam), times in counts.items():
            if len(times) >= min_count:
                self.connect_cameras(from_cam, to_cam, bidirectional=True)
                
                edge = self._adjacency[(from_cam, to_cam)]
                edge.avg_transit_time = np.mean(times)
                edge.transition_count = len(times)
        
        logger.info(f"Inferred topology: {len(self._adjacency)} edges from {len(transitions)} transitions")
    
    def get_adjacency_matrix(self) -> Tuple[List[int], np.ndarray]:
        """
        Get adjacency matrix representation.
        
        Returns:
            Tuple of (camera_ids, adjacency_matrix)
            Matrix values are inverse distances (0 = not connected)
        """
        camera_ids = sorted(self.nodes.keys())
        n = len(camera_ids)
        id_to_idx = {cid: i for i, cid in enumerate(camera_ids)}
        
        matrix = np.zeros((n, n))
        for (cam1, cam2), edge in self._adjacency.items():
            i, j = id_to_idx[cam1], id_to_idx[cam2]
            matrix[i, j] = 1.0 / max(edge.distance, 1.0)  # Inverse distance
        
        return camera_ids, matrix
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance between two GPS coordinates in meters."""
        R = 6371000  # Earth radius
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi / 2) ** 2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    @property
    def camera_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self._adjacency)
