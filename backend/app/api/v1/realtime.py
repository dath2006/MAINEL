"""
Real-time WebSocket API

Provides real-time streaming of tracking events.
"""

from typing import Dict, Set
from datetime import datetime
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger


router = APIRouter()

# Active WebSocket connections
_active_connections: Set[WebSocket] = set()
_connection_lock = asyncio.Lock()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")


manager = ConnectionManager()


@router.websocket("/tracks")
async def track_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time track updates.
    
    Streams events:
    - detection: New person detected
    - track_update: Track position updated
    - transit: Cross-camera transition
    - reid_match: ReID match found
    - track_lost: Track lost
    - track_finished: Track finished
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to MCMT tracking stream",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client messages (ping/pong, subscriptions, etc.)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # Send heartbeat if no message
                )
                
                # Handle client messages
                try:
                    message = json.loads(data)
                    await handle_client_message(websocket, message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON",
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: dict):
    """Handle incoming client messages."""
    msg_type = message.get("type")
    
    if msg_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    elif msg_type == "subscribe":
        # Client wants to subscribe to specific cameras
        camera_ids = message.get("camera_ids", [])
        await websocket.send_json({
            "type": "subscribed",
            "camera_ids": camera_ids,
            "message": f"Subscribed to {len(camera_ids)} cameras",
        })
    
    elif msg_type == "unsubscribe":
        await websocket.send_json({
            "type": "unsubscribed",
            "message": "Unsubscribed from all cameras",
        })
    
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
        })


# Functions to broadcast events (called by tracking services)

async def broadcast_detection(
    camera_id: int,
    track_id: int,
    bbox: dict,
    confidence: float,
    timestamp: datetime,
):
    """Broadcast new detection event."""
    await manager.broadcast({
        "type": "detection",
        "data": {
            "camera_id": camera_id,
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
        },
        "timestamp": timestamp.isoformat(),
    })


async def broadcast_transit(
    global_track_id: str,
    from_camera_id: int,
    to_camera_id: int,
    visual_similarity: float,
    st_probability: float,
    timestamp: datetime,
):
    """Broadcast cross-camera transit event."""
    await manager.broadcast({
        "type": "transit",
        "data": {
            "global_track_id": global_track_id,
            "from_camera_id": from_camera_id,
            "to_camera_id": to_camera_id,
            "visual_similarity": visual_similarity,
            "st_probability": st_probability,
        },
        "timestamp": timestamp.isoformat(),
    })


async def broadcast_track_lost(global_track_id: str, last_camera_id: int):
    """Broadcast track lost event."""
    await manager.broadcast({
        "type": "track_lost",
        "data": {
            "global_track_id": global_track_id,
            "last_camera_id": last_camera_id,
        },
        "timestamp": datetime.utcnow().isoformat(),
    })


async def broadcast_event(event: dict):
    """Broadcast any event to all clients."""
    await manager.broadcast(event)


async def broadcast_track_path_update(
    global_track_id: str,
    camera_sequence: list,
    path_points: list,
    from_camera_id: int,
    to_camera_id: int,
):
    """
    Broadcast track path update for real-time map visualization.
    
    Called when a person moves from one camera to another.
    """
    await manager.broadcast({
        "type": "track_path_update",
        "data": {
            "global_track_id": global_track_id,
            "from_camera_id": from_camera_id,
            "to_camera_id": to_camera_id,
            "camera_sequence": camera_sequence,
            "path_points": path_points,
        },
        "timestamp": datetime.utcnow().isoformat(),
    })

