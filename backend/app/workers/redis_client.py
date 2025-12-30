"""
Redis Message Queue Client

Provides async Redis stream operations for message queuing.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as redis
from loguru import logger

from app.config import settings


class RedisClient:
    """Async Redis client for message streaming."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.redis_url
        self._pool: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish connection pool."""
        if self._pool is None:
            self._pool = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info(f"Connected to Redis: {self.url}")
    
    async def disconnect(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Redis connection closed")
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client."""
        if self._pool is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._pool
    
    # Stream operations
    
    async def publish_event(
        self,
        stream: str,
        event_type: str,
        data: Dict[str, Any],
        maxlen: int = 10000,
    ) -> str:
        """
        Publish event to a stream.
        
        Args:
            stream: Stream name
            event_type: Type of event
            data: Event data (will be JSON serialized)
            maxlen: Max stream length (older entries trimmed)
            
        Returns:
            Message ID
        """
        message = {
            "type": event_type,
            "data": json.dumps(data),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        msg_id = await self.client.xadd(
            stream,
            message,
            maxlen=maxlen,
        )
        
        return msg_id
    
    async def consume_events(
        self,
        stream: str,
        consumer_group: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Consume events from a stream (consumer group).
        
        Args:
            stream: Stream name
            consumer_group: Consumer group name
            consumer_name: Consumer identifier
            count: Max messages to read
            block_ms: Block time in milliseconds
            
        Returns:
            List of (message_id, data) tuples
        """
        try:
            # Ensure consumer group exists
            try:
                await self.client.xgroup_create(
                    stream, consumer_group, id="0", mkstream=True
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Read messages
            result = await self.client.xreadgroup(
                groupname=consumer_group,
                consumername=consumer_name,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )
            
            if not result:
                return []
            
            messages = []
            for stream_name, stream_messages in result:
                for msg_id, fields in stream_messages:
                    event = {
                        "id": msg_id,
                        "type": fields.get("type"),
                        "data": json.loads(fields.get("data", "{}")),
                        "timestamp": fields.get("timestamp"),
                    }
                    messages.append(event)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error consuming from stream {stream}: {e}")
            return []
    
    async def ack_event(
        self,
        stream: str,
        consumer_group: str,
        message_id: str,
    ):
        """Acknowledge message processing."""
        await self.client.xack(stream, consumer_group, message_id)
    
    # Pub/Sub for real-time broadcasts
    
    async def publish(self, channel: str, message: Dict[str, Any]):
        """Publish message to channel."""
        await self.client.publish(channel, json.dumps(message))
    
    async def subscribe(self, channel: str):
        """Subscribe to channel."""
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub
    
    # Key-value cache operations
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
    ):
        """Set cache value with TTL."""
        await self.client.setex(
            key,
            ttl_seconds,
            json.dumps(value),
        )
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def cache_delete(self, key: str):
        """Delete cache key."""
        await self.client.delete(key)


# Stream names
STREAM_DETECTIONS = "mcmt:detections"
STREAM_TRACKLETS = "mcmt:tracklets"
STREAM_TRANSITS = "mcmt:transits"
STREAM_ALERTS = "mcmt:alerts"

# Pub/Sub channels
CHANNEL_REALTIME = "mcmt:realtime"

# Singleton
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None
