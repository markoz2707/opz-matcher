"""
Cache service for OPZ Product Matcher
Provides Redis and in-memory caching with TTL support
"""
import json
import asyncio
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from loguru import logger

from config.settings import settings


class CacheService:
    """Unified caching service with Redis and in-memory fallback"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized:
            return

        try:
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {e}")
            self.redis_client = None

        self._initialized = True

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client, initialize if needed"""
        if not self._initialized:
            await self.initialize()
        return self.redis_client

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set cache value with optional TTL

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds

        Returns:
            bool: True if successful
        """
        try:
            serialized_value = json.dumps(value, default=str)

            redis_client = await self._get_redis()
            if redis_client:
                if ttl_seconds:
                    await redis_client.setex(key, ttl_seconds, serialized_value)
                else:
                    await redis_client.set(key, serialized_value)
                return True
            else:
                # In-memory cache
                self.memory_cache[key] = {
                    'value': serialized_value,
                    'expires_at': datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
                }
                return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get cache value

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        try:
            redis_client = await self._get_redis()
            if redis_client:
                value = await redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # In-memory cache
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry['expires_at'] is None or cache_entry['expires_at'] > datetime.now():
                        return json.loads(cache_entry['value'])
                    else:
                        # Expired, remove
                        del self.memory_cache[key]

            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """
        Delete cache key

        Args:
            key: Cache key

        Returns:
            bool: True if successful
        """
        try:
            redis_client = await self._get_redis()
            if redis_client:
                await redis_client.delete(key)
                return True
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            bool: True if exists and not expired
        """
        try:
            redis_client = await self._get_redis()
            if redis_client:
                return bool(await redis_client.exists(key))
            else:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry['expires_at'] is None or cache_entry['expires_at'] > datetime.now():
                        return True
                    else:
                        del self.memory_cache[key]
                return False
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern

        Args:
            pattern: Redis pattern (e.g., "product:*")

        Returns:
            int: Number of keys deleted
        """
        try:
            redis_client = await self._get_redis()
            if redis_client:
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
                    return len(keys)
                return 0
            else:
                # In-memory - find matching keys
                deleted_count = 0
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace('*', '') in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    deleted_count += 1
                return deleted_count
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get TTL for key in seconds

        Args:
            key: Cache key

        Returns:
            TTL in seconds or None if key doesn't exist or no TTL
        """
        try:
            redis_client = await self._get_redis()
            if redis_client:
                ttl = await redis_client.ttl(key)
                return ttl if ttl > 0 else None
            else:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry['expires_at']:
                        remaining = int((cache_entry['expires_at'] - datetime.now()).total_seconds())
                        return max(0, remaining)
                return None
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return None

    # Specialized cache methods for common use cases

    async def cache_product_search(self, query_hash: str, results: List[Dict], ttl_seconds: int = 300) -> bool:
        """Cache product search results"""
        return await self.set(f"search:{query_hash}", results, ttl_seconds)

    async def get_cached_product_search(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached product search results"""
        return await self.get(f"search:{query_hash}")

    async def cache_product_details(self, product_id: int, details: Dict, ttl_seconds: int = 600) -> bool:
        """Cache product details"""
        return await self.set(f"product:{product_id}", details, ttl_seconds)

    async def get_cached_product_details(self, product_id: int) -> Optional[Dict]:
        """Get cached product details"""
        return await self.get(f"product:{product_id}")

    async def cache_benchmark_data(self, benchmark_id: str, data: Dict, ttl_seconds: int = 3600) -> bool:
        """Cache benchmark data"""
        return await self.set(f"benchmark:{benchmark_id}", data, ttl_seconds)

    async def get_cached_benchmark_data(self, benchmark_id: str) -> Optional[Dict]:
        """Get cached benchmark data"""
        return await self.get(f"benchmark:{benchmark_id}")

    async def invalidate_product_cache(self, product_id: int) -> bool:
        """Invalidate all caches related to a product"""
        success = True
        success &= await self.delete(f"product:{product_id}")
        # Also clear any search caches that might contain this product
        await self.clear_pattern(f"search:*")
        return success

    async def invalidate_benchmark_cache(self, benchmark_id: str) -> bool:
        """Invalidate benchmark cache"""
        return await self.delete(f"benchmark:{benchmark_id}")


# Global cache service instance
cache_service = CacheService()