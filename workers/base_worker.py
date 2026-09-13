"""
Resilient Message Broker and Worker Base Subsystem.
Provides a unified abstraction supporting Redis queues (BRPOP/LPUSH/PUBLISH)
with an automatic thread-safe in-memory FIFO queue fallback.
Ensures zero-crash out-of-the-box operation even if Redis daemon is not started.
"""

import time
import json
import threading
import queue
from typing import Dict, Any, Optional, Callable
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

try:
    import redis
    REDIS_LIB_AVAILABLE = True
except ImportError:
    REDIS_LIB_AVAILABLE = False


class MessageBroker:
    """
    Unified broker providing queue (push/pop) and Pub/Sub (publish/subscribe) capabilities.
    Automatically detects Redis on port 6379, falling back gracefully to thread-safe memory queues.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MessageBroker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB, password: Optional[str] = REDIS_PASSWORD):
        if self._initialized:
            return

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis_client = None
        self.is_redis_online = False

        # In-memory fallbacks
        self._memory_queues: Dict[str, queue.Queue] = {}
        self._channel_subscribers: Dict[str, list] = {}
        self._mem_lock = threading.Lock()

        self._connect()
        self._initialized = True

    def _connect(self):
        """Attempts connection to Redis broker."""
        if REDIS_LIB_AVAILABLE:
            try:
                r = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0
                )
                r.ping()
                self.redis_client = r
                self.is_redis_online = True
                print(f"[MessageBroker] CONNECTED to Redis at {self.host}:{self.port} (DB {self.db})")
                return
            except Exception as e:
                self.is_redis_online = False
                self.redis_client = None
                print(f"[MessageBroker] Redis unavailable ({e}). Activating High-Performance In-Memory Broker.")
        else:
            print("[MessageBroker] 'redis' package not installed. Activating In-Memory Broker.")
            self.is_redis_online = False

    def _get_mem_queue(self, queue_name: str) -> queue.Queue:
        with self._mem_lock:
            if queue_name not in self._memory_queues:
                self._memory_queues[queue_name] = queue.Queue(maxsize=1000)
            return self._memory_queues[queue_name]

    def push(self, queue_name: str, payload: Any) -> bool:
        """Pushes an item (dict or string) to a designated queue."""
        serialized = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

        if self.is_redis_online and self.redis_client is not None:
            try:
                self.redis_client.lpush(queue_name, serialized)
                return True
            except Exception as e:
                print(f"[MessageBroker] Redis push error: {e}. Falling back to memory queue.")
                self.is_redis_online = False

        # Fallback to in-memory queue
        q = self._get_mem_queue(queue_name)
        try:
            if q.full():
                # Drop oldest to prevent memory explosion
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(serialized)
            return True
        except Exception as e:
            print(f"[MessageBroker] Memory queue push error: {e}")
            return False

    def pop(self, queue_name: str, timeout: float = 1.0) -> Optional[Any]:
        """
        Pops an item from the queue with a timeout.
        Returns parsed dict/string or None if queue is empty.
        """
        if self.is_redis_online and self.redis_client is not None:
            try:
                # BRPOP returns a tuple (queue_name, item)
                result = self.redis_client.brpop(queue_name, timeout=int(max(1, timeout)))
                if result:
                    _, raw = result
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return raw
                return None
            except Exception as e:
                self.is_redis_online = False

        # Fallback to memory queue
        q = self._get_mem_queue(queue_name)
        try:
            raw = q.get(timeout=timeout)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        except queue.Empty:
            return None

    def publish(self, channel: str, payload: Any):
        """Publishes a payload to a channel."""
        serialized = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

        if self.is_redis_online and self.redis_client is not None:
            try:
                self.redis_client.publish(channel, serialized)
            except Exception:
                pass

        # Trigger in-memory subscribers
        with self._mem_lock:
            subs = list(self._channel_subscribers.get(channel, []))
        parsed = payload if isinstance(payload, (dict, list)) else serialized
        for callback in subs:
            try:
                callback(parsed)
            except Exception as e:
                print(f"[MessageBroker] Subscriber error on {channel}: {e}")

    def subscribe(self, channel: str, callback: Callable[[Any], None]):
        """Subscribes a callback to an in-memory channel."""
        with self._mem_lock:
            if channel not in self._channel_subscribers:
                self._channel_subscribers[channel] = []
            self._channel_subscribers[channel].append(callback)

    def queue_size(self, queue_name: str) -> int:
        """Returns approximate number of pending items in queue."""
        if self.is_redis_online and self.redis_client is not None:
            try:
                return int(self.redis_client.llen(queue_name))
            except Exception:
                pass
        return self._get_mem_queue(queue_name).qsize()


class BaseWorker:
    """
    Abstract base class for asynchronous specialized workers (ANPR, FRS, Anomaly).
    Executes in a dedicated daemon thread, continuously consuming from assigned queue.
    """
    def __init__(self, name: str, queue_name: str, broker: Optional[MessageBroker] = None):
        self.name = name
        self.queue_name = queue_name
        self.broker = broker if broker is not None else MessageBroker()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.items_processed = 0
        self.last_active_time = time.time()

    def start(self):
        """Starts worker in background daemon thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, name=f"Worker-{self.name}", daemon=True)
        self.thread.start()
        print(f"[{self.name}] Worker started and listening on queue '{self.queue_name}'")

    def stop(self):
        """Signals worker to gracefully stop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print(f"[{self.name}] Worker stopped.")

    def _run_loop(self):
        while self.running:
            try:
                item = self.broker.pop(self.queue_name, timeout=1.0)
                if item is not None:
                    self.last_active_time = time.time()
                    self.items_processed += 1
                    self.process_item(item)
                else:
                    # Idle loop sleep
                    time.sleep(0.01)
            except Exception as e:
                print(f"[{self.name}] Processing exception: {e}")
                time.sleep(0.5)

    def process_item(self, item: Any):
        """Abstract processing method to be overridden by specialist worker."""
        raise NotImplementedError("Specialist worker must implement process_item()")
