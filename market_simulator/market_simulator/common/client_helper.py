import redis

class Client:
    def _create_redis_client(self) -> redis.Redis:
        _redis_: redis.Redis = redis.Redis(host="192.168.1.92", port=6379)
        return _redis_

    def __init__(self) -> None:
        self.redis: redis.Redis = self._create_redis_client()

client: Client = Client()