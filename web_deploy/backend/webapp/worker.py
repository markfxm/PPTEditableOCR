from __future__ import annotations

import os

import redis
from rq import Queue, Worker


def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    connection = redis.from_url(redis_url)
    worker = Worker([Queue("ppttoedit", connection=connection)], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
