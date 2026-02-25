"""
Celery application factory.
"""
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "tryon_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.queue.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    worker_max_tasks_per_child=1,   # restart worker after each task to free GPU memory
    worker_concurrency=settings.celery_max_concurrency,
    broker_connection_retry_on_startup=True,
)
