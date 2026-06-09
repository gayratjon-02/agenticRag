from celery import Celery

from app.config import get_settings


def create_celery() -> Celery:
    """Celery application skeleton. Tasks are registered in later phases."""
    settings = get_settings()
    celery = Celery(
        "agentic_rag",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.documents.tasks"],
    )
    celery.conf.update(task_track_started=True)
    return celery


celery_app = create_celery()
