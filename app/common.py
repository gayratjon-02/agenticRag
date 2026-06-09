from enum import StrEnum


class Status(StrEnum):
    """Connectivity status of a downstream dependency."""

    OK = "ok"
    DOWN = "down"
