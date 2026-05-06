"""Re-export; prefer :mod:`bbo.tasks.http_json` for new code."""

from ..http_json import get_json, post_json

__all__ = ["get_json", "post_json"]
