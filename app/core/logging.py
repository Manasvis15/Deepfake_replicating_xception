"""Logging setup: JSON lines (prod) or plain text (dev), each tagged with the request ID."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# Standard LogRecord attributes, plus uvicorn's ANSI-coloured duplicate of the message.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "request_id", "color_message"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Keep an explicitly passed request_id (e.g. from an error handler running after
        # the request context was reset); otherwise take it from the current context.
        if not hasattr(record, "request_id"):
            record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) becomes a JSON field.
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Let uvicorn's own loggers flow through our handler instead of duplicating output.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = name != "uvicorn.access"  # we log access ourselves
