"""Entry point: start the Nexus agent + WebUI server."""
import sys
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexus.api.server import app
from nexus.memory import database as db
from nexus import heartbeat

logger = logging.getLogger(__name__)


async def _ensure_vision_deps() -> None:
    """Verify local vision (Florence-2) dependencies are available.

    Prints a clear message if something is missing so the user knows
    to install it rather than hitting an ImportError at runtime.
    """
    _missing: list[str] = []
    _check_packages = [
        ("transformers", "transformers", None),
        ("PIL", "Pillow", "Pillow"),
        ("einops", "einops", None),
        ("timm", "timm", None),
        ("accelerate", "accelerate", "accelerate (optional but recommended)"),
    ]
    for import_name, package, label in _check_packages:
        try:
            __import__(import_name)
        except ImportError:
            _missing.append(label or package)

    if _missing:
        missing_list = ", ".join(_missing)
        logger.warning(
            "Local vision (Florence-2) will not be available — "
            "missing packages: %s. Install with: "
            "pip install %s",
            missing_list, " ".join(_missing),
        )
        print(
            f"[run.py] ⚠ Local vision (Florence-2) unavailable — "
            f"install: pip install {' '.join(_missing)}"
        )
    else:
        logger.info("Local vision (Florence-2) dependencies: OK")
        # Optionally trigger a quick smoke-import to pre-warm the cache
        # (the actual model download happens on first use)
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
            )
            logger.debug("Transformers import for Florence-2: OK")
        except Exception as e:
            logger.warning(
                "Transformers import for Florence-2 succeeded but "
                "a deeper error occurred: %s", e
            )


async def _init_db_with_retry(max_retries: int = 5, delay: float = 1.0) -> None:
    """Initialize the SQLite database (idempotent, fast on subsequent calls)."""
    for attempt in range(1, max_retries + 1):
        try:
            await db.init()
            logger.info("Database connected successfully (attempt %d/%d)",
                        attempt, max_retries)
            return
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Failed to connect to database after %d attempts: %s",
                    max_retries, exc,
                )
                raise
            wait = delay * attempt
            logger.warning(
                "Database connection failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_retries, exc, wait,
            )
            await asyncio.sleep(wait)


@asynccontextmanager
async def lifespan(app):
    """Application lifespan: initialize DB and start heartbeat.

    NOTE: server.py also registers @app.on_event("startup"/"shutdown")
    handlers — those are idempotent and safe to run alongside this lifespan.
    """
    # Initialize SQLite database and schema (with retries)
    await _init_db_with_retry()

    # Start Celery beat scheduler (heartbeat replacement)
    try:
        await heartbeat.start()
    except Exception as exc:
        logger.warning("Heartbeat/Celery failed to start (non-fatal): %s", exc)

    # Check vision dependencies (warmly awaited)
    try:
        await _ensure_vision_deps()
    except Exception as exc:
        logger.warning("Vision dependency check failed: %s", exc)

    yield

    # Shutdown
    try:
        await heartbeat.stop()
    except Exception:
        pass
    try:
        await db.close()
    except Exception:
        pass
    try:
        from nexus.core import llm
        await llm.shutdown_client()
    except Exception:
        pass


if __name__ == "__main__":
    # Patch the app lifespan — @asynccontextmanager makes it a proper async context manager
    app.router.lifespan_context = lifespan

    uvicorn.run(
        "run:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=300,
    )
