import logging
import sys

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("voclaire")

if settings.logtail_source_token:
    from logtail import LogtailHandler

    logtail_handler = LogtailHandler(
        source_token=settings.logtail_source_token,
        host=f"https://{settings.logtail_ingesting_host}",
    )
    logger.addHandler(logtail_handler)
