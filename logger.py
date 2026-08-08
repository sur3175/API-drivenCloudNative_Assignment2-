# Configure application-wide logging to display timestamp, log level, and message in the console.
# The logger writes INFO-level and higher messages to standard output (stdout).

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
log = logging.getLogger(__name__)
