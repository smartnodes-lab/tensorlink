import json
import time
from pathlib import Path


class RateLimiter:
    def __init__(self, max_attempts_per_minute, block_duration, blacklist_file="blacklist.json"):
        self.rate_limit = {}
        self.max_attempts_per_minute = max_attempts_per_minute
        self.block_duration = block_duration
        self.blacklist_file = Path(blacklist_file)

        # Load existing blacklist
        self.blacklist = self.load_blacklist()

    def load_blacklist(self):
        """Load the blacklist from a file."""
        if self.blacklist_file.exists():
            with self.blacklist_file.open("r") as file:
                return json.load(file)
        return {}

    def save_blacklist(self):
        """Save the blacklist to a file."""
        with self.blacklist_file.open("w") as file:
            json.dump(self.blacklist, file)
