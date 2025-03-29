import time


class ConnectionMonitor:
    def __init__(self, max_attempts_per_minute=5, block_duration=300):
        self.rate_limit = {}  # Tracks IP attempts
        self.max_attempts_per_minute = max_attempts_per_minute
        self.block_duration = block_duration

    def is_blocked(self, ip_address):
        """Check if an IP is currently blocked."""
        current_time = int(time.time())
        block_info = self.rate_limit.get(ip_address, {"blocked_until": 0})
        return block_info["blocked_until"] > current_time

    def record_attempt(self, ip_address):
        """Record an attempt and block IP if needed."""
        current_time = time.time()
        if ip_address not in self.rate_limit:
            self.rate_limit[ip_address] = {
                "attempts": 0,
                "last_attempt": 0,
                "blocked_until": 0,
            }

        block_info = self.rate_limit[ip_address]

        # Reset count if over a minute since last attempt
        if current_time - block_info["last_attempt"] > 60:
            self.rate_limit[ip_address] = {
                "attempts": 0,
                "last_attempt": 0,
                "blocked_until": 0,
            }

        self.rate_limit[ip_address]["attempts"] += 1
        self.rate_limit[ip_address]["last_attempt"] = current_time

        # Block IP if it exceeds the limit
        if self.rate_limit[ip_address]["attempts"] > self.max_attempts_per_minute:
            self.rate_limit[ip_address]["blocked_until"] = round(
                current_time + self.block_duration
            )
