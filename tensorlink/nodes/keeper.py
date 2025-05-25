from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from datetime import datetime
import logging
import time
import json
import os

if TYPE_CHECKING:
    from tensorlink.p2p.smart_node import Smartnode

NETWORK_STATS = "logs/network_stats.json"
ALL_STATES = "logs/dht_state.json"
LATEST_STATE = "logs/latest_state.json"

CATEGORIES = ["workers", "validators", "users", "jobs", "proposals"]

# 30 days in seconds
THIRTY_DAYS_SECONDS = 60 * 60 * 24 * 30
ONE_DAY_SECONDS = 60 * 60 * 24
CLEAN_ARCHIVE_FREQ = 10


def _load_historical_stats() -> Dict:
    """Load historical statistics from file."""
    if os.path.exists(NETWORK_STATS):
        try:
            with open(NETWORK_STATS, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Default structure
    return {
        "daily": [],  # Last 90 days of data
        "weekly": [],  # Last 52 weeks of weekly aggregates
        "monthly": [],  # Last 1200 months (100 years) of monthly aggregates
        "jobs_archive": {},  # Jobs from last 30 days
    }


def _count_active_entities_for_date(entities_data: Dict, target_date: datetime) -> int:
    """
    Count entities that were active (last_seen) on or before the target date.

    Args:
        entities_data: Dictionary of entity_id -> entity_data
        target_date: The date to check activity for

    Returns:
        Count of active entities for that date
    """
    target_timestamp = target_date.timestamp()
    active_count = 0

    for entity_id, entity_data in entities_data.items():
        if entity_data and isinstance(entity_data, dict):
            last_seen = entity_data.get("last_seen")

            if last_seen is not None:
                try:
                    if isinstance(last_seen, str):
                        last_seen = float(last_seen)

                    # Count as active if last seen within 24 hours of target date
                    if abs(last_seen - target_timestamp) <= ONE_DAY_SECONDS:
                        active_count += 1
                except (ValueError, TypeError):
                    continue

    return active_count


def _save_latest_state(current_data):
    """Save current snapshot to latest state file."""
    os.makedirs(os.path.dirname(LATEST_STATE), exist_ok=True)
    with open(LATEST_STATE, "w") as f:
        json.dump(current_data, f, indent=4)


def _is_entity_current(entity_data):
    """Check if entity should be included based on last_seen timestamp."""
    if not entity_data or not isinstance(entity_data, dict):
        return True  # Include if no data or not a dict

    last_seen = entity_data.get("last_seen")
    if last_seen is None:
        return True  # Include if no last_seen field

    try:
        if isinstance(last_seen, str):
            last_seen = float(last_seen)
        current_time = time.time()
        return current_time - last_seen <= THIRTY_DAYS_SECONDS
    except (ValueError, TypeError):
        return True  # Include if timestamp is invalid (safer approach)


def _is_entity_active_today(entity_data, target_date: datetime) -> bool:
    """Check if an entity was active (last_seen) today."""
    if not entity_data or not isinstance(entity_data, dict):
        return False

    last_seen = entity_data.get("last_seen")
    if last_seen is None:
        return False

    try:
        if isinstance(last_seen, str):
            last_seen = float(last_seen)

        entity_date = datetime.fromtimestamp(last_seen).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return entity_date == target_date

    except (ValueError, TypeError):
        return False


class Keeper:
    def __init__(self, node: Smartnode):
        """
        Cleans up old data stored in node and keeps a log of network statistics that maintains:
        - Daily granularity for the last 90 days
        - Weekly aggregates for 1 year
        - Monthly aggregates for 100 years
        - Detailed job and node information from the last 30 days
        """
        self.node = node
        self.network_stats = _load_historical_stats()
        self.clean_ticker = 0
        self.last_daily_stats_update = 0

    def _archive_daily_to_weekly(self):
        """Archive daily statistics older than 90 days into weekly aggregates."""
        current_time = time.time()
        ninety_days_ago = current_time - THIRTY_DAYS_SECONDS * 3

        # Find daily stats that are older than 90 days and not yet archived
        daily_to_archive = [
            stat
            for stat in self.network_stats["daily"]
            if stat["timestamp"] < ninety_days_ago
        ]

        if not daily_to_archive:
            return

        # Group daily stats by week (ISO week)
        weekly_groups = {}
        for daily_stat in daily_to_archive:
            stat_date = datetime.fromtimestamp(daily_stat["timestamp"])
            # Get ISO year and week number
            iso_year, iso_week, _ = stat_date.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"

            if week_key not in weekly_groups:
                weekly_groups[week_key] = []
            weekly_groups[week_key].append(daily_stat)

        # Create weekly aggregates
        for week_key, daily_stats in weekly_groups.items():
            # Check if this week is already archived
            existing_weekly = next(
                (w for w in self.network_stats["weekly"] if w["week"] == week_key), None
            )

            if existing_weekly:
                continue  # Skip if already archived

            # Calculate weekly averages and totals
            week_start = min(stat["timestamp"] for stat in daily_stats)
            week_end = max(stat["timestamp"] for stat in daily_stats)

            weekly_stat = {
                "week": week_key,
                "week_start": week_start,
                "week_end": week_end,
                "days_count": len(daily_stats),
                # Average counts for the week
                "avg_workers": sum(stat["workers"] for stat in daily_stats)
                / len(daily_stats),
                "avg_validators": sum(stat["validators"] for stat in daily_stats)
                / len(daily_stats),
                "avg_users": sum(stat["users"] for stat in daily_stats)
                / len(daily_stats),
                "avg_jobs": sum(stat["jobs"] for stat in daily_stats)
                / len(daily_stats),
                "avg_proposals": sum(stat["proposals"] for stat in daily_stats)
                / len(daily_stats),
            }

            self.network_stats["weekly"].append(weekly_stat)

        # Remove archived daily stats
        self.network_stats["daily"] = [
            stat
            for stat in self.network_stats["daily"]
            if stat["timestamp"] >= ninety_days_ago
        ]

        # Sort weekly stats by week_start timestamp
        self.network_stats["weekly"].sort(key=lambda x: x["week_start"])

        # Keep only last 104 weeks (2 years) of weekly data
        if len(self.network_stats["weekly"]) > 104:
            self.network_stats["weekly"] = self.network_stats["weekly"][-104:]

        if daily_to_archive:
            self.node.debug_print(
                f"Archived {len(daily_to_archive)} daily stats into {len(weekly_groups)} weekly aggregates",
                level=logging.INFO,
                colour="blue",
                tag="Keeper",
            )

    def _update_daily_statistics(self):
        current_time = time.time()
        current_date = datetime.fromtimestamp(current_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Check if we already have stats for today
        today_str = current_date.strftime("%Y-%m-%d")
        existing_stat = next(
            (
                stat
                for stat in self.network_stats["daily"]
                if stat.get("date") == today_str
            ),
            None,
        )

        # Only update once per day and if we don't already have today's stats
        if existing_stat is not None:
            return

        if current_time - self.last_daily_stats_update < ONE_DAY_SECONDS:
            return

        self.last_daily_stats_update = current_time

        try:
            current_data = self._build_current_state()
            archive_data = self._load_existing_archive()

            # Merge current and archive data
            all_entities = {
                "workers": {
                    **archive_data.get("workers", {}),
                    **current_data["workers"],
                },
                "validators": {
                    **archive_data.get("validators", {}),
                    **current_data["validators"],
                },
                "users": {**archive_data.get("users", {}), **current_data["users"]},
                "jobs": {**archive_data.get("jobs", {}), **current_data["jobs"]},
                "proposals": {
                    **archive_data.get("proposals", {}),
                    **current_data["proposals"],
                },
            }

            daily_stat = {
                "date": current_date.strftime("%Y-%m-%d"),
                "timestamp": current_date.timestamp(),
                "workers": _count_active_entities_for_date(
                    all_entities["workers"], current_date
                ),
                "validators": _count_active_entities_for_date(
                    all_entities["validators"], current_date
                ),
                "users": _count_active_entities_for_date(
                    all_entities["users"], current_date
                ),
                "jobs": _count_active_entities_for_date(
                    all_entities["jobs"], current_date
                ),
                "proposals": _count_active_entities_for_date(
                    all_entities["proposals"], current_date
                ),
            }

            self.network_stats["daily"].append(daily_stat)
            self.network_stats["daily"].sort(key=lambda x: x["timestamp"])

            self._archive_daily_to_weekly()
            self._save_network_stats()

            self.node.debug_print(
                f"Daily statistics updated - Workers: {daily_stat['workers']}, "
                f"Validators: {daily_stat['validators']}, Users: {daily_stat['users']}, "
                f"Jobs: {daily_stat['jobs']}, Proposals: {daily_stat['proposals']}",
                level=logging.INFO,
                colour="cyan",
                tag="Keeper",
            )

        except Exception as e:
            self.node.debug_print(
                f"Error updating daily statistics: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def _save_network_stats(self):
        """Save network statistics to file."""
        os.makedirs(os.path.dirname(NETWORK_STATS), exist_ok=True)
        with open(NETWORK_STATS, "w") as f:
            json.dump(self.network_stats, f, indent=4)

    def get_daily_statistics(self, days: int = 30) -> list:
        """
        Get daily statistics for the last N days (max 90 days for daily data).

        Args:
            days: Number of days to retrieve (default: 30, max: 90)

        Returns:
            List of daily statistics dictionaries
        """
        if days <= 0:
            return []

        # Limit to 90 days since that's our daily retention
        days = min(days, 90)

        # Get the last N entries
        return self.network_stats["daily"][-days:]

    def get_weekly_statistics(self, weeks: int = 12) -> list:
        """
        Get weekly statistics for the last N weeks.

        Args:
            weeks: Number of weeks to retrieve (default: 12)

        Returns:
            List of weekly statistics dictionaries
        """
        if weeks <= 0:
            return []

        # Get the last N entries
        return self.network_stats["weekly"][-weeks:]

    def get_network_summary(self) -> Dict:
        """
        Get a summary of current network statistics.

        Returns:
            Dictionary with current counts and recent trends
        """
        try:
            current_data = self._build_current_state()
            archive_data = self._load_existing_archive()

            # Merge current and archive data for complete picture
            all_entities = {
                "workers": {
                    **archive_data.get("workers", {}),
                    **current_data["workers"],
                },
                "validators": {
                    **archive_data.get("validators", {}),
                    **current_data["validators"],
                },
                "users": {**archive_data.get("users", {}), **current_data["users"]},
                "jobs": {**archive_data.get("jobs", {}), **current_data["jobs"]},
                "proposals": {
                    **archive_data.get("proposals", {}),
                    **current_data["proposals"],
                },
            }

            current_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Current counts using the proper method
            current_counts = {
                "workers": _count_active_entities_for_date(
                    all_entities["workers"], current_date
                ),
                "validators": _count_active_entities_for_date(
                    all_entities["validators"], current_date
                ),
                "users": _count_active_entities_for_date(
                    all_entities["users"], current_date
                ),
                "jobs": _count_active_entities_for_date(
                    all_entities["jobs"], current_date
                ),
                "proposals": _count_active_entities_for_date(
                    all_entities["proposals"], current_date
                ),
            }

            # Recent trends (last 7 days and 4 weeks)
            recent_daily = self.get_daily_statistics(7)
            recent_weekly = self.get_weekly_statistics(4)

            return {
                "current": current_counts,
                "recent_daily": recent_daily,
                "recent_weekly": recent_weekly,
                "total_days_tracked": len(self.network_stats["daily"]),
                "total_weeks_archived": len(self.network_stats["weekly"]),
                "retention_policy": {"daily_days": 90, "weekly_weeks": 104},
            }

        except Exception as e:
            self.node.debug_print(
                f"Error generating network summary: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )
            return {"error": str(e)}

    def _filter_old_entities(self, entities_data: Dict) -> Dict:
        """
        Filter out entities that haven't been seen in the last 30 days.

        Args:
            entities_data: Dictionary of entity_id -> entity_data

        Returns:
            Filtered dictionary with only entities seen within 30 days
        """
        current_time = time.time()
        cutoff_time = current_time - THIRTY_DAYS_SECONDS
        filtered_data = {}

        for entity_id, entity_data in entities_data.items():
            if entity_data and isinstance(entity_data, dict):
                # Check if entity has last_seen field
                last_seen = entity_data.get("last_seen")

                if last_seen is not None:
                    try:
                        if isinstance(last_seen, str):
                            last_seen = float(last_seen)

                        # Keep entity if it was seen within the last 30 days
                        if last_seen >= cutoff_time:
                            filtered_data[entity_id] = entity_data
                        else:
                            self.node.debug_print(
                                f"Removing old entity {entity_id} (last seen: {time.ctime(last_seen)})",
                                level=logging.DEBUG,
                                colour="yellow",
                                tag="Keeper",
                            )
                    except (ValueError, TypeError):
                        self.node.debug_print(
                            f"Invalid last_seen timestamp for entity {entity_id}, keeping entity",
                            level=logging.WARNING,
                            colour="yellow",
                            tag="Keeper",
                        )
                        filtered_data[entity_id] = entity_data
                else:
                    # If no last_seen field, keep the entity (safest approach for now...)
                    filtered_data[entity_id] = entity_data
            else:
                # If entity_data is not a dict, keep it as is
                filtered_data[entity_id] = entity_data

        return filtered_data

    def clean_old_data(self):
        """
        Clean up entities that haven't been seen in the last 30 days from saved data.
        This affects both the latest state and historical archive.
        """
        try:
            # Clean latest state file
            if os.path.exists(LATEST_STATE):
                with open(LATEST_STATE, "r") as f:
                    latest_data = json.load(f)

                entities_cleaned = 0
                for category in CATEGORIES:
                    if category in latest_data:
                        original_count = len(latest_data[category])
                        latest_data[category] = self._filter_old_entities(
                            latest_data[category]
                        )
                        entities_cleaned += original_count - len(latest_data[category])

                # Save cleaned latest state
                with open(LATEST_STATE, "w") as f:
                    json.dump(latest_data, f, indent=4)

                self.node.debug_print(
                    f"Cleaned {entities_cleaned} old entities from latest state",
                    level=logging.INFO,
                    colour="green",
                    tag="Keeper",
                )

            # Clean historical archive
            if os.path.exists(ALL_STATES):
                with open(ALL_STATES, "r") as f:
                    archive_data = json.load(f)

                entities_cleaned = 0
                for category in CATEGORIES:
                    if category in archive_data:
                        original_count = len(archive_data[category])
                        archive_data[category] = self._filter_old_entities(
                            archive_data[category]
                        )
                        entities_cleaned += original_count - len(archive_data[category])

                # Save cleaned archive
                with open(ALL_STATES, "w") as f:
                    json.dump(archive_data, f, indent=4)

                self.node.debug_print(
                    f"Cleaned {entities_cleaned} old entities from historical archive",
                    level=logging.INFO,
                    colour="green",
                    tag="Keeper",
                )

        except Exception as e:
            self.node.debug_print(
                f"Error cleaning old data: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def write_state(self, latest_only=False):
        """Write current DHT state to files."""
        try:
            current_data = self._build_current_state()
            _save_latest_state(current_data)

            if not latest_only:
                self._update_historical_archive(current_data)
                self._update_daily_statistics()

            self._log_success(latest_only)

        except Exception as e:
            self.node.debug_print(
                f"Error saving DHT state: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def _build_current_state(self):
        """Build the current network state data structure."""
        current_data = {
            "workers": {},
            "validators": {},
            "users": {},
            "jobs": {},
            "proposals": {},
            "timestamp": time.time(),
        }

        # Load network entities
        self._load_network_entities(current_data)

        # Load proposals if available
        self._load_proposals(current_data)

        return current_data

    def _load_network_entities(self, current_data):
        """Load workers, validators, users, and jobs into current_data."""
        for category in CATEGORIES:
            if category != "proposals":
                collection = getattr(self.node, category)
                for entity_id in collection:
                    entity_data = self.node.dht.query(entity_id)
                    if _is_entity_current(entity_data):
                        current_data[category][entity_id] = entity_data

    def _load_proposals(self, current_data):
        """Load proposals into current_data if contract_manager exists."""
        if hasattr(self.node, "contract_manager") and self.node.contract_manager:
            for proposal_id in self.node.contract_manager.proposals:
                proposal_data = self.node.dht.query(proposal_id)
                current_data["proposals"][proposal_id] = proposal_data

    def _update_historical_archive(self, current_data):
        """Update historical archive with current snapshot."""
        os.makedirs(os.path.dirname(ALL_STATES), exist_ok=True)

        existing_data = self._load_existing_archive()

        # Update the archive with current data
        for category in ["workers", "validators", "users", "jobs", "proposals"]:
            existing_data[category].update(current_data[category])

        # Save updated archive data
        with open(ALL_STATES, "w") as f:
            json.dump(existing_data, f, indent=4)

    def _load_existing_archive(self):
        """Load existing archive data or return default structure."""
        default_data = {
            "workers": {},
            "validators": {},
            "users": {},
            "jobs": {},
            "proposals": {},
        }

        if not os.path.exists(ALL_STATES):
            return default_data

        try:
            with open(ALL_STATES, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.node.debug_print(
                "Existing state file read error.",
                level=logging.ERROR,
                colour="red",
                tag="Keeper",
            )
            return default_data

    def _log_success(self, latest_only):
        """Log successful state save operation."""
        file_desc = "latest file only" if latest_only else "both files"
        self.node.debug_print(
            f"DHT state saved successfully to {file_desc}.",
            level=logging.INFO,
            colour="green",
            tag="Keeper",
        )

    def load_previous_state(self):
        """Load the DHT state from a file."""
        if os.path.exists(LATEST_STATE):
            try:
                with open(LATEST_STATE, "r") as f:
                    state = json.load(f)

                # Restructure state: list only hash and corresponding data
                structured_state = {}
                for category, items in state.items():
                    if category != "timestamp":
                        # Filter old entities during load as well
                        filtered_items = self._filter_old_entities(items)
                        structured_state[category] = {
                            hash_key: data for hash_key, data in filtered_items.items()
                        }
                        self.node.dht.routing_table.update(filtered_items)

                self.node.debug_print(
                    "DHT state loaded successfully.",
                    level=logging.INFO,
                    tag="Keeper",
                )

            except Exception as e:
                self.node.debug_print(
                    f"Error loading DHT state: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                    tag="Keeper",
                )
        else:
            self.node.debug_print(
                "No DHT state file found.",
                level=logging.INFO,
                tag="Keeper",
            )

    def clean_node(self):
        """Clean up inactive nodes"""

        def clean_nodes(nodes):
            nodes_to_remove = []
            for node_id in nodes:
                # Remove any ghost ids in the list
                if node_id not in self.node.nodes:
                    nodes_to_remove.append(node_id)

                # Remove any terminated connections
                elif self.node.nodes[node_id].terminate_flag.is_set():
                    role = self.node.nodes[node_id].role
                    nodes_to_remove.append(node_id)
                    del self.node.nodes[node_id]

                    if role == "W" and hasattr(self.node, "all_workers"):
                        del self.node.all_workers[node_id]

            for node in nodes_to_remove:
                nodes.remove(node)

            # TODO method / request to delete job after certain time or by request of the user.
            #   Perhaps after a job is finished there is a delete request

        clean_nodes(self.node.workers)
        clean_nodes(self.node.validators)
        clean_nodes(self.node.users)

        # After cleaning nodes, also clean old data from saved files
        if self.clean_ticker == CLEAN_ARCHIVE_FREQ:
            self.clean_old_data()
            self.clean_ticker = 0

        self.clean_ticker += 1
