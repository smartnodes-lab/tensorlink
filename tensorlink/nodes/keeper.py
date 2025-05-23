from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import logging
import time
import json
import os

if TYPE_CHECKING:
    from tensorlink.p2p.smart_node import Smartnode

NETWORK_STATS = "logs/network_stats.json"
ALL_STATES = "logs/dht_state.json"
LATEST_STATE = "logs/latest_state.json"

# 30 days in seconds
THIRTY_DAYS_SECONDS = 30 * 24 * 60 * 60


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
        "hourly": [],  # Last 30 days of hourly data
        "weekly": [],  # Last 52 weeks of weekly aggregates
        "monthly": [],  # Last 1200 months (100 years) of monthly aggregates
        "jobs_archive": {},  # Jobs from last 30 days
    }


class Keeper:
    def __init__(self, node: Smartnode):
        """
        Cleans up old data stored in node and keeps a log of network statistics that maintains:
        - Hourly granularity for the last 30 days
        - Weekly aggregates for 1 year
        - Monthly aggregates for 100 years
        - Job Data from the last 30 days
        - Worker, User, and Validator information
        """
        self.node = node
        self.network_stats = _load_historical_stats()

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
                        # Convert to float if it's a string timestamp
                        if isinstance(last_seen, str):
                            last_seen = float(last_seen)

                        # Keep entity if it was seen within the last 30 days
                        if last_seen >= cutoff_time:
                            filtered_data[entity_id] = entity_data
                        else:
                            self.node.debug_print(
                                f"Removing old entity {entity_id} (last seen: {time.ctime(last_seen)})",
                                level=logging.INFO,
                                colour="yellow",
                                tag="Keeper",
                            )
                    except (ValueError, TypeError):
                        # If last_seen is not a valid timestamp, keep the entity (safer approach)
                        self.node.debug_print(
                            f"Invalid last_seen timestamp for entity {entity_id}, keeping entity",
                            level=logging.WARNING,
                            colour="yellow",
                            tag="Keeper",
                        )
                        filtered_data[entity_id] = entity_data
                else:
                    # If no last_seen field, keep the entity (safer approach)
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
                for category in ["workers", "validators", "users", "jobs", "proposals"]:
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
                for category in ["workers", "validators", "users", "jobs", "proposals"]:
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
            self._save_latest_state(current_data)

            if not latest_only:
                self._update_historical_archive(current_data)

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
        for category in ["workers", "validators", "users", "jobs"]:
            collection = getattr(self.node, category)
            for entity_id in collection:
                entity_data = self.node.dht.query(entity_id)
                if self._is_entity_current(entity_data):
                    current_data[category][entity_id] = entity_data

    def _load_proposals(self, current_data):
        """Load proposals into current_data if contract_manager exists."""
        if hasattr(self.node, "contract_manager") and self.node.contract_manager:
            for proposal_id in self.node.contract_manager.proposals:
                proposal_data = self.node.dht.query(proposal_id)
                current_data["proposals"][proposal_id] = proposal_data

    def _is_entity_current(self, entity_data):
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

    def _save_latest_state(self, current_data):
        """Save current snapshot to latest state file."""
        os.makedirs(os.path.dirname(LATEST_STATE), exist_ok=True)
        with open(LATEST_STATE, "w") as f:
            json.dump(current_data, f, indent=4)

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
        self.clean_old_data()
