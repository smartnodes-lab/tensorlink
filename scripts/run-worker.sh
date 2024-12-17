#!/bin/bash

VENV_PATH="venv"

# Error handling function
handle_error() {
    echo "Error: $1"
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    exit 1
}

# Function to compare versions
version_gt() {
    # Use Python to handle complex version comparisons
    python3 -c "from packaging import version; print(version.parse('$1') > version.parse('$2'))"
}

# Trap any unexpected errors
trap 'handle_error "Unexpected error occurred on line $LINENO"' ERR

# Check Python 3 availability
if ! command -v python3 &>/dev/null; then
    handle_error "Python 3 not installed. Please install Python 3 to continue."
fi

# Create virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating new environment..."
    if ! python3 -m venv "$VENV_PATH"; then
        handle_error "Failed to create virtual environment"
    fi
    echo "Virtual environment created successfully"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate" || handle_error "Failed to activate virtual environment"

# Install or upgrade Tensorlink
installed_version=$(pip show tensorlink 2>/dev/null | grep Version | awk '{print $2}')

if [ -z "$installed_version" ]; then
    echo "Tensorlink is not installed. Installing..."
    if ! pip install tensorlink; then
        handle_error "Failed to install Tensorlink"
    fi
else
    # Fetch the latest version
    latest_version=$(pip install tensorlink --dry-run 2>&1 | grep 'Requirement already satisfied: tensorlink' | grep -o '([^)]*)' | tr -d '()' | awk '{print $1}')

    # Compare versions using Python's version parsing
    if [ "$(version_gt "$latest_version" "$installed_version")" == "True" ]; then
        echo "Tensorlink is outdated (current: $installed_version). Upgrading to $latest_version..."
        if ! pip install --upgrade tensorlink; then
            handle_error "Failed to upgrade Tensorlink"
        fi
    else
        echo "Tensorlink is up-to-date (version: $installed_version)."
    fi
fi



# Run Tensorlink worker
echo "Starting worker..."
python run_worker.py

# Optional: Deactivate virtual environment
deactivate
