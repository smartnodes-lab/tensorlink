#!/bin/bash

VENV_PATH="venv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Error handling function
handle_error() {
    echo "Error: $1"
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    exit 1
}

# Function to get latest version from PyPI
get_latest_version() {
    python3 -c "
import requests
import json
try:
    response = requests.get('https://pypi.org/pypi/tensorlink/json', timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(data['info']['version'])
    else:
        print('ERROR')
except:
    print('ERROR')
"
}

# Function to compare versions
version_gt() {
    python3 -c "
try:
    from packaging import version
    print(version.parse('$1') > version.parse('$2'))
except ImportError:
    # Fallback to simple string comparison if packaging not available
    print('$1' != '$2')
"
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

# Install requests if not available (needed for version checking)
if ! python3 -c "import requests" &>/dev/null; then
    echo "Installing requests for version checking..."
    pip install requests
fi

# Install packaging if not available (needed for version comparison)
if ! python3 -c "import packaging" &>/dev/null; then
    echo "Installing packaging for version comparison..."
    pip install packaging
fi

# Get current installed version
installed_version=$(pip show tensorlink 2>/dev/null | grep Version | awk '{print $2}')

# Always try to update to latest version
echo "Checking for latest Tensorlink version..."
latest_version=$(get_latest_version)

if [ "$latest_version" = "ERROR" ]; then
    echo "Warning: Could not fetch latest version from PyPI. Attempting upgrade anyway..."
    if [ -z "$installed_version" ]; then
        echo "Tensorlink is not installed. Installing..."
        if ! pip install tensorlink; then
            handle_error "Failed to install Tensorlink"
        fi
    else
        echo "Attempting to upgrade Tensorlink..."
        if ! pip install --upgrade tensorlink; then
            handle_error "Failed to upgrade Tensorlink"
        fi
    fi
else
    if [ -z "$installed_version" ]; then
        echo "Tensorlink is not installed. Installing version $latest_version..."
        if ! pip install tensorlink==$latest_version; then
            handle_error "Failed to install Tensorlink"
        fi
    else
        # Compare versions
        if [ "$(version_gt "$latest_version" "$installed_version")" == "True" ]; then
            echo "Upgrading Tensorlink from $installed_version to $latest_version..."
            if ! pip install --upgrade tensorlink==$latest_version; then
                handle_error "Failed to upgrade Tensorlink"
            fi
        else
            echo "Tensorlink is already at the latest version ($installed_version)."
        fi
    fi
fi

# Verify installation
final_version=$(pip show tensorlink 2>/dev/null | grep Version | awk '{print $2}')
if [ -n "$final_version" ]; then
    echo "Tensorlink is now installed at version: $final_version"
else
    handle_error "Tensorlink installation verification failed"
fi

# Check if script is run with sudo
RUN_AS_SUDO=""
if [ "$EUID" -eq 0 ]; then
    echo "Script running with sudo privileges."
    RUN_AS_SUDO="sudo"
fi

# Run Tensorlink worker
echo "Starting worker..."
$RUN_AS_SUDO python run_worker.py

deactivate
