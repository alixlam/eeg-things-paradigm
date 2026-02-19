#!/bin/bash

ENV_NAME=".venv"

echo "--- EEG Paradigm Setup ---"

# 1. Environment Creation
if [ -d "$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' already exists."
    source $ENV_NAME/bin/activate
else
    echo "Creating new environment..."
    uv venv .venv --python 3.10
    source .venv/bin/activate
fi

# 2. Install dependencies
# We use -e . (editable) or just list them to avoid the 'BackendUnavailable' build error
echo "Installing dependencies from pyproject.toml..."
if command -v uv &> /dev/null; then
    uv pip install psychopy==2023.2.3 --no-deps
    uv pip install .
else
    # This installs the requirements directly
    pip install . 
fi
# 2. Create required directories
mkdir -p stimuli stimuli_orders data

echo ""
echo "--- Setup Complete ---"
echo "1. Activate the environment:  source $ENV_NAME/bin/activate"
echo "2. Generate the orders:      python generate_orders.py"
echo "3. Start the experiment:     python main.py"
echo "-----------------------"