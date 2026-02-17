#!/bin/bash

# 1. Create and activate the virtual environment, then install dependencies
ENV_NAME=".venv"

echo "--- EEG Paradigm Setup ---"

if [ -d "$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' already exists. Updating dependencies..."
    source $ENV_NAME/bin/activate
else
    echo "Creating new environment..."
    if command -v uv &> /dev/null; then
        uv venv --python 3.10
        source $ENV_NAME/bin/activate
    else
        python3.10 -m venv $ENV_NAME
        source $ENV_NAME/bin/activate
        pip install --upgrade pip
    fi
fi

if command -v uv &> /dev/null; then
    uv pip install .
else
    pip install psychopy==2024.1.0 "numpy<2.0.0" pandas pyparallel pyyaml
fi

# 2. Create required directories
mkdir -p stimuli stimuli_orders data

echo ""
echo "--- Setup Complete ---"
echo "1. Activate the environment:  source $ENV_NAME/bin/activate"
echo "2. Generate the orders:      python generate_orders.py"
echo "3. Start the experiment:     python main.py"
echo "-----------------------"