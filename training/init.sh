#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Define environment name and Python version
ENV_NAME="yopo"
PYTHON_VERSION="3.12"

echo "🚀 Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"

# Create the environment
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "✅ Environment '$ENV_NAME' created successfully."

# Activate the environment
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "📦 Installing JupyterLab and Notebook from conda-forge..."
conda install -y -c conda-forge jupyterlab notebook

echo "🎉 Setup complete!"
echo "To start using it, run:"
echo "    conda activate $ENV_NAME"
echo "    jupyter lab"

