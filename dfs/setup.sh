#!/bin/bash

# DFS Project Setup Script
# Run this script when setting up the project on a new machine

set -e  # Exit on error

echo "========================================="
echo "DFS Fantasy Project Setup"
echo "========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed"
else
    echo "✓ uv already installed"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing project dependencies..."
uv pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p lineups
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
# DFS Environment Configuration
DATABASE_PATH=data/nfl_dfs.db
MODEL_DIR=models/
LINEUP_DIR=lineups/
LOG_LEVEL=INFO
EOF
    echo "✓ .env file created"
else
    echo "✓ .env file already exists"
fi

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "
from data import initialize_database
import os
db_path = 'data/nfl_dfs.db'
if not os.path.exists(db_path):
    initialize_database(db_path)
    print('✓ Database initialized')
else:
    print('✓ Database already exists')
"

# Download initial data (optional - commented out by default)
# echo "📊 Downloading initial NFL data (2023-2024 seasons)..."
# uv run python run.py collect --seasons 2023 2024

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "Activate the virtual environment: source .venv/bin/activate"
echo "uv run python run.py train --positions RB --tune-all --trials 100"
echo ""
echo "For help: uv run python run.py --help"
