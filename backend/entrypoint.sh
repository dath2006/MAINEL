#!/bin/bash
set -e

echo "=========================================="
echo "  MCMT-ReID Backend - Starting Up"
echo "=========================================="

# Check if requirements.txt has changed and install new packages
if [ -f /app/requirements.txt ]; then
    echo "[1/3] Checking for new Python packages..."
    pip install --quiet -r /app/requirements.txt 2>/dev/null || pip install -r /app/requirements.txt
    echo "      ✓ Packages synchronized"
else
    echo "[1/3] No requirements.txt found, skipping..."
fi

# Run database migrations
echo "[2/3] Running database migrations..."
if [ -f /app/alembic.ini ]; then
    alembic upgrade head 2>/dev/null && echo "      ✓ Migrations complete" || echo "      ⚠ Migration skipped (database may not be ready)"
else
    echo "      ⚠ No alembic.ini found, skipping migrations"
fi

# Display device configuration
echo "[3/3] Configuration:"
echo "      DEVICE: ${DEVICE:-cpu}"
echo "      DEBUG: ${DEBUG:-false}"
echo ""
echo "=========================================="
echo "  Starting uvicorn server..."
echo "=========================================="

# Execute the main command
exec "$@"
