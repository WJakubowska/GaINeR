#!/bin/bash
set -e

echo "🚀 Starting GaINeR container..."

# Check if OptiX is already built
if [ ! -f /workspace/gainer/gainer/knn/optix_knn.so ]; then
    echo "🔧 Building OptiX with GPU detection..."
    cd /workspace/gainer/gainer/knn
    ./build_optix.sh
    echo "✅ OptiX build complete"
else
    echo "✅ OptiX already built, skipping..."
fi

echo "📦 Installing GaINeR package..."
cd /workspace/gainer
pip install -e .
ns-install-cli
echo "✅ GaINeR installation complete"
echo "🎉 Container ready!"

# Execute the original command
exec "$@"
