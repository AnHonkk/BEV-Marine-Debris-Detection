# run_all_tests.sh

#!/bin/bash

echo "========================================"
echo "CAMERA BEV MODULE - FULL TEST SUITE"
echo "========================================"

# Create results directory
mkdir -p results

# 1. Component tests
echo -e "\n[1/3] Running component tests..."
python tests/test_components.py

# 2. Integration tests
echo -e "\n[2/3] Running integration tests..."
python tests/test_integration.py

# 3. Visualization tests
echo -e "\n[3/3] Running visualization tests..."
python tests/visualize_results.py

echo -e "\n========================================"
echo "ALL TESTS COMPLETED!"
echo "========================================"
