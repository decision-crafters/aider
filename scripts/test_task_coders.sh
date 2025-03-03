#!/bin/bash

# test_task_coders.sh - Script for testing task functionality in coders
# This script tests how different coders handle task functionality

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Aider Task Coder Testing Script${NC}"
echo -e "${BLUE}===============================================${NC}"

# Find compatible Python version (prefer 3.11 if available)
PYTHON_BIN="python3"
if command -v python3.11 > /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 > /dev/null; then
    PYTHON_BIN="python3.10"
elif command -v python3.9 > /dev/null; then
    PYTHON_BIN="python3.9"
fi

echo -e "${GREEN}Using Python: $($PYTHON_BIN --version)${NC}"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Creating one with $PYTHON_BIN...${NC}"
    $PYTHON_BIN -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -e .
fi

# Create test directory
TEST_DIR=$(mktemp -d)
echo -e "${GREEN}Created temporary test directory: ${TEST_DIR}${NC}"

# Create test files in the test directory
cat > "${TEST_DIR}/test_file.py" << 'EOF'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def main():
    print(add(5, 3))
    print(subtract(10, 4))

if __name__ == "__main__":
    main()
EOF

# Function to test a specific coder with tasks
test_coder() {
    local coder_type=$1
    local test_name="$2"
    
    echo -e "${PURPLE}Testing ${coder_type} coder with task: ${test_name}${NC}"
    
    # Create a Python script to test the coder with tasks
    cat > "${TEST_DIR}/test_${coder_type}_coder.py" << EOF
from aider.io import InputOutput
from aider.coders.${coder_type}_coder import ${coder_type^}Coder
from aider.taskmanager import get_task_manager

# Setup
io = InputOutput(yes=True)
coder = ${coder_type^}Coder(io=io, show_timings=False)

# Ensure task manager is initialized
task_manager = get_task_manager("${TEST_DIR}/.aider/tasks")

# Create a test task
task_id = task_manager.create_task("${test_name}", "Test task for ${coder_type} coder")
task_manager.switch_task(task_id)

# Test task awareness
active_task = task_manager.get_active_task()
print(f"Active task: {active_task.title} (ID: {active_task.id})")

# Add file to task
task_manager.add_file_to_task(active_task.id, "test_file.py")

# Print task info
task = task_manager.get_task(task_id)
print(f"Files in task: {task.files}")

# Print coder info
print(f"Coder has task_manager: {coder.task_manager is not None}")
print(f"Coder active_task: {coder.active_task is not None}")
EOF
    
    # Run the test
    $PYTHON_BIN "${TEST_DIR}/test_${coder_type}_coder.py"
    
    echo -e "${GREEN}Test for ${coder_type} coder completed${NC}"
    echo
}

# Test both Ask and Architect coders
test_coder "ask" "Test Ask Coder Tasks"
test_coder "architect" "Test Architect Coder Tasks"

# Clean up
echo -e "${YELLOW}Cleaning up test directory: ${TEST_DIR}${NC}"
rm -rf "${TEST_DIR}"

echo -e "${GREEN}All tests completed successfully${NC}"