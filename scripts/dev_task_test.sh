#!/bin/bash

# dev_task_test.sh - Development script for testing task manager in different environments
# Created by Claude Code - 2025-03-02

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Aider Task Manager Development Testing Script${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Installing development dependencies...${NC}"
    pip install -e ".[dev]"
fi

# Parse command line arguments
SKIP_TESTS=false
SKIP_LINT=false
CUSTOM_ENV=""
TEST_DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-lint)
            SKIP_LINT=true
            shift
            ;;
        --custom-env)
            CUSTOM_ENV="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./dev_task_test.sh [--skip-tests] [--skip-lint] [--custom-env ENV_NAME] [--test-data DIR]"
            exit 1
            ;;
    esac
done

# Run linting if not skipped
if ! $SKIP_LINT; then
    echo -e "${PURPLE}Running linting checks...${NC}"
    python -m flake8 aider/taskmanager.py tests/basic/test_taskmanager.py tests/basic/test_task_integration.py
fi

# Run tests if not skipped
if ! $SKIP_TESTS; then
    echo -e "${PURPLE}Running task manager tests...${NC}"
    python -m pytest tests/basic/test_taskmanager.py tests/basic/test_task_integration.py -v
fi

# Setup test environment if specified
if [ -n "$CUSTOM_ENV" ]; then
    echo -e "${PURPLE}Setting up custom test environment: ${CUSTOM_ENV}${NC}"
    
    case $CUSTOM_ENV in
        "minimal")
            # Minimal environment with just core features
            export AIDER_TASK_FEATURES="basic"
            ;;
        "full")
            # Full feature set
            export AIDER_TASK_FEATURES="all"
            ;;
        "debug")
            # Debug mode with verbose logging
            export AIDER_DEBUG=1
            export AIDER_TASK_DEBUG=1
            ;;
        *)
            echo -e "${YELLOW}Unknown environment: ${CUSTOM_ENV}, using default settings${NC}"
            ;;
    esac
fi

# Create test data directory if specified
if [ -n "$TEST_DATA_DIR" ]; then
    echo -e "${PURPLE}Creating test data in: ${TEST_DATA_DIR}${NC}"
    mkdir -p "$TEST_DATA_DIR/.aider/tasks"
    
    # Create sample tasks for testing
    echo '{"id": "task-1", "title": "Test task 1", "description": "Sample task for testing", "status": "active", "created_at": "2025-03-02T12:00:00Z"}' > "$TEST_DATA_DIR/.aider/tasks/task-1.json"
    echo '{"id": "task-2", "title": "Test task 2", "description": "Another sample task", "status": "completed", "created_at": "2025-03-02T13:00:00Z", "completed_at": "2025-03-02T14:00:00Z"}' > "$TEST_DATA_DIR/.aider/tasks/task-2.json"
    
    echo -e "${GREEN}Created sample task data for testing${NC}"
fi

# Launch Aider with task manager enabled
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Starting Aider with task manager...${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}Quick commands:${NC}"
echo -e "${YELLOW}/task list${NC} - List all tasks"
echo -e "${YELLOW}/task create \"Task title\" \"Task description\"${NC} - Create a new task"
echo -e "${YELLOW}/task switch task-id${NC} - Switch to a specific task"
echo -e "${YELLOW}/task info task-id${NC} - Show task details"
echo -e "${BLUE}===============================================${NC}"

# Determine working directory
WORK_DIR="."
if [ -n "$TEST_DATA_DIR" ]; then
    WORK_DIR="$TEST_DATA_DIR"
fi

# Run Aider with task manager enabled
python -m aider.main --architect-auto-tasks --auto-test-tasks --auto-test-retry-limit 3 --work-dir "$WORK_DIR"

echo -e "${GREEN}Testing session completed.${NC}"