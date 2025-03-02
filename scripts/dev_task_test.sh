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

# Check for venv deletion flag
if [ "$1" == "--delete-venv" ]; then
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Deleting existing virtual environment as requested...${NC}"
        rm -rf venv
        shift # Remove this argument from the parameters
    fi
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Creating one with $PYTHON_BIN...${NC}"
    $PYTHON_BIN -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    echo -e "${GREEN}Upgrading pip...${NC}"
    pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    echo -e "${GREEN}Installing core dependencies...${NC}"
    pip install -r requirements.txt
    
    # Install package in development mode
    echo -e "${GREEN}Installing aider in development mode...${NC}"
    pip install -e .
    
    # Install test dependencies
    echo -e "${GREEN}Installing test dependencies...${NC}"
    pip install pytest flake8
fi

# Parse command line arguments
SKIP_TESTS=false
SKIP_LINT=false
CUSTOM_ENV=""
TEST_DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --delete-venv)
            # Already handled earlier, but need to include here for help message
            shift
            ;;
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
            echo "Usage: ./dev_task_test.sh [--delete-venv] [--skip-tests] [--skip-lint] [--custom-env ENV_NAME] [--test-data DIR]"
            exit 1
            ;;
    esac
done

# Run linting if not skipped
if ! $SKIP_LINT; then
    echo -e "${PURPLE}Running linting checks...${NC}"
    
    # Check if files exist before linting
    LINT_FILES=""
    if [ -f "aider/taskmanager.py" ]; then
        LINT_FILES="$LINT_FILES aider/taskmanager.py"
    else
        echo -e "${YELLOW}aider/taskmanager.py not found. Skipping linting for this file.${NC}"
    fi
    
    if [ -f "tests/basic/test_taskmanager.py" ]; then
        LINT_FILES="$LINT_FILES tests/basic/test_taskmanager.py"
    else
        echo -e "${YELLOW}tests/basic/test_taskmanager.py not found. Skipping linting for this file.${NC}"
    fi
    
    if [ -f "tests/basic/test_task_integration.py" ]; then
        LINT_FILES="$LINT_FILES tests/basic/test_task_integration.py"
    else
        echo -e "${YELLOW}tests/basic/test_task_integration.py not found. Skipping linting for this file.${NC}"
    fi
    
    if [ -n "$LINT_FILES" ]; then
        python -m flake8 $LINT_FILES
    else
        echo -e "${YELLOW}No files found to lint. This is expected if the task manager implementation is still in progress.${NC}"
    fi
fi

# Run tests if not skipped
if ! $SKIP_TESTS; then
    if [ -f "tests/basic/test_taskmanager.py" ] && [ -f "tests/basic/test_task_integration.py" ]; then
        echo -e "${PURPLE}Running task manager tests...${NC}"
        python -m pytest tests/basic/test_taskmanager.py tests/basic/test_task_integration.py -v
    else
        echo -e "${YELLOW}Task manager test files not found. Skipping tests.${NC}"
        echo -e "${YELLOW}This is expected if the task manager implementation is still in progress.${NC}"
    fi
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

# Check if the taskmanager module exists
if [ ! -f "aider/taskmanager.py" ]; then
    echo -e "${RED}Error: aider/taskmanager.py not found!${NC}"
    echo -e "${YELLOW}The task manager implementation appears to be missing.${NC}"
    echo -e "${YELLOW}Please create this file first, then run the script again.${NC}"
    
    # Create skeleton file structure
    read -p "Would you like to create a skeleton taskmanager.py file to get started? (y/n) " ANSWER
    if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
        mkdir -p aider
        cat > aider/taskmanager.py << 'EOF'
"""
Task Manager for Aider - Manages tasks with persistent memory across sessions
"""

import os
import json
import time
import uuid
import platform
import datetime
from typing import Dict, List, Optional, Set, Any, Union

class TestInfo:
    """Information about test runs associated with a task"""
    
    def __init__(self):
        """Initialize test info tracking"""
        self.failures = []
        self.attempt_count = 0
        self.last_run_timestamp = None
        self.success_count = 0
        
    def record_failure(self, test_name: str, error_message: str) -> None:
        """Record a test failure"""
        self.failures.append({
            "test_name": test_name,
            "error_message": error_message,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self.attempt_count += 1
        self.last_run_timestamp = datetime.datetime.now().isoformat()
        
    def record_success(self) -> None:
        """Record a successful test run"""
        self.failures = []  # Clear failures on success
        self.success_count += 1
        self.last_run_timestamp = datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "failures": self.failures,
            "attempt_count": self.attempt_count,
            "last_run_timestamp": self.last_run_timestamp,
            "success_count": self.success_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestInfo':
        """Create a TestInfo instance from a dictionary"""
        test_info = cls()
        test_info.failures = data.get("failures", [])
        test_info.attempt_count = data.get("attempt_count", 0)
        test_info.last_run_timestamp = data.get("last_run_timestamp")
        test_info.success_count = data.get("success_count", 0)
        return test_info

class Environment:
    """Environment information for a task"""
    
    def __init__(self):
        """Initialize with current environment information"""
        self.os_name = platform.system()
        self.os_version = platform.version()
        self.python_version = platform.python_version()
        self.working_directory = os.getcwd()
        self.creation_timestamp = datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization"""
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "working_directory": self.working_directory,
            "creation_timestamp": self.creation_timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Environment':
        """Create an Environment instance from a dictionary"""
        env = cls()
        env.os_name = data.get("os_name", platform.system())
        env.os_version = data.get("os_version", platform.version())
        env.python_version = data.get("python_version", platform.python_version())
        env.working_directory = data.get("working_directory", os.getcwd())
        env.creation_timestamp = data.get("creation_timestamp", datetime.datetime.now().isoformat())
        return env

class Task:
    """Represents a task with persistent memory"""
    
    def __init__(self, title: str, description: str = "", parent_id: Optional[str] = None):
        """Initialize a new task"""
        self.id = str(uuid.uuid4())[:8]  # Short unique ID
        self.title = title
        self.description = description
        self.status = "active"  # active, completed, archived
        self.created_at = datetime.datetime.now().isoformat()
        self.completed_at = None
        self.archived_at = None
        self.parent_id = parent_id
        self.child_ids = []
        self.environment = Environment()
        self.test_info = TestInfo()
        self.files = []  # List of files associated with this task
        self.conversation_history = []  # History of conversations related to this task
        
    def complete(self) -> None:
        """Mark the task as completed"""
        self.status = "completed"
        self.completed_at = datetime.datetime.now().isoformat()
        
    def archive(self) -> None:
        """Archive the task"""
        self.status = "archived"
        self.archived_at = datetime.datetime.now().isoformat()
        
    def reactivate(self) -> None:
        """Reactivate a completed or archived task"""
        self.status = "active"
        self.completed_at = None
        self.archived_at = None
        
    def add_child(self, child_id: str) -> None:
        """Add a child task ID to this task"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            
    def remove_child(self, child_id: str) -> None:
        """Remove a child task ID from this task"""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
            
    def add_file(self, file_path: str) -> None:
        """Associate a file with this task"""
        if file_path not in self.files:
            self.files.append(file_path)
            
    def remove_file(self, file_path: str) -> None:
        """Remove a file association from this task"""
        if file_path in self.files:
            self.files.remove(file_path)
            
    def add_conversation(self, message: str) -> None:
        """Add a conversation message to the task history"""
        self.conversation_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "archived_at": self.archived_at,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "environment": self.environment.to_dict(),
            "test_info": self.test_info.to_dict(),
            "files": self.files,
            "conversation_history": self.conversation_history
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a Task instance from a dictionary"""
        task = cls(data["title"], data.get("description", ""))
        task.id = data.get("id", task.id)
        task.status = data.get("status", "active")
        task.created_at = data.get("created_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.archived_at = data.get("archived_at")
        task.parent_id = data.get("parent_id")
        task.child_ids = data.get("child_ids", [])
        
        if "environment" in data:
            task.environment = Environment.from_dict(data["environment"])
            
        if "test_info" in data:
            task.test_info = TestInfo.from_dict(data["test_info"])
            
        task.files = data.get("files", [])
        task.conversation_history = data.get("conversation_history", [])
        
        return task

# Singleton instance
_task_manager_instance = None

class TaskManager:
    """Manages tasks with persistent storage"""
    
    def __init__(self, tasks_dir: Optional[str] = None):
        """Initialize the task manager"""
        if tasks_dir is None:
            self.tasks_dir = os.path.join(os.getcwd(), ".aider", "tasks")
        else:
            self.tasks_dir = tasks_dir
            
        os.makedirs(self.tasks_dir, exist_ok=True)
        self.tasks = {}
        self.active_task_id = None
        self._load_tasks()
        
    def _load_tasks(self) -> None:
        """Load tasks from disk"""
        task_files = [f for f in os.listdir(self.tasks_dir) if f.endswith(".json")]
        
        for task_file in task_files:
            try:
                with open(os.path.join(self.tasks_dir, task_file), "r") as f:
                    task_data = json.load(f)
                    task = Task.from_dict(task_data)
                    self.tasks[task.id] = task
            except Exception as e:
                print(f"Error loading task {task_file}: {e}")
                
    def _save_task(self, task_id: str) -> None:
        """Save a task to disk"""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task_path = os.path.join(self.tasks_dir, f"{task_id}.json")
        
        with open(task_path, "w") as f:
            json.dump(task.to_dict(), f, indent=2)
            
    def create_task(self, title: str, description: str = "", parent_id: Optional[str] = None) -> str:
        """Create a new task and return its ID"""
        task = Task(title, description, parent_id)
        self.tasks[task.id] = task
        
        # Update parent if specified
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].add_child(task.id)
            self._save_task(parent_id)
            
        self._save_task(task.id)
        return task.id
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
        
    def set_active_task(self, task_id: str) -> bool:
        """Set the active task"""
        if task_id in self.tasks:
            self.active_task_id = task_id
            return True
        return False
        
    def get_active_task(self) -> Optional[Task]:
        """Get the currently active task"""
        if self.active_task_id:
            return self.tasks.get(self.active_task_id)
        return None
        
    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by status"""
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        return list(self.tasks.values())

# Get the global task manager instance
def get_task_manager(tasks_dir: Optional[str] = None) -> TaskManager:
    """Get the singleton task manager instance"""
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager(tasks_dir)
    return _task_manager_instance
EOF
        echo -e "${GREEN}Created skeleton taskmanager.py file.${NC}"
    fi
    
    # Exit with error
    exit 1
fi

# Check for common dependencies needed by Aider
missing_deps=false
for dep in "packaging" "litellm" "openai" "configargparse" "diskcache" "pydantic"; do
    if ! python -c "import $dep" &>/dev/null; then
        echo -e "${YELLOW}Missing dependency: $dep. Installing...${NC}"
        pip install $dep
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install $dep${NC}"
            missing_deps=true
        fi
    fi
done

# Abort if any critical dependencies couldn't be installed
if $missing_deps; then
    echo -e "${RED}Some dependencies could not be installed. Please fix the issues and try again.${NC}"
    exit 1
fi

# Run Aider with task manager enabled
echo -e "${GREEN}Launching Aider with task manager...${NC}"

# Save current directory
ORIGINAL_DIR=$(pwd)

# Change to the working directory and run Aider
cd "$WORK_DIR"
python -m aider.main --architect-auto-tasks --auto-test-tasks --auto-test-retry-limit 3

# Return to original directory when done
cd "$ORIGINAL_DIR"

echo -e "${GREEN}Testing session completed.${NC}"