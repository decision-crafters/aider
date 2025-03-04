"""
Task manager for aider
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Global singleton task manager instance
_task_manager_instance = None


@dataclass
class Environment:
    os: str
    python_version: str
    aider_version: str = ""


@dataclass
class TestInfo:
    failing_tests: List[str] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    attempt_count: int = 0


@dataclass
class Task:
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    status: str = "active"  # active, completed, archived
    parent_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    conversation_context: str = ""
    environment: Environment = field(default_factory=lambda: Environment(
        os=os.name,
        python_version=".".join(map(str, __import__("sys").version_info[:3])),
        aider_version="0.0.0"
    ))
    test_info: Optional[TestInfo] = None
    metadata: Dict = field(default_factory=dict)

    def add_files(self, files):
        """Add files to the task"""
        for file in files:
            if file not in self.files:
                self.files.append(file)

    def add_conversation_context(self, context):
        """Add conversation context to the task"""
        self.conversation_context = context


class TaskManager:
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the task manager"""
        self.tasks = {}
        self.active_task_id = None
        self.storage_dir = storage_dir or os.path.expanduser("~/.aider/tasks")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing tasks
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from storage"""
        if not os.path.exists(self.storage_dir):
            return
            
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.storage_dir, filename), "r") as f:
                        task_data = json.load(f)
                        task = Task(**task_data)
                        self.tasks[task.id] = task
                        
                        # Check if this is the active task
                        active_marker = os.path.join(self.storage_dir, "active_task")
                        if os.path.exists(active_marker):
                            with open(active_marker, "r") as f:
                                self.active_task_id = f.read().strip()
                except Exception as e:
                    print(f"Error loading task {filename}: {e}")

    def _save_task(self, task: Task):
        """Save a task to storage"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
            
        filename = os.path.join(self.storage_dir, f"{task.id}.json")
        with open(filename, "w") as f:
            json.dump(asdict(task), f)
            
        # Update active task marker
        if self.active_task_id:
            active_marker = os.path.join(self.storage_dir, "active_task")
            with open(active_marker, "w") as f:
                f.write(self.active_task_id)

    def create_task(self, name: str, description: str, parent_id: Optional[str] = None) -> Task:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        task = Task(
            id=task_id,
            name=name,
            description=description,
            created_at=timestamp,
            updated_at=timestamp,
            parent_id=parent_id,
        )
        
        self.tasks[task_id] = task
        self._save_task(task)
        
        # Set as active task if there's no active task
        if not self.active_task_id:
            self.switch_task(task_id)
            
        return task

    def update_task(self, task: Task):
        """Update a task"""
        if task.id not in self.tasks:
            raise ValueError(f"Task {task.id} not found")
            
        task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.tasks[task.id] = task
        self._save_task(task)
        
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def get_task_by_name(self, name: str) -> Optional[Task]:
        """Get a task by name"""
        for task in self.tasks.values():
            if task.name == name:
                return task
        return None

    def get_active_task(self) -> Optional[Task]:
        """Get the active task"""
        if not self.active_task_id:
            return None
        return self.tasks.get(self.active_task_id)

    def switch_task(self, task_id: str):
        """Switch to a different task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        self.active_task_id = task_id
        
        # Update active task marker
        active_marker = os.path.join(self.storage_dir, "active_task")
        with open(active_marker, "w") as f:
            f.write(task_id)

    def complete_task(self, task_id: str):
        """Mark a task as completed"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        task.status = "completed"
        task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_task(task)
        
        # If this was the active task, clear the active task
        if self.active_task_id == task_id:
            self.active_task_id = None
            
            # Remove active task marker
            active_marker = os.path.join(self.storage_dir, "active_task")
            if os.path.exists(active_marker):
                os.remove(active_marker)
        
        return task

    def archive_task(self, task_id: str):
        """Archive a task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        task.status = "archived"
        task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_task(task)
        
        # If this was the active task, clear the active task
        if self.active_task_id == task_id:
            self.active_task_id = None
            
            # Remove active task marker
            active_marker = os.path.join(self.storage_dir, "active_task")
            if os.path.exists(active_marker):
                os.remove(active_marker)
        
        return task

    def reactivate_task(self, task_id: str):
        """Reactivate a completed or archived task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        task.status = "active"
        task.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_task(task)
        
        return task

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List tasks, optionally filtered by status"""
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        else:
            return list(self.tasks.values())


def get_task_manager(storage_dir: Optional[str] = None) -> TaskManager:
    """Get the singleton task manager instance"""
    global _task_manager_instance
    
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager(storage_dir)
        
    return _task_manager_instance
