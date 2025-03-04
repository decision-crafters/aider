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
    working_directory: str = ""
    
    @classmethod
    def capture_current(cls):
        """Create an Environment instance with current system info"""
        import os
        import sys
        import pkg_resources
        
        # Try to get aider version, default to "0.0.0" if not found
        try:
            aider_version = pkg_resources.get_distribution("aider").version
        except pkg_resources.DistributionNotFound:
            aider_version = "0.0.0"
            
        return cls(
            os=os.name,
            python_version=".".join(map(str, sys.version_info[:3])),
            aider_version=aider_version,
            working_directory=os.getcwd()
        )


@dataclass
class TestInfo:
    failing_tests: List[str] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    attempt_count: int = 0
    attempted_solutions: List[str] = field(default_factory=list)
    name: Optional[str] = None
    status: str = "pending"
    threshold: int = 3


@dataclass
class Task:
    id: str
    name: str
    description: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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
        
    def complete(self):
        """Mark the task as completed"""
        self.status = "completed"
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
    def archive(self):
        """Mark the task as archived"""
        self.status = "archived"
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
    def reactivate(self):
        """Mark the task as active"""
        self.status = "active"
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
    def add_tag(self, tag):
        """Add a tag to the task metadata"""
        if "tags" not in self.metadata:
            self.metadata["tags"] = []
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)

    @property
    def parent_task_id(self):
        """Alias for parent_id to maintain backward compatibility"""
        return self.parent_id
        
    @property
    def tags(self):
        """Get list of tags for this task"""
        return self.metadata.get("tags", [])
        
    @property
    def subtask_ids(self):
        """Get list of subtask IDs for this task"""
        if not hasattr(self, '_subtask_ids'):
            self._subtask_ids = []
        return self._subtask_ids
        
    def to_dict(self):
        """Convert task to dictionary for serialization"""
        # Use asdict from dataclasses to convert to dict
        task_dict = asdict(self)
        return task_dict
        
    @classmethod
    def from_dict(cls, data):
        """Create a Task instance from a dictionary"""
        # Handle nested dataclass objects
        if "environment" in data and data["environment"]:
            data["environment"] = Environment(**data["environment"])
        if "test_info" in data and data["test_info"]:
            data["test_info"] = TestInfo(**data["test_info"])
        
        return cls(**data)


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
        
        # If this is a subtask, add it to the parent's subtask_ids
        if parent_id and parent_id in self.tasks:
            parent_task = self.tasks[parent_id]
            if not hasattr(parent_task, '_subtask_ids'):
                parent_task._subtask_ids = []
            parent_task._subtask_ids.append(task_id)
            self._save_task(parent_task)
        
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
        """Get a task by name (case-insensitive)"""
        name_lower = name.lower()
        for task in self.tasks.values():
            if task.name.lower() == name_lower:
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
        """Reactivate an archived task"""
        task = self.get_task(task_id)
        if task:
            task.reactivate()
            self._save_task(task)
        return task

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by status"""
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        return list(self.tasks.values())
    
    def get_subtasks(self, task_id: str) -> List[Task]:
        """Get all subtasks for a given task
        
        Args:
            task_id: The ID of the parent task
            
        Returns:
            List of subtask Task objects
        """
        parent_task = self.get_task(task_id)
        if not parent_task:
            return []
            
        return [self.tasks[subtask_id] for subtask_id in parent_task.subtask_ids 
                if subtask_id in self.tasks]
    
    def delete_task(self, task_id: str):
        """Delete a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            # Remove task file if it exists
            task_file = Path(self.storage_dir) / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
                
    def add_test_failure(self, task_id: str, test_name: str):
        """Record a test failure for a task
        
        Returns True if the failure count has reached a threshold
        """
        task = self.get_task(task_id)
        if not task:
            return False
            
        # Initialize test_info if not already done
        if not task.test_info:
            task.test_info = TestInfo()
            
        # Add failing test if not already in the list
        if test_name not in task.test_info.failing_tests:
            task.test_info.failing_tests.append(test_name)
            
        # Increment failure count
        count = task.test_info.failure_counts.get(test_name, 0) + 1
        task.test_info.failure_counts[test_name] = count
        
        # Increment attempt count
        task.test_info.attempt_count += 1
        
        # Save the task
        self._save_task(task)
        
        # Return True if failure threshold reached (3 failures)
        return count >= 3
        
    def reset_test_failures(self, task_id: str, test_name: str):
        """Reset the failure count for a specific test in a task
        
        Args:
            task_id: The ID of the task to reset failures for
            test_name: The name of the test to reset
            
        Returns:
            The updated task or None if the task doesn't exist
        """
        task = self.get_task(task_id)
        if not task or not task.test_info:
            return None
            
        # Remove test from failing tests list if present
        if test_name in task.test_info.failing_tests:
            task.test_info.failing_tests.remove(test_name)
            
        # Reset failure count for this test
        if test_name in task.test_info.failure_counts:
            del task.test_info.failure_counts[test_name]
            
        # Save the task
        self._save_task(task)
        
        return task
        
    def add_attempted_solution(self, task_id: str, test_name: str, solution: str, success: bool = False):
        """Record an attempted solution for a test
        
        Args:
            task_id: The ID of the task
            test_name: The name of the test
            solution: Description of the attempted solution
            success: Whether the solution was successful
            
        Returns:
            The updated task or None if the task doesn't exist
        """
        task = self.get_task(task_id)
        if not task:
            return None
            
        # Initialize test_info if not already done
        if not task.test_info:
            task.test_info = TestInfo()
            
        # Add to attempted solutions
        task.test_info.attempted_solutions.append(f"{test_name}: {solution} ({'success' if success else 'failed'})")
        
        # Save the task
        self._save_task(task)
        
        return task


def get_task_manager(storage_dir: Optional[str] = None) -> TaskManager:
    """Get the singleton task manager instance"""
    global _task_manager_instance
    
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager(storage_dir)
        
    return _task_manager_instance
