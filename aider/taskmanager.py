"""
Task management system with memory for Aider.

This module provides a task management system that allows users to create, switch between,
and complete tasks while maintaining context and memory across sessions.
"""

import os
import json
import platform
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys


@dataclass
class Environment:
    """Information about the environment in which a task is executed."""

    os: str = ""
    python_version: str = ""
    git_branch: str = ""
    git_repo: str = ""
    working_directory: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture_current(cls):
        """Capture the current environment information."""
        env = cls()
        env.os = platform.system()
        env.python_version = sys.version
        # Git information will be populated by the GitRepo class
        env.working_directory = os.getcwd()
        # Dependencies would need to be determined based on project type
        return env


@dataclass
class TestInfo:
    """Information specific to testing tasks."""

    failing_tests: List[str] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    threshold: int = 3  # Default threshold for failures before assistance
    attempted_solutions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Task:
    """Represents a single task in the task management system."""

    id: str
    name: str
    description: str
    status: str = "active"  # active, completed, archived
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: Environment = field(default_factory=Environment)
    files: List[str] = field(default_factory=list)
    conversation_context: str = ""  # Reference to conversation history
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    test_info: Optional[TestInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self):
        """Update the task's updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def add_files(self, files: List[str]):
        """Add files to the task's list of associated files."""
        for file in files:
            if file not in self.files:
                self.files.append(file)
        self.update()

    def add_conversation_context(self, context: str):
        """Update the task's conversation context."""
        self.conversation_context = context
        self.update()

    def complete(self):
        """Mark the task as completed."""
        self.status = "completed"
        self.update()

    def archive(self):
        """Archive the task."""
        self.status = "archived"
        self.update()

    def reactivate(self):
        """Reactivate an archived or completed task."""
        self.status = "active"
        self.update()

    def add_subtask(self, subtask_id: str):
        """Add a subtask to this task."""
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)
        self.update()

    def remove_subtask(self, subtask_id: str):
        """Remove a subtask from this task."""
        if subtask_id in self.subtask_ids:
            self.subtask_ids.remove(subtask_id)
        self.update()

    def add_dependency(self, task_id: str):
        """Add a dependency to this task."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
        self.update()

    def remove_dependency(self, task_id: str):
        """Remove a dependency from this task."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
        self.update()

    def add_tag(self, tag: str):
        """Add a tag to this task."""
        if tag not in self.tags:
            self.tags.append(tag)
        self.update()

    def remove_tag(self, tag: str):
        """Remove a tag from this task."""
        if tag in self.tags:
            self.tags.remove(tag)
        self.update()

    def to_dict(self):
        """Convert the task to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a task from a dictionary."""
        if 'environment' in data and isinstance(data['environment'], dict):
            data['environment'] = Environment(**data['environment'])

        if 'test_info' in data and isinstance(data['test_info'], dict):
            data['test_info'] = TestInfo(**data['test_info'])

        return cls(**data)


class TaskManager:
    """Manages tasks, providing persistence and operations."""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the task manager with a storage directory."""
        if storage_dir is None:
            home = Path.home()
            self.storage_dir = home / '.aider' / 'tasks'
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = {}
        self.active_task_id = None
        self._load_tasks()

    def _load_tasks(self):
        """Load all tasks from storage."""
        for file_path in self.storage_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    task = Task.from_dict(task_data)
                    self.tasks[task.id] = task
            except Exception as e:
                print(f"Error loading task {file_path}: {e}")

    def _save_task(self, task: Task):
        """Save a task to storage."""
        # Ensure the storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        task_path = self.storage_dir / f"{task.id}.json"
        with open(task_path, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)

    def create_task(self, name: str, description: str, parent_id: Optional[str] = None) -> Task:
        """Create a new task and save it."""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            description=description,
            environment=Environment.capture_current(),
            parent_task_id=parent_id
        )

        # If this is a subtask, update the parent
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].add_subtask(task_id)
            self._save_task(self.tasks[parent_id])

        self.tasks[task_id] = task
        self._save_task(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_task_by_name(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        for task in self.tasks.values():
            if task.name.lower() == name.lower():
                return task
        return None

    def update_task(self, task: Task):
        """Update an existing task."""
        if task.id in self.tasks:
            self.tasks[task.id] = task
            self._save_task(task)

    def delete_task(self, task_id: str):
        """Delete a task."""
        if task_id in self.tasks:
            task_path = self.storage_dir / f"{task_id}.json"
            if task_path.exists():
                task_path.unlink()
            del self.tasks[task_id]

    def list_tasks(self, status: Optional[str] = None, tag: Optional[str] = None) -> List[Task]:
        """List tasks, optionally filtered by status or tag."""
        tasks = list(self.tasks.values())

        if status:
            tasks = [task for task in tasks if task.status == status]

        if tag:
            tasks = [task for task in tasks if tag in task.tags]

        # Sort by updated_at, most recent first
        tasks.sort(key=lambda t: t.updated_at, reverse=True)
        return tasks

    def switch_task(self, task_id: str):
        """Switch to a different task."""
        if task_id in self.tasks:
            self.active_task_id = task_id
            return self.tasks[task_id]
        return None

    def get_active_task(self) -> Optional[Task]:
        """Get the currently active task."""
        if self.active_task_id and self.active_task_id in self.tasks:
            return self.tasks[self.active_task_id]
        return None

    def complete_task(self, task_id: str):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].complete()
            self._save_task(self.tasks[task_id])

    def archive_task(self, task_id: str):
        """Archive a task."""
        if task_id in self.tasks:
            self.tasks[task_id].archive()
            self._save_task(self.tasks[task_id])

    def reactivate_task(self, task_id: str):
        """Reactivate an archived or completed task."""
        if task_id in self.tasks:
            self.tasks[task_id].reactivate()
            self._save_task(self.tasks[task_id])

    def get_subtasks(self, task_id: str) -> List[Task]:
        """Get all subtasks for a task."""
        if task_id not in self.tasks:
            return []

        task = self.tasks[task_id]
        return [self.tasks[subtask_id] for subtask_id in task.subtask_ids
                if subtask_id in self.tasks]

    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        """Get all tasks that depend on this task."""
        dependent_tasks = []
        for task in self.tasks.values():
            if task_id in task.dependencies:
                dependent_tasks.append(task)
        return dependent_tasks

    def add_test_failure(self, task_id: str, test_name: str):
        """Add a test failure to a task and increment its count."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.test_info is None:
                task.test_info = TestInfo()

            if test_name not in task.test_info.failing_tests:
                task.test_info.failing_tests.append(test_name)

            if test_name in task.test_info.failure_counts:
                task.test_info.failure_counts[test_name] += 1
            else:
                task.test_info.failure_counts[test_name] = 1

            self._save_task(task)
            return task.test_info.failure_counts[test_name] >= task.test_info.threshold
        return False

    def reset_test_failures(self, task_id: str, test_name: Optional[str] = None):
        """Reset test failure counts for a specific test or all tests."""
        if task_id in self.tasks and self.tasks[task_id].test_info:
            task = self.tasks[task_id]
            if test_name:
                if test_name in task.test_info.failing_tests:
                    task.test_info.failing_tests.remove(test_name)
                if test_name in task.test_info.failure_counts:
                    del task.test_info.failure_counts[test_name]
            else:
                task.test_info.failing_tests = []
                task.test_info.failure_counts = {}

            self._save_task(task)

    def add_attempted_solution(self, task_id: str, test_name: str, solution: str, successful: bool):
        """Add an attempted solution for a test failure."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.test_info is None:
                task.test_info = TestInfo()

            solution_entry = {
                "test_name": test_name,
                "solution": solution,
                "successful": successful,
                "timestamp": datetime.now().isoformat()
            }

            task.test_info.attempted_solutions.append(solution_entry)
            self._save_task(task)

    def get_attempted_solutions(
        self, task_id: str, test_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get attempted solutions for a test or all tests."""
        if task_id in self.tasks and self.tasks[task_id].test_info:
            task = self.tasks[task_id]
            if test_name:
                return [
                    sol for sol in task.test_info.attempted_solutions
                    if sol["test_name"] == test_name
                ]
            return task.test_info.attempted_solutions
        return []


# Task manager singleton instance
_task_manager_instance = None


def get_task_manager(storage_dir: Optional[str] = None) -> TaskManager:
    """Get or create the task manager singleton instance."""
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager(storage_dir)
    return _task_manager_instance
