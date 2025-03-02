import tempfile
import unittest
from pathlib import Path

from aider.taskmanager import Task, Environment, TestInfo, TaskManager, get_task_manager


class TaskTest(unittest.TestCase):
    def test_task_creation(self):
        task = Task(id="123", name="Test Task", description="A test task")
        self.assertEqual(task.id, "123")
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.description, "A test task")
        self.assertEqual(task.status, "active")

    def test_task_status_changes(self):
        task = Task(id="123", name="Test Task", description="A test task")
        self.assertEqual(task.status, "active")

        task.complete()
        self.assertEqual(task.status, "completed")

        task.archive()
        self.assertEqual(task.status, "archived")

        task.reactivate()
        self.assertEqual(task.status, "active")

    def test_file_management(self):
        task = Task(id="123", name="Test Task", description="A test task")
        self.assertEqual(len(task.files), 0)

        task.add_files(["file1.py", "file2.py"])
        self.assertEqual(len(task.files), 2)
        self.assertIn("file1.py", task.files)
        self.assertIn("file2.py", task.files)

        # Test idempotence
        task.add_files(["file1.py"])
        self.assertEqual(len(task.files), 2)

    def test_to_from_dict(self):
        task = Task(id="123", name="Test Task", description="A test task")
        task.add_files(["file1.py"])
        task.add_tag("bug")

        task_dict = task.to_dict()
        self.assertIsInstance(task_dict, dict)

        reconstituted_task = Task.from_dict(task_dict)
        self.assertEqual(reconstituted_task.id, task.id)
        self.assertEqual(reconstituted_task.name, task.name)
        self.assertEqual(reconstituted_task.files, task.files)
        self.assertEqual(reconstituted_task.tags, task.tags)

    def test_environment_capture(self):
        env = Environment.capture_current()
        self.assertIsInstance(env.os, str)
        self.assertIsInstance(env.python_version, str)
        self.assertIsInstance(env.working_directory, str)

    def test_test_info(self):
        test_info = TestInfo()
        self.assertEqual(len(test_info.failing_tests), 0)
        self.assertEqual(len(test_info.failure_counts), 0)
        self.assertEqual(len(test_info.attempted_solutions), 0)
        self.assertEqual(test_info.threshold, 3)  # Default


class TaskManagerTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for task storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.task_manager = TaskManager(self.temp_dir.name)

    def tearDown(self):
        # Clean up
        self.temp_dir.cleanup()

    def test_create_task(self):
        task = self.task_manager.create_task("Test Task", "A test task")
        self.assertIsInstance(task, Task)
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.description, "A test task")

        # Verify task was saved
        task_file = Path(self.temp_dir.name) / f"{task.id}.json"
        self.assertTrue(task_file.exists())

    def test_get_task(self):
        task = self.task_manager.create_task("Test Task", "A test task")
        retrieved_task = self.task_manager.get_task(task.id)
        self.assertEqual(retrieved_task.id, task.id)
        self.assertEqual(retrieved_task.name, task.name)

    def test_get_task_by_name(self):
        task = self.task_manager.create_task("Unique Task Name", "A test task")
        retrieved_task = self.task_manager.get_task_by_name("Unique Task Name")
        self.assertEqual(retrieved_task.id, task.id)

        # Test case insensitivity
        retrieved_task = self.task_manager.get_task_by_name("unique task name")
        self.assertEqual(retrieved_task.id, task.id)

    def test_list_tasks(self):
        self.task_manager.create_task("Task 1", "Description 1")
        self.task_manager.create_task("Task 2", "Description 2")

        tasks = self.task_manager.list_tasks()
        self.assertEqual(len(tasks), 2)

    def test_switch_task(self):
        task1 = self.task_manager.create_task("Task 1", "Description 1")
        task2 = self.task_manager.create_task("Task 2", "Description 2")

        self.task_manager.switch_task(task2.id)
        active_task = self.task_manager.get_active_task()
        self.assertEqual(active_task.id, task2.id)

        self.task_manager.switch_task(task1.id)
        active_task = self.task_manager.get_active_task()
        self.assertEqual(active_task.id, task1.id)

    def test_delete_task(self):
        task = self.task_manager.create_task("Task to delete", "Will be deleted")
        task_id = task.id

        self.task_manager.delete_task(task_id)
        self.assertIsNone(self.task_manager.get_task(task_id))

        # Verify file was deleted
        task_file = Path(self.temp_dir.name) / f"{task_id}.json"
        self.assertFalse(task_file.exists())

    def test_subtasks(self):
        parent_task = self.task_manager.create_task("Parent Task", "Parent task")
        subtask = self.task_manager.create_task("Subtask", "A subtask", parent_id=parent_task.id)

        # Verify parent-child relationship
        self.assertEqual(subtask.parent_task_id, parent_task.id)
        self.assertIn(subtask.id, parent_task.subtask_ids)

        # Test get_subtasks method
        subtasks = self.task_manager.get_subtasks(parent_task.id)
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].id, subtask.id)

    def test_test_tracking(self):
        task = self.task_manager.create_task("Test tracking task", "For testing test tracking")

        # Add first failure
        threshold_reached = self.task_manager.add_test_failure(task.id, "test_something")
        self.assertFalse(threshold_reached)  # 1 < threshold

        # Add more failures to reach threshold
        self.task_manager.add_test_failure(task.id, "test_something")
        threshold_reached = self.task_manager.add_test_failure(task.id, "test_something")
        self.assertTrue(threshold_reached)  # 3 >= threshold

        # Reset failure counts
        self.task_manager.reset_test_failures(task.id, "test_something")
        updated_task = self.task_manager.get_task(task.id)
        self.assertNotIn("test_something", updated_task.test_info.failure_counts)

    def test_singleton(self):
        # Test that get_task_manager returns a singleton
        task_manager1 = get_task_manager(self.temp_dir.name)
        task_manager2 = get_task_manager()  # Should return the same instance

        self.assertIs(task_manager1, task_manager2)

        # Create a task using the first instance
        task = task_manager1.create_task("Singleton test", "Testing singleton pattern")

        # Verify it's accessible from the second instance
        retrieved_task = task_manager2.get_task_by_name("Singleton test")
        self.assertEqual(retrieved_task.id, task.id)


if __name__ == '__main__':
    unittest.main()
