import os
import pytest
import tempfile
import yaml
from unittest.mock import MagicMock, patch
from pathlib import Path

from aider.commands import Commands
from aider.taskmanager import Task, TaskManager, TestInfo, Environment


class TestTDDWorkflow:
    """Test suite for Aider's Test-Driven Development workflow functionality."""

    def setup_method(self):
        """Set up test environment before each test method."""
        # Create mock objects
        self.io = MagicMock()
        self.coder = MagicMock()
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        
        # Initialize task manager with temporary storage
        self.task_manager = TaskManager(storage_dir=str(self.test_path))
        
        # Initialize commands with mocks
        self.commands = Commands(self.io, self.coder)
        
        # Patch the _get_task_manager method to return our test task manager
        patch_task_manager = patch.object(
            self.commands, '_get_task_manager', return_value=self.task_manager
        )
        patch_task_manager.start()
        self.addCleanup(patch_task_manager.stop)
        
        # Create a sample system card
        self.sample_system_card = {
            "name": "Test Project",
            "description": "A test project for TDD workflow",
            "architecture": {
                "components": ["api", "database", "frontend"],
                "data_flow": "Frontend calls API which accesses database"
            },
            "test_plan": {
                "frameworks": "pytest",
                "dependencies": "pytest-mock, pytest-cov",
                "environments": "development, ci",
                "strategy": "unit tests, integration tests",
                "automation": "GitHub Actions",
                "known_issues": "Database tests require local PostgreSQL instance"
            }
        }
        
        # Save system card to temporary directory
        with open(os.path.join(self.test_path, "aider.systemcard.yaml"), "w") as f:
            yaml.dump(self.sample_system_card, f)
    
    def addCleanup(self, func):
        """Add cleanup function to be called during teardown."""
        self._cleanup_functions = getattr(self, '_cleanup_functions', [])
        self._cleanup_functions.append(func)
    
    def teardown_method(self):
        """Clean up resources after each test method."""
        cleanup_functions = getattr(self, '_cleanup_functions', [])
        for func in cleanup_functions:
            func()
        self.test_dir.cleanup()
    
    def create_basic_task(self):
        """Create a basic task for testing."""
        return self.task_manager.create_task(
            name="Test Task",
            description="Implement a user authentication system",
        )
    
    def create_brownfield_project_structure(self):
        """Create a typical brownfield project structure with existing code and tests."""
        # Create src directory with some Python files
        src_dir = self.test_path / "src"
        src_dir.mkdir(exist_ok=True)
        
        # Create existing implementation file
        user_py = src_dir / "user.py"
        user_py.write_text("""
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_authenticated = False
    
    def authenticate(self, password):
        # TODO: Implement real authentication
        return False
""")
        
        # Create tests directory with existing tests
        tests_dir = self.test_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Create existing test file
        test_user_py = tests_dir / "test_user.py" 
        test_user_py.write_text("""
import pytest
from src.user import User

def test_user_creation():
    user = User("testuser", "test@example.com")
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_authenticated is False
""")
        
        # Create requirements.txt
        reqs_file = self.test_path / "requirements.txt"
        reqs_file.write_text("pytest==7.3.1\npytest-cov==4.1.0\n")
        
        return src_dir, tests_dir
    
    # GREEN FIELD TESTS
    
    def test_generate_tests_for_greenfield_task(self):
        """Test generation of tests for a new project with no existing code."""
        # Create a task
        task = self.create_basic_task()
        
        # Mock the LLM response for test generation
        self.coder.get_chat_response.return_value = """
# Generated Test Requirements

For the user authentication system, we should test:

1. User registration functionality
2. User login with valid credentials
3. User login with invalid credentials
4. Password reset functionality
5. Session management

Here's the proposed test structure:

```python
# test_auth.py

import pytest
from auth import User, AuthenticationService

def test_user_registration():
    auth_service = AuthenticationService()
    new_user = auth_service.register("testuser", "test@example.com", "password123")
    assert new_user is not None
    assert new_user.username == "testuser"
    assert new_user.email == "test@example.com"
    assert new_user.is_authenticated is False
```
"""
        
        # Call the test generation method
        result = self.commands._generate_tests_for_task(task, self.sample_system_card)
        
        # Assert expectations
        assert result is not None
        assert "Generated Test Requirements" in result
        assert "User registration" in result
        assert "import pytest" in result
        
        # Verify the task was updated
        assert task.metadata.get("test_requirements") is not None
    
    def test_implement_tests_for_greenfield_task(self):
        """Test implementation of tests for a new project with no existing code."""
        # Create a task with test requirements already set
        task = self.create_basic_task()
        task.metadata["test_requirements"] = """
# Test Requirements
1. User registration
2. User login
3. Password validation
"""
        
        # Mock the LLM response for test implementation
        self.coder.get_chat_response.return_value = """
I'll create the tests based on the requirements. Since this is a new project, I'll set up a proper test structure.

```python
# tests/test_auth.py

import pytest
from auth.user import User
from auth.service import AuthenticationService

def test_user_registration():
    """Test that a user can be registered with valid credentials."""
    auth_service = AuthenticationService()
    user = auth_service.register("newuser", "test@example.com", "SecurePass123!")
    
    assert user is not None
    assert user.username == "newuser"
    assert user.email == "test@example.com"
    assert not user.is_authenticated

def test_user_login_valid_credentials():
    """Test that a user can log in with valid credentials."""
    auth_service = AuthenticationService()
    auth_service.register("loginuser", "login@example.com", "SecurePass123!")
    
    result = auth_service.login("loginuser", "SecurePass123!")
    
    assert result.success is True
    assert result.user.is_authenticated is True

def test_user_login_invalid_credentials():
    """Test that login fails with invalid credentials."""
    auth_service = AuthenticationService()
    auth_service.register("failuser", "fail@example.com", "SecurePass123!")
    
    result = auth_service.login("failuser", "WrongPassword")
    
    assert result.success is False
    assert result.error == "Invalid credentials"
```

This test file tests the core authentication functionality. I'm assuming we'll need to implement:

1. A `User` class in `auth/user.py`
2. An `AuthenticationService` class in `auth/service.py`

The tests are designed to fail initially (RED phase) since we haven't implemented the actual functionality yet.
"""
        
        # Call the test implementation method
        with patch('builtins.open', new_callable=MagicMock):
            with patch('pathlib.Path.exists', return_value=False):
                with patch('pathlib.Path.mkdir', return_value=None):
                    result = self.commands._implement_tests_for_task(task)
        
        # Assert expectations
        assert result is not None
        assert "test_auth.py" in result
        assert "test_user_registration" in result
        assert "test_user_login_valid_credentials" in result
        assert "test_user_login_invalid_credentials" in result
        
        # Verify the task was updated
        assert task.test_info is not None
        assert task.test_info.name == "auth_tests"
    
    # BROWNFIELD TESTS
    
    def test_generate_tests_for_brownfield_task(self):
        """Test generation of tests for an existing project with code."""
        # Create brownfield project structure
        src_dir, tests_dir = self.create_brownfield_project_structure()
        
        # Create a task
        task = self.create_basic_task()
        
        # Add existing files to the task
        task.add_files([str(src_dir / "user.py"), str(tests_dir / "test_user.py")])
        
        # Mock the LLM response for test generation
        self.coder.get_chat_response.return_value = """
# Generated Test Requirements for Existing Project

After analyzing the existing code, we need to add tests for:

1. User authentication functionality (currently only stubbed)
2. Password validation
3. User logout

I'll follow the existing test patterns in test_user.py and extend them.

These tests should be added to the existing test file or a new one, following the current project structure.
"""
        
        # Call the test generation method with the existing files
        result = self.commands._generate_tests_for_task(task, self.sample_system_card)
        
        # Assert expectations
        assert result is not None
        assert "Generated Test Requirements for Existing Project" in result
        assert "User authentication functionality" in result
        assert "existing test patterns" in result
        
        # Verify the task was updated
        assert task.metadata.get("test_requirements") is not None
    
    def test_implement_tests_for_brownfield_task(self):
        """Test implementation of tests for an existing project with code."""
        # Create brownfield project structure
        src_dir, tests_dir = self.create_brownfield_project_structure()
        
        # Create a task with test requirements already set
        task = self.create_basic_task()
        task.add_files([str(src_dir / "user.py"), str(tests_dir / "test_user.py")])
        
        task.metadata["test_requirements"] = """
# Test Requirements
1. User authentication with password
2. User logout
3. Password validation
"""
        
        # Mock the LLM response for test implementation
        self.coder.get_chat_response.return_value = """
I'll implement the tests for the authentication functionality, following the existing project structure and test patterns.

Looking at the existing code, I'll add the following tests to the existing test_user.py file:

```python
# Additions to test_user.py

def test_user_authentication_valid_password():
    """Test that a user can authenticate with valid password."""
    user = User("authuser", "auth@example.com")
    
    # Mock the authentication to return True for testing
    with patch.object(User, 'authenticate', return_value=True):
        result = user.authenticate("valid_password")
        
    assert result is True
    assert user.is_authenticated is True

def test_user_authentication_invalid_password():
    """Test that authentication fails with invalid password."""
    user = User("authuser", "auth@example.com")
    
    # The real implementation should verify the password and return False
    result = user.authenticate("invalid_password")
    
    assert result is False
    assert user.is_authenticated is False

def test_user_logout():
    """Test that a user can log out."""
    user = User("logoutuser", "logout@example.com")
    user.is_authenticated = True  # Set as authenticated first
    
    user.logout()
    
    assert user.is_authenticated is False
```

These tests follow the existing pattern but add the new functionality we need to test. They assume we'll add a logout method to the User class.
"""
        
        # Call the test implementation method with existing files
        with patch('builtins.open', new_callable=MagicMock):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="existing test content"):
                    result = self.commands._implement_tests_for_task(task)
        
        # Assert expectations
        assert result is not None
        assert "test_user_authentication_valid_password" in result
        assert "test_user_authentication_invalid_password" in result
        assert "test_user_logout" in result
        
        # Verify the task was updated
        assert task.test_info is not None
    
    # IMPLEMENTATION TESTS
    
    def test_implement_functionality_greenfield(self):
        """Test code implementation for tests in a new project."""
        # Create a task with test info
        task = self.create_basic_task()
        task.test_info = TestInfo(name="auth_tests")
        
        # Mock test code
        test_code = """
import pytest
from auth.user import User
from auth.service import AuthenticationService

def test_user_registration():
    auth_service = AuthenticationService()
    user = auth_service.register("newuser", "test@example.com", "SecurePass123!")
    
    assert user is not None
    assert user.username == "newuser"
    assert user.email == "test@example.com"
    assert not user.is_authenticated
"""
        
        # Mock the LLM response for implementation
        self.coder.get_chat_response.return_value = """
To make the tests pass, I'll need to implement:

1. The `auth/user.py` file with a User class
2. The `auth/service.py` file with an AuthenticationService class

Here's the implementation:

```python
# auth/user.py
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_authenticated = False
        self._password_hash = None
    
    def authenticate(self, password):
        # In a real implementation, check password hash
        if self._password_hash == password:  # Simplified for demo
            self.is_authenticated = True
            return True
        return False
    
    def logout(self):
        self.is_authenticated = False
```

```python
# auth/service.py
from auth.user import User

class AuthenticationResult:
    def __init__(self, success, user=None, error=None):
        self.success = success
        self.user = user
        self.error = error

class AuthenticationService:
    def __init__(self):
        self.users = {}  # username -> User
    
    def register(self, username, email, password):
        if username in self.users:
            return None
        
        user = User(username, email)
        user._password_hash = password  # Simplified, should hash password
        self.users[username] = user
        return user
    
    def login(self, username, password):
        if username not in self.users:
            return AuthenticationResult(False, error="User not found")
        
        user = self.users[username]
        if user.authenticate(password):
            return AuthenticationResult(True, user=user)
        
        return AuthenticationResult(False, error="Invalid credentials")
```

This is a simplified implementation that should make the tests pass. In a real application, you would:
1. Hash passwords properly
2. Use a real database instead of an in-memory dictionary
3. Add proper error handling and validation
"""
        
        # Call the implementation method
        with patch('builtins.open', new_callable=MagicMock):
            with patch('pathlib.Path.exists', return_value=False):
                with patch('pathlib.Path.mkdir', return_value=None):
                    result = self.commands._implement_functionality_for_tests(task, "tests/test_auth.py", test_code)
        
        # Assert expectations
        assert result is not None
        assert "auth/user.py" in result
        assert "auth/service.py" in result
        assert "class User:" in result
        assert "class AuthenticationService:" in result
    
    def test_implement_functionality_brownfield(self):
        """Test code implementation for tests in an existing project."""
        # Create brownfield project structure
        src_dir, tests_dir = self.create_brownfield_project_structure()
        
        # Create a task with test info
        task = self.create_basic_task()
        task.test_info = TestInfo(name="user_tests")
        task.add_files([str(src_dir / "user.py"), str(tests_dir / "test_user.py")])
        
        # Mock the test code that was added to the existing test file
        test_code = """
def test_user_authentication_valid_password():
    user = User("authuser", "auth@example.com")
    
    # Mock the authentication to return True for testing
    with patch.object(User, 'authenticate', return_value=True):
        result = user.authenticate("valid_password")
        
    assert result is True
    assert user.is_authenticated is True

def test_user_logout():
    user = User("logoutuser", "logout@example.com")
    user.is_authenticated = True  # Set as authenticated first
    
    user.logout()
    
    assert user.is_authenticated is False
"""
        
        # Mock the LLM response for implementation updates
        self.coder.get_chat_response.return_value = """
To make the new tests pass, I need to update the existing User class to add:
1. Proper password authentication
2. A logout method

Here's the updated implementation:

```python
# Updated src/user.py
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_authenticated = False
        self._password = None  # In reality, would store hashed password
    
    def set_password(self, password):
        # In real implementation, would hash the password
        self._password = password
    
    def authenticate(self, password):
        # Check if password matches stored password
        # In reality, would compare hashed passwords
        if self._password == password:
            self.is_authenticated = True
            return True
        return False
    
    def logout(self):
        self.is_authenticated = False
```

This update adds the missing functionality while preserving the existing behavior. The authenticate method now actually checks the password (in a simplified way) and sets the is_authenticated flag when successful. I've also added the logout method as required by the tests.
"""
        
        # Call the implementation method with the existing file
        with patch('builtins.open', new_callable=MagicMock):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="class User:\n    def __init__(self, username, email):\n        self.username = username\n        self.email = email\n        self.is_authenticated = False\n    \n    def authenticate(self, password):\n        # TODO: Implement real authentication\n        return False"):
                    result = self.commands._implement_functionality_for_tests(task, str(tests_dir / "test_user.py"), test_code)
        
        # Assert expectations
        assert result is not None
        assert "src/user.py" in result
        assert "def authenticate(self, password):" in result
        assert "def logout(self):" in result
        assert "self.is_authenticated = True" in result
    
    # TEST EXECUTION TESTS
    
    def test_run_tests_passing(self):
        """Test running tests that pass."""
        # Create a task with test info
        task = self.create_basic_task()
        task.test_info = TestInfo(name="passing_tests")
        
        # Mock the subprocess run to simulate passing tests
        with patch('subprocess.run') as mock_run:
            # Configure the mock to return a completed process with success
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "============================= test session starts ==============================\ncollected 3 items\n\ntest_auth.py::test_user_registration PASSED\ntest_auth.py::test_user_login_valid_credentials PASSED\ntest_auth.py::test_user_login_invalid_credentials PASSED\n\n============================== 3 passed in 0.05s ==============================="
            
            # Call the run tests method
            result = self.commands._run_tests_for_task(task, "tests/test_auth.py")
        
        # Assert expectations
        assert result is not None
        assert "All tests passed" in result
        assert task.test_info.status == "passing"
    
    def test_run_tests_failing(self):
        """Test running tests that fail."""
        # Create a task with test info
        task = self.create_basic_task()
        task.test_info = TestInfo(name="failing_tests")
        
        # Mock the subprocess run to simulate failing tests
        with patch('subprocess.run') as mock_run:
            # Configure the mock to return a completed process with failure
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = "============================= test session starts ==============================\ncollected 3 items\n\ntest_auth.py::test_user_registration PASSED\ntest_auth.py::test_user_login_valid_credentials FAILED\ntest_auth.py::test_user_login_invalid_credentials PASSED\n\n================================= FAILURES =================================\n___________________ test_user_login_valid_credentials ____________________\n\n    def test_user_login_valid_credentials():\n        auth_service = AuthenticationService()\n        auth_service.register(\"loginuser\", \"login@example.com\", \"SecurePass123!\")\n        \n        result = auth_service.login(\"loginuser\", \"SecurePass123!\")\n        \n>       assert result.success is True\nE       AttributeError: 'NoneType' object has no attribute 'success'\n\ntest_auth.py:25: AttributeError\n========================= 1 failed, 2 passed in 0.08s =========================="
            
            # Call the run tests method
            result = self.commands._run_tests_for_task(task, "tests/test_auth.py")
        
        # Assert expectations
        assert result is not None
        assert "Tests are failing" in result
        assert task.test_info.status == "failing"
        assert "test_user_login_valid_credentials" in task.test_info.failing_tests
        assert task.test_info.failure_counts["test_user_login_valid_credentials"] == 1
    
    # REFACTORING TESTS
    
    def test_refactor_passing_code(self):
        """Test refactoring code that already has passing tests."""
        # Create a task with test info (passing status)
        task = self.create_basic_task()
        task.test_info = TestInfo(name="passing_tests", status="passing")
        
        # Mock code before refactoring
        code_before = """
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_authenticated = False
        self._password = None
    
    def authenticate(self, password):
        if self._password == password:
            self.is_authenticated = True
            return True
        return False
"""
        
        # Mock the LLM response for refactoring
        self.coder.get_chat_response.return_value = """
I can improve this code by:
1. Adding proper password hashing
2. Adding validation
3. Improving the API design

Here's the refactored code:

```python
import hashlib
import re

class User:
    def __init__(self, username, email):
        self._validate_username(username)
        self._validate_email(email)
        
        self.username = username
        self.email = email
        self.is_authenticated = False
        self._password_hash = None
        self._salt = None
    
    def set_password(self, password):
        """Set a hashed password for the user."""
        self._validate_password(password)
        self._salt = self._generate_salt()
        self._password_hash = self._hash_password(password, self._salt)
    
    def authenticate(self, password):
        """Authenticate a user with the given password."""
        if not self._password_hash or not self._salt:
            return False
            
        hashed = self._hash_password(password, self._salt)
        if hashed == self._password_hash:
            self.is_authenticated = True
            return True
        return False
    
    def _validate_username(self, username):
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
    
    def _validate_email(self, email):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format")
    
    def _validate_password(self, password):
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
    
    def _generate_salt(self):
        """Generate a random salt for password hashing."""
        import os
        return os.urandom(16).hex()
    
    def _hash_password(self, password, salt):
        """Hash a password with the given salt."""
        return hashlib.sha256((password + salt).encode()).hexdigest()
```

This refactored code improves security by properly hashing passwords with a salt, adds validation for inputs, and uses a cleaner API design with helper methods. It maintains the same interface so tests will still pass.
"""
        
        # Mock running tests after refactoring
        with patch('subprocess.run') as mock_run:
            # Tests still pass after refactoring
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "============================= test session starts ==============================\ncollected 3 items\n\ntest_user.py::test_user_creation PASSED\ntest_user.py::test_user_authentication PASSED\ntest_user.py::test_user_logout PASSED\n\n============================== 3 passed in 0.05s ==============================="
            
            # Mock file operations
            with patch('builtins.open', new_callable=MagicMock):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_text', return_value=code_before):
                        result = self.commands._refactor_code_for_task(task, "tests/test_user.py")
        
        # Assert expectations
        assert result is not None
        assert "Refactored implementation" in result
        assert "import hashlib" in result
        assert "_hash_password" in result
        assert "All tests still pass" in result
    
    # SYSTEM INTEGRATION TESTS
    
    def test_autotest_command_integration(self):
        """Test the complete autotest command workflow."""
        # Create a task
        task = self.create_basic_task()
        self.task_manager.switch_task(task.id)
        
        # Mock all the component methods
        with patch.object(self.commands, '_generate_tests_for_task', return_value="Test requirements generated") as mock_generate:
            with patch.object(self.commands, '_implement_tests_for_task', return_value="Tests implemented") as mock_implement_tests:
                with patch.object(self.commands, '_implement_functionality_for_tests', return_value="Functionality implemented") as mock_implement_func:
                    with patch.object(self.commands, '_run_tests_for_task', return_value="All tests pass") as mock_run_tests:
                        with patch.object(self.commands, '_refactor_code_for_task', return_value="Code refactored") as mock_refactor:
                            # Mock user inputs to progress through workflow
                            self.io.input.side_effect = ["y", "y", "y", "y"]
                            
                            # Call the autotest command
                            result = self.commands.cmd_autotest([])
        
        # Assert expectations
        assert result is None  # Command completes without returning
        assert mock_generate.called
        assert mock_implement_tests.called
        assert mock_implement_func.called
        assert mock_run_tests.called
        assert mock_refactor.called
        
        # Verify the correct messages were printed
        self.io.tool_output.assert_any_call("Starting Test-Driven Development workflow for task: Test Task")


class TestTDDEnvironmentSpecificTests:
    """Test suite for environment-specific testing in the TDD workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.io = MagicMock()
        self.coder = MagicMock()
        self.commands = Commands(self.io, self.coder)
        
        # Create different environment configurations
        self.dev_env = Environment(os="posix", python_version="3.9.0", aider_version="0.8.0")
        self.prod_env = Environment(os="nt", python_version="3.8.5", aider_version="0.8.0")
    
    def test_environment_specific_test_generation(self):
        """Test generation of environment-specific tests."""
        # Create task with specific environment
        task = Task(id="env-test", name="Environment Test", description="Test cross-platform compatibility")
        task.environment = self.dev_env
        
        # System card with environment-specific test plan
        system_card = {
            "test_plan": {
                "frameworks": "pytest",
                "environments": {
                    "posix": {"dependencies": "pytest-mock"},
                    "nt": {"dependencies": "pytest-mock, pywin32"}
                },
                "known_issues": {
                    "posix": "File paths need to use os.path.join",
                    "nt": "Case sensitivity in filenames"
                }
            }
        }
        
        # Mock the LLM response
        self.coder.get_chat_response.return_value = """
# Environment-Specific Test Requirements

I've analyzed the task and the system card, noting the current environment is POSIX (Unix/Linux/Mac).

## Test Requirements:
1. File path handling across platforms
2. File operations with proper encoding
3. Process handling differences

For the POSIX environment specifically, tests should check:
- Path separators are properly used with os.path.join
- File permissions are properly set and checked
- Symbolic links are handled correctly

```python
# Example test for POSIX environment
def test_file_path_handling_posix():
    # This test is specific to POSIX systems
    import os
    assert os.sep == '/'
    
    # Test path joining works correctly
    path = os.path.join('dir', 'subdir', 'file.txt')
    assert path == 'dir/subdir/file.txt'
```

The tests will need to be adapted for Windows environments to account for different path separators and file permission models.
"""
        
        # Call the test generation method
        with patch.object(self.commands, '_get_task_manager', return_value=MagicMock()):
            result = self.commands._generate_tests_for_task(task, system_card)
        
        # Assert environment-specific content
        assert "POSIX (Unix/Linux/Mac)" in result
        assert "Path separators" in result
        assert "os.sep == '/'" in result
    
    def test_windows_specific_test_generation(self):
        """Test generation of Windows-specific tests."""
        # Create task with Windows environment
        task = Task(id="win-test", name="Windows Test", description="Test Windows-specific features")
        task.environment = self.prod_env  # Windows environment
        
        # System card with environment-specific test plan
        system_card = {
            "test_plan": {
                "frameworks": "pytest",
                "environments": {
                    "posix": {"dependencies": "pytest-mock"},
                    "nt": {"dependencies": "pytest-mock, pywin32"}
                },
                "known_issues": {
                    "posix": "File paths need to use os.path.join",
                    "nt": "Case sensitivity in filenames"
                }
            }
        }
        
        # Mock the LLM response for Windows environment
        self.coder.get_chat_response.return_value = """
# Environment-Specific Test Requirements

I've analyzed the task and the system card, noting the current environment is Windows (NT).

## Test Requirements:
1. Windows-specific file path handling
2. Registry access testing
3. Windows service integration

For the Windows environment specifically, tests should check:
- Path separators using backslashes
- Case insensitivity in file operations
- Windows-specific API calls using pywin32

```python
# Example test for Windows environment
def test_file_path_handling_windows():
    # This test is specific to Windows systems
    import os
    assert os.sep == '\\'
    
    # Test path joining works correctly
    path = os.path.join('dir', 'subdir', 'file.txt')
    assert path == 'dir\\subdir\\file.txt'
    
    # Test case insensitivity
    import pathlib
    p1 = pathlib.Path('TestFile.txt')
    p2 = pathlib.Path('testfile.txt')
    assert p1.exists() == p2.exists()
```

Note: These tests require the pywin32 package installed as specified in the system card.
"""
        
        # Call the test generation method
        with patch.object(self.commands, '_get_task_manager', return_value=MagicMock()):
            result = self.commands._generate_tests_for_task(task, system_card)
        
        # Assert environment-specific content
        assert "Windows (NT)" in result
        assert "Case insensitivity" in result
        assert "os.sep == '\\\\'" in result
        assert "pywin32" in result 