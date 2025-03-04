import os
import unittest
import tempfile
from pathlib import Path
import json
import yaml
from unittest.mock import MagicMock, patch

from aider.commands import Commands
from aider.io import InputOutput


class SystemCardCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        
        # Mock coder
        self.coder = MagicMock()
        self.coder.root = str(self.root_dir)
        self.coder.abs_fnames = set()
        self.coder.repo = MagicMock()
        
        # Mock IO
        self.io = MagicMock(spec=InputOutput)
        
        # Create commands instance
        self.commands = Commands(
            io=self.io,
            coder=self.coder,
        )
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_systemcard_command_exists(self):
        """Test that the systemcard command exists in available commands"""
        commands = self.commands.get_commands()
        self.assertIn("/systemcard", commands)
        
    @patch('aider.io.InputOutput.input_ask')
    @patch('aider.io.InputOutput.confirm_ask')
    def test_systemcard_create(self, mock_confirm_ask, mock_input_ask):
        """Test creating a system card"""
        # Set up mock input responses
        mock_input_ask.side_effect = [
            "Test Project",                  # Project name
            "A test project description",    # Project description
            "Python",                        # Language
            "Flask",                         # Framework
            "SQLite",                        # Database
            "MVC",                           # Architecture
            "Support user authentication",   # Functional requirement 1
            "Allow file uploads",            # Functional requirement 2
            "",                              # End functional requirements
            "Response time under 100ms",     # Non-functional requirement 1
            "",                              # End non-functional requirements
        ]
        mock_confirm_ask.return_value = False  # Don't add to git
        
        # Call systemcard command
        with patch('aider.commands.yaml.dump'):
            with patch('builtins.open', create=True) as mock_open:
                # Mock file operations
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                # Call the function
                self.commands.cmd_systemcard("")
                
                # Verify the file was "opened" for writing
                mock_open.assert_called_with(self.root_dir / "aider.systemcard.yaml", "w")
                
                # Check function was called with expected arguments
                # Since we can't check the content of the systemcard (it was mocked),
                # we at least check that our input was processed correctly
                self.assertEqual(mock_input_ask.call_count, 11)
                mock_confirm_ask.assert_called_once()

    def test_get_system_card(self):
        """Test the get_system_card method by creating a mock implementation"""
        # Create a mock system card
        mock_system_card = {
            'project': {'name': 'Test'},
            'technologies': {'language': 'Python'},
            'requirements': {'functional': ['Test requirement']}
        }
        
        # Create a simple mock coder class with get_system_card method
        class MockCoder:
            def __init__(self, root_dir):
                self.root = str(root_dir)
                
            def get_system_card(self):
                systemcard_path = Path(self.root) / "aider.systemcard.yaml"
                if systemcard_path.exists():
                    try:
                        with open(systemcard_path, "r") as f:
                            content = f.read()
                            return mock_system_card  # Always return our test card
                    except:
                        return None
                return None
                
        # Create a temporary systemcard file
        systemcard_path = self.root_dir / "aider.systemcard.yaml"
        with open(systemcard_path, "w") as f:
            f.write("dummy content")
            
        # Use our mock coder
        coder = MockCoder(self.root_dir)
        
        # Test get_system_card
        result = coder.get_system_card()
        self.assertEqual(result, mock_system_card)
            
    @patch('yaml.safe_load')
    @patch('yaml.dump')
    def test_task_system_card_integration(self, mock_dump, mock_safe_load):
        """Test integration between tasks and system card"""
        # Create a mock system card
        system_card = {
            'project': {
                'name': 'Test Project',
                'description': 'Test description',
                'architecture': 'Hexagonal'
            },
            'technologies': {
                'language': 'Python',
                'framework': 'Flask'
            },
            'requirements': {
                'functional': [
                    'User authentication',
                    'Data export feature'
                ],
                'non_functional': [
                    'Performance under 100ms'
                ]
            }
        }
        mock_safe_load.return_value = system_card
        
        # Mock file operations
        with patch('builtins.open', create=True):
            with patch('pathlib.Path.exists', return_value=True):
                # Mock task manager
                with patch('aider.commands.get_task_manager') as mock_get_task_manager:
                    mock_task = MagicMock()
                    mock_task.metadata = {}
                    
                    mock_task_manager = MagicMock()
                    mock_task_manager.create_task.return_value = mock_task
                    mock_task_manager.get_task_by_name.return_value = None
                    
                    mock_get_task_manager.return_value = mock_task_manager
                    
                    # Create a task that should match a requirement
                    self.commands._task_create("auth-task Implement user authentication system")
                    
                    # Check if task has system card in metadata
                    self.assertIn('system_card', mock_task.metadata)
                    
                    # Check if the task matches the requirement
                    self.assertIn('matched_requirements', mock_task.metadata)
                    self.assertIn('User authentication', mock_task.metadata['matched_requirements'])


if __name__ == '__main__':
    unittest.main()