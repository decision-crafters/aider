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
        
    def test_systemcard_create(self):
        """Test creating a system card"""
        # Set up mock input responses
        self.io.input_ask.side_effect = [
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
        self.io.confirm_ask.return_value = False  # Don't add to git
        
        # Call systemcard command
        self.commands.cmd_systemcard("")
        
        # Check if systemcard was created
        systemcard_path = self.root_dir / "aider.systemcard.yaml"
        self.assertTrue(systemcard_path.exists())
        
        # Verify content
        with open(systemcard_path, "r") as f:
            systemcard = yaml.safe_load(f)
            
        self.assertEqual(systemcard['project']['name'], "Test Project")
        self.assertEqual(systemcard['project']['description'], "A test project description")
        self.assertEqual(systemcard['project']['architecture'], "MVC")
        self.assertEqual(systemcard['technologies']['language'], "Python")
        self.assertEqual(systemcard['technologies']['framework'], "Flask")
        self.assertEqual(systemcard['technologies']['database'], "SQLite")
        self.assertEqual(len(systemcard['requirements']['functional']), 2)
        self.assertEqual(len(systemcard['requirements']['non_functional']), 1)
        self.assertIn("Support user authentication", systemcard['requirements']['functional'])
        self.assertIn("Response time under 100ms", systemcard['requirements']['non_functional'])

    @patch('aider.coders.base_coder.yaml')
    def test_get_system_card(self, mock_yaml):
        """Test the get_system_card method"""
        # Create a mock system card
        mock_system_card = {
            'project': {'name': 'Test'},
            'technologies': {'language': 'Python'},
            'requirements': {'functional': ['Test requirement']}
        }
        mock_yaml.safe_load.return_value = mock_system_card
        
        # Create a temporary systemcard file
        systemcard_path = self.root_dir / "aider.systemcard.yaml"
        with open(systemcard_path, "w") as f:
            f.write("dummy content")
            
        # Mock the base_coder class
        with patch('aider.coders.base_coder.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            from aider.coders.base_coder import BaseCoder
            
            # Initialize the coder with our mock
            coder = BaseCoder(io=self.io)
            coder.root = str(self.root_dir)
            
            # Test get_system_card
            result = coder.get_system_card()
            self.assertEqual(result, mock_system_card)
            
    def test_task_system_card_integration(self):
        """Test integration between tasks and system card"""
        # Create a system card
        systemcard_path = self.root_dir / "aider.systemcard.yaml"
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
        
        with open(systemcard_path, "w") as f:
            yaml.dump(system_card, f)
            
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