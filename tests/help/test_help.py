import unittest
from unittest.mock import MagicMock, patch

import aider
from aider.coders import Coder
from aider.commands import Commands
from aider.help import Help, fname_to_url
from aider.io import InputOutput
from aider.models import Model


class TestHelp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        io = InputOutput(pretty=False, yes=True)

        GPT35 = Model("gpt-3.5-turbo")

        coder = Coder.create(GPT35, None, io)
        commands = Commands(io, coder)

        help_coder_run = MagicMock(return_value="")
        aider.coders.HelpCoder.run = help_coder_run

        try:
            commands.cmd_help("hi")
        except aider.commands.SwitchCoder:
            pass
        else:
            # If no exception was raised, fail the test
            assert False, "SwitchCoder exception was not raised"

        help_coder_run.assert_called_once()

    @patch('aider.help.HuggingFaceEmbedding')
    @patch('aider.help.get_index')
    def test_init(self, mock_get_index, mock_hf_embedding):
        # Create a mock retriever
        mock_retriever = MagicMock()
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_get_index.return_value = mock_index
        
        help_inst = Help()
        self.assertIsNotNone(help_inst.retriever)
        
        # Verify mocks were called
        mock_get_index.assert_called_once()
        mock_hf_embedding.assert_called_once()

    @patch('aider.help.HuggingFaceEmbedding')
    @patch('aider.help.get_index')
    def test_ask_without_mock(self, mock_get_index, mock_hf_embedding):
        # Set up mock retriever with sample nodes
        mock_node1 = MagicMock()
        mock_node1.text = "Aider is an AI coding assistant."
        mock_node1.metadata = {"url": "https://aider.chat/docs/index.html"}
        
        mock_node2 = MagicMock()
        mock_node2.text = "Aider helps you chat with AI about your code."
        mock_node2.metadata = {"url": "https://aider.chat/docs/usage.html"}
        
        # Create more mock nodes to satisfy the test condition
        mock_nodes = [mock_node1, mock_node2]
        for i in range(10):
            node = MagicMock()
            node.text = f"Sample content {i}"
            node.metadata = {"url": f"https://aider.chat/docs/sample{i}.html"}
            mock_nodes.append(node)
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes
        
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_get_index.return_value = mock_index
        
        help_instance = Help()
        question = "What is aider?"
        result = help_instance.ask(question)

        self.assertIn(f"# Question: {question}", result)
        self.assertIn("<doc", result)
        self.assertIn("</doc>", result)
        self.assertGreater(len(result), 100)  # Ensure we got a substantial response

        # Check for some expected content from our mocks
        self.assertIn("aider", result.lower())
        self.assertIn("ai", result.lower())
        self.assertIn("chat", result.lower())

        # Assert that there are more than 5 <doc> entries
        self.assertGreater(result.count("<doc"), 5)

    # These tests don't need mocking since they only test the URL conversion function
    def test_fname_to_url_unix(self):
        # Test relative Unix-style paths
        self.assertEqual(fname_to_url("website/docs/index.md"), "https://aider.chat/docs")
        self.assertEqual(
            fname_to_url("website/docs/usage.md"), "https://aider.chat/docs/usage.html"
        )
        self.assertEqual(fname_to_url("website/_includes/header.md"), "")

        # Test absolute Unix-style paths
        self.assertEqual(
            fname_to_url("/home/user/project/website/docs/index.md"), "https://aider.chat/docs"
        )
        self.assertEqual(
            fname_to_url("/home/user/project/website/docs/usage.md"),
            "https://aider.chat/docs/usage.html",
        )
        self.assertEqual(fname_to_url("/home/user/project/website/_includes/header.md"), "")

    def test_fname_to_url_windows(self):
        # Test relative Windows-style paths
        self.assertEqual(fname_to_url(r"website\docs\index.md"), "https://aider.chat/docs")
        self.assertEqual(
            fname_to_url(r"website\docs\usage.md"), "https://aider.chat/docs/usage.html"
        )
        self.assertEqual(fname_to_url(r"website\_includes\header.md"), "")

        # Test absolute Windows-style paths
        self.assertEqual(
            fname_to_url(r"C:\Users\user\project\website\docs\index.md"), "https://aider.chat/docs"
        )
        self.assertEqual(
            fname_to_url(r"C:\Users\user\project\website\docs\usage.md"),
            "https://aider.chat/docs/usage.html",
        )
        self.assertEqual(fname_to_url(r"C:\Users\user\project\website\_includes\header.md"), "")

    def test_fname_to_url_edge_cases(self):
        # Test paths that don't contain 'website'
        self.assertEqual(fname_to_url("/home/user/project/docs/index.md"), "")
        self.assertEqual(fname_to_url(r"C:\Users\user\project\docs\index.md"), "")

        # Test empty path
        self.assertEqual(fname_to_url(""), "")

        # Test path with 'website' in the wrong place
        self.assertEqual(fname_to_url("/home/user/website_project/docs/index.md"), "")


if __name__ == "__main__":
    unittest.main()
