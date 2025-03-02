import re

from ..taskmanager import get_task_manager
from .architect_prompts import ArchitectPrompts
from .ask_coder import AskCoder
from .base_coder import Coder


class ArchitectCoder(AskCoder):
    edit_format = "architect"
    gpt_prompts = ArchitectPrompts()

    def reply_completed(self):
        content = self.partial_response_content

        if not content or not content.strip():
            return

        # Check for auto-task generation if enabled
        if self.architect_auto_tasks:
            self._handle_task_management(content)

        # Record this discussion in the active task if one exists
        if self.active_task and self.task_manager:
            # Add conversation context to task
            self.active_task.add_conversation_context(content[:500] + "..." if len(content) > 500 else content)
            # Update task in storage
            self.task_manager.update_task(self.active_task)

        if not self.io.confirm_ask("Edit the files?"):
            return

        kwargs = dict()

        # Use the editor_model from the main_model if it exists, otherwise use the main_model itself
        editor_model = self.main_model.editor_model or self.main_model

        kwargs["main_model"] = editor_model
        kwargs["edit_format"] = self.main_model.editor_edit_format
        kwargs["suggest_shell_commands"] = False
        kwargs["map_tokens"] = 0
        kwargs["total_cost"] = self.total_cost
        kwargs["cache_prompts"] = False
        kwargs["num_cache_warming_pings"] = 0
        kwargs["summarize_from_coder"] = False
        
        # Pass task information to the editor coder
        kwargs["task_manager"] = self.task_manager
        kwargs["active_task"] = self.active_task
        kwargs["architect_auto_tasks"] = self.architect_auto_tasks
        kwargs["auto_test_tasks"] = self.auto_test_tasks
        kwargs["auto_test_retry_limit"] = self.auto_test_retry_limit

        new_kwargs = dict(io=self.io, from_coder=self)
        new_kwargs.update(kwargs)

        editor_coder = Coder.create(**new_kwargs)
        editor_coder.cur_messages = []
        editor_coder.done_messages = []

        if self.verbose:
            editor_coder.show_announcements()

        editor_coder.run(with_message=content, preproc=False)

        self.move_back_cur_messages("I made those changes to the files.")
        self.total_cost = editor_coder.total_cost
        self.aider_commit_hashes = editor_coder.aider_commit_hashes
        
        # Update the active task with any new files that were created
        if self.active_task and self.task_manager:
            # Get current files in task
            current_files = set(self.active_task.files)
            # Get all files in editor
            new_files = set([self.get_rel_fname(fname) for fname in editor_coder.abs_fnames])
            # Add any new files
            for file in new_files:
                if file not in current_files:
                    self.active_task.add_files([file])
            # Update task
            self.task_manager.update_task(self.active_task)
        
    def _handle_task_management(self, content):
        """
        Analyze the architect's response to identify potential tasks
        and suggest task creation if appropriate.
        """
        # Get or create task manager
        task_manager = get_task_manager()
        
        # Check if we're already in a task
        active_task = task_manager.get_active_task()
        
        # Patterns to identify task-like content
        task_patterns = [
            r"(?:need|should) to (implement|create|build|fix|add|update) ([^\.\n]+)",
            r"(?:I'll|Let's|We should) (implement|create|build|fix|add|update) ([^\.\n]+)",
            r"(?:Task|TODO|Step) (?:\d+)?: ([^\.\n]+)",
            r"(?:Bug|Issue|Error): ([^\.\n]+)",
        ]
        
        potential_tasks = []
        
        for pattern in task_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    action, description = match.groups()
                    task_name = f"{action} {description}"
                else:
                    task_name = match.group(1)
                potential_tasks.append(task_name.strip())
        
        # If we found potential tasks, suggest creating them
        if potential_tasks and not active_task:
            # Take the first identified task as the main one
            main_task = potential_tasks[0]
            
            # Suggest creating the task
            if self.io.confirm_ask(f"Would you like to create a task for '{main_task}'?"):
                task = task_manager.create_task(main_task, main_task)
                
                # Associate current files with the task
                for fname in self.abs_fnames:
                    rel_fname = self.get_rel_fname(fname)
                    task.add_files([rel_fname])
                
                # Switch to the new task
                task_manager.switch_task(task.id)
                
                self.io.tool_output(f"Created and switched to task: {main_task}")
                
                # If there are additional tasks, suggest creating them as subtasks
                if len(potential_tasks) > 1:
                    if self.io.confirm_ask(f"Would you like to create {len(potential_tasks)-1} subtasks?"):
                        for subtask_name in potential_tasks[1:]:
                            subtask = task_manager.create_task(subtask_name, subtask_name, parent_id=task.id)
                            self.io.tool_output(f"Created subtask: {subtask_name}")
        elif potential_tasks and active_task:
            # If already in a task, suggest creating subtasks
            if self.io.confirm_ask(f"Would you like to add {len(potential_tasks)} subtasks to '{active_task.name}'?"):
                for subtask_name in potential_tasks:
                    subtask = task_manager.create_task(subtask_name, subtask_name, parent_id=active_task.id)
                    self.io.tool_output(f"Created subtask: {subtask_name}")
