from .ask_prompts import AskPrompts
from .base_coder import Coder


class AskCoder(Coder):
    """Ask questions about code without making any changes."""

    edit_format = "ask"
    gpt_prompts = AskPrompts()
    
    def reply_completed(self):
        content = self.partial_response_content
        
        if not content or not content.strip():
            return
        
        # Record this discussion in the active task if one exists
        if self.active_task and self.task_manager:
            # Add conversation context to task
            self.active_task.add_conversation_context(content[:500] + "..." if len(content) > 500 else content)
            # Update task in storage
            self.task_manager.update_task(self.active_task)
