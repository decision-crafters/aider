# Task Management with Memory

Aider includes a powerful task management system that helps you organize your work, maintain context across sessions, and track progress on multiple parallel tasks. This system is particularly useful for long-term projects, collaborative workflows, and debugging sessions.

## Getting Started with Tasks

### Creating Tasks

Tasks can be created using the `/task create` command:

```bash
/task create fix-login-bug "Fix the login functionality bug that prevents admin login"
```

The first parameter is the task name (used for reference), and the second parameter is a more detailed description.

When using architect mode with the `--architect-auto-tasks` flag, Aider can also suggest task creation automatically when it identifies task-like content in the architect's response.

### Listing Tasks

To see all your tasks:

```bash
/task list
```

You can filter tasks by status:

```bash
/task list active    # Show only active tasks
/task list completed # Show only completed tasks
/task list archived  # Show only archived tasks
```

### Switching Between Tasks

To switch to a different task:

```bash
/task switch fix-login-bug
```

When switching tasks, Aider will:
1. Save the current state (files and conversation)
2. Clear the current context
3. Restore the files and context associated with the selected task

### Completing and Archiving Tasks

When you finish a task, you can mark it as completed:

```bash
/task complete fix-login-bug
```

For tasks you want to set aside but keep for reference:

```bash
/task archive fix-login-bug
```

You can reactivate completed or archived tasks at any time:

```bash
/task reactivate fix-login-bug
```

### Viewing Task Details

To see details about a specific task:

```bash
/task info fix-login-bug
```

This shows:
- Task metadata (name, description, status, creation date)
- Associated files
- Parent/subtask relationships
- Environment information
- Test failure history (if applicable)

## Advanced Features

### Task Hierarchy

Tasks can be organized in a hierarchy with parent tasks and subtasks:

```bash
/task create refactor-auth "Refactor authentication system"
/task create update-login "Update login form" refactor-auth  # Creates a subtask
```

### Testing Integration

When working on tasks involving tests, Aider will:
1. Track test failures for each task
2. Monitor failure thresholds and offer help when tests fail repeatedly
3. Remember attempted solutions and their outcomes
4. Reset failure tracking when tests pass

### Environment Awareness

Tasks automatically capture information about the environment:
- Operating system
- Programming language versions
- Git branch and repository state
- Working directory and paths

This helps ensure consistent behavior when resuming tasks, even if the environment has changed.

## Best Practices

1. **Create focused tasks**: Each task should represent a distinct unit of work
2. **Use descriptive names**: Make task names clear and specific
3. **Leverage task hierarchy**: Use parent-child relationships for complex work
4. **Switch tasks when context-switching**: Use the task system rather than trying to juggle multiple tasks in a single context
5. **Complete or archive finished tasks**: Keep your active task list clean

## Architect Integration

When using Aider's architect mode (with the `--architect` flag), you can enable automatic task suggestion by adding the `--architect-auto-tasks` flag:

```bash
aider --architect --architect-auto-tasks myfile.py
```

With this flag enabled, the architect will analyze its responses for potential tasks and suggest creating them when appropriate. This works particularly well for breaking down complex implementations into manageable chunks.