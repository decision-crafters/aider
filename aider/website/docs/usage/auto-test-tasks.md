# Automated Test Resolution with Tasks

Aider includes a powerful feature that automatically resolves failing tests by creating and managing tasks. This is particularly useful for test-driven development, debugging, and fixing regressions.

## Getting Started with Auto-Test-Tasks

### Basic Usage

To enable automated test resolution, use the `--auto-test-tasks` flag when running Aider:

```bash
aider --auto-test-tasks --test-cmd "pytest" file_to_fix.py
```

This will:
1. Run the specified test command
2. For each failing test, create a task to track the fix
3. Use the AI to analyze and fix the failing tests
4. Re-run tests after each fix attempt
5. Continue until tests pass or the retry limit is reached

### Running Only Tests

You can run only the test resolution process without entering chat mode by combining with the `--test` flag:

```bash
aider --auto-test-tasks --test-cmd "pytest" --test file_to_fix.py
```

This will run tests, fix any failures, and exit when complete.

## Configuration Options

### Retry Limit

By default, Aider will attempt to fix each failing test up to 5 times before giving up. You can adjust this limit:

```bash
aider --auto-test-tasks --auto-test-retry-limit 10 --test-cmd "pytest" file_to_fix.py
```

## How It Works

When a test fails with `--auto-test-tasks` enabled, Aider:

1. **Creates a Task**: A dedicated task is created for each failing test
2. **Analyzes the Failure**: The test output is analyzed to understand the issue
3. **Implements Fixes**: Code changes are made to resolve the failure
4. **Tracks Progress**: All attempts are recorded in the task
5. **Learns from Failures**: Each failed attempt informs the next fix approach

If the test fails multiple times, Aider increases the sophistication of its analysis:
- First attempt: Basic fix attempt
- Second/third attempts: More careful analysis of requirements and edge cases
- Later attempts: Comprehensive investigation, possibly modifying the test itself

## Task Integration

Test failures are fully integrated with the task system:

- Each test gets its own dedicated task
- The task stores all attempted solutions
- You can view detailed information about fix attempts with `/task info <taskname>`
- The history is preserved across sessions for long-running debugging efforts

## Best Practices

1. **Start with Good Tests**: The more specific and clear your tests are, the better Aider can fix them
2. **Include Test Context**: Make sure test error messages are informative
3. **Fix One Thing at a Time**: For complex projects, focus on one module or feature area
4. **Review Changes**: While automated fixes are convenient, always review the changes made

## Example Workflow

Here's a typical workflow for using auto-test-tasks:

1. Write tests for a new feature
2. Run `aider --auto-test-tasks --test-cmd "pytest" --test`
3. Aider automatically creates tasks and fixes failing tests
4. Review the changes and commit them
5. For any remaining issues, enter interactive mode to refine the solutions

This approach lets you focus on defining requirements through tests while Aider handles the implementation details.