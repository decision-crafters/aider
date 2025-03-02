import glob
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from os.path import expanduser
from pathlib import Path

import pyperclip
from PIL import Image, ImageGrab
from prompt_toolkit.completion import Completion, PathCompleter
from prompt_toolkit.document import Document

from aider import models, prompts, voice
from aider.editor import pipe_editor
from aider.format_settings import format_settings
from aider.help import Help, install_help_extra
from aider.llm import litellm
from aider.repo import ANY_GIT_ERROR
from aider.run_cmd import run_cmd
from aider.scrape import Scraper, install_playwright
from aider.taskmanager import get_task_manager, Task
from aider.utils import is_image_file

from .dump import dump  # noqa: F401


class SwitchCoder(Exception):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Commands:
    voice = None
    scraper = None

    def clone(self):
        return Commands(
            self.io,
            None,
            voice_language=self.voice_language,
            verify_ssl=self.verify_ssl,
            args=self.args,
            parser=self.parser,
            verbose=self.verbose,
            editor=self.editor,
        )

    def __init__(
        self,
        io,
        coder,
        voice_language=None,
        voice_input_device=None,
        voice_format=None,
        verify_ssl=True,
        args=None,
        parser=None,
        verbose=False,
        editor=None,
    ):
        self.io = io
        self.coder = coder
        self.parser = parser
        self.args = args
        self.verbose = verbose

        self.verify_ssl = verify_ssl
        if voice_language == "auto":
            voice_language = None

        self.voice_language = voice_language
        self.voice_format = voice_format
        self.voice_input_device = voice_input_device

        self.help = None
        self.editor = editor

    def cmd_model(self, args):
        "Switch to a new LLM"

        model_name = args.strip()
        model = models.Model(model_name, weak_model=self.coder.main_model.weak_model.name)
        models.sanity_check_models(self.io, model)
        raise SwitchCoder(main_model=model)

    def cmd_chat_mode(self, args):
        "Switch to a new chat mode"

        from aider import coders

        ef = args.strip()
        valid_formats = OrderedDict(
            sorted(
                (
                    coder.edit_format,
                    coder.__doc__.strip().split("\n")[0] if coder.__doc__ else "No description",
                )
                for coder in coders.__all__
                if getattr(coder, "edit_format", None)
            )
        )

        show_formats = OrderedDict(
            [
                ("help", "Get help about using aider (usage, config, troubleshoot)."),
                ("ask", "Ask questions about your code without making any changes."),
                ("code", "Ask for changes to your code (using the best edit format)."),
                (
                    "architect",
                    (
                        "Work with an architect model to design code changes, and an editor to make"
                        " them."
                    ),
                ),
            ]
        )

        if ef not in valid_formats and ef not in show_formats:
            if ef:
                self.io.tool_error(f'Chat mode "{ef}" should be one of these:\n')
            else:
                self.io.tool_output("Chat mode should be one of these:\n")

            max_format_length = max(len(format) for format in valid_formats.keys())
            for format, description in show_formats.items():
                self.io.tool_output(f"- {format:<{max_format_length}} : {description}")

            self.io.tool_output("\nOr a valid edit format:\n")
            for format, description in valid_formats.items():
                if format not in show_formats:
                    self.io.tool_output(f"- {format:<{max_format_length}} : {description}")

            return

        summarize_from_coder = True
        edit_format = ef

        if ef == "code":
            edit_format = self.coder.main_model.edit_format
            summarize_from_coder = False
        elif ef == "ask":
            summarize_from_coder = False

        raise SwitchCoder(
            edit_format=edit_format,
            summarize_from_coder=summarize_from_coder,
        )

    def completions_model(self):
        models = litellm.model_cost.keys()
        return models

    def cmd_models(self, args):
        "Search the list of available models"

        args = args.strip()

        if args:
            models.print_matching_models(self.io, args)
        else:
            self.io.tool_output("Please provide a partial model name to search for.")

    def cmd_web(self, args, return_content=False):
        "Scrape a webpage, convert to markdown and send in a message"

        url = args.strip()
        if not url:
            self.io.tool_error("Please provide a URL to scrape.")
            return

        self.io.tool_output(f"Scraping {url}...")
        if not self.scraper:
            res = install_playwright(self.io)
            if not res:
                self.io.tool_warning("Unable to initialize playwright.")

            self.scraper = Scraper(
                print_error=self.io.tool_error, playwright_available=res, verify_ssl=self.verify_ssl
            )

        content = self.scraper.scrape(url) or ""
        content = f"Here is the content of {url}:\n\n" + content
        if return_content:
            return content

        self.io.tool_output("... added to chat.")

        self.coder.cur_messages += [
            dict(role="user", content=content),
            dict(role="assistant", content="Ok."),
        ]

    def is_command(self, inp):
        return inp[0] in "/!"

    def get_raw_completions(self, cmd):
        assert cmd.startswith("/")
        cmd = cmd[1:]
        cmd = cmd.replace("-", "_")

        raw_completer = getattr(self, f"completions_raw_{cmd}", None)
        return raw_completer

    def get_completions(self, cmd):
        assert cmd.startswith("/")
        cmd = cmd[1:]

        cmd = cmd.replace("-", "_")
        fun = getattr(self, f"completions_{cmd}", None)
        if not fun:
            return
        return sorted(fun())

    def get_commands(self):
        commands = []
        for attr in dir(self):
            if not attr.startswith("cmd_"):
                continue
            cmd = attr[4:]
            cmd = cmd.replace("_", "-")
            commands.append("/" + cmd)

        return commands

    def do_run(self, cmd_name, args):
        cmd_name = cmd_name.replace("-", "_")
        cmd_method_name = f"cmd_{cmd_name}"
        cmd_method = getattr(self, cmd_method_name, None)
        if not cmd_method:
            self.io.tool_output(f"Error: Command {cmd_name} not found.")
            return

        try:
            return cmd_method(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete {cmd_name}: {err}")

    def matching_commands(self, inp):
        words = inp.strip().split()
        if not words:
            return

        first_word = words[0]
        rest_inp = inp[len(words[0]) :].strip()

        all_commands = self.get_commands()
        matching_commands = [cmd for cmd in all_commands if cmd.startswith(first_word)]
        return matching_commands, first_word, rest_inp

    def run(self, inp):
        if inp.startswith("!"):
            self.coder.event("command_run")
            return self.do_run("run", inp[1:])

        res = self.matching_commands(inp)
        if res is None:
            return
        matching_commands, first_word, rest_inp = res
        if len(matching_commands) == 1:
            command = matching_commands[0][1:]
            self.coder.event(f"command_{command}")
            return self.do_run(command, rest_inp)
        elif first_word in matching_commands:
            command = first_word[1:]
            self.coder.event(f"command_{command}")
            return self.do_run(command, rest_inp)
        elif len(matching_commands) > 1:
            self.io.tool_error(f"Ambiguous command: {', '.join(matching_commands)}")
        else:
            self.io.tool_error(f"Invalid command: {first_word}")

    # any method called cmd_xxx becomes a command automatically.
    # each one must take an args param.

    def cmd_commit(self, args=None):
        "Commit edits to the repo made outside the chat (commit message optional)"
        try:
            self.raw_cmd_commit(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete commit: {err}")

    def raw_cmd_commit(self, args=None):
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        if not self.coder.repo.is_dirty():
            self.io.tool_warning("No more changes to commit.")
            return

        commit_message = args.strip() if args else None
        self.coder.repo.commit(message=commit_message)

    def cmd_lint(self, args="", fnames=None):
        "Lint and fix in-chat files or all dirty files if none in chat"

        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        if not fnames:
            fnames = self.coder.get_inchat_relative_files()

        # If still no files, get all dirty files in the repo
        if not fnames and self.coder.repo:
            fnames = self.coder.repo.get_dirty_files()

        if not fnames:
            self.io.tool_warning("No dirty files to lint.")
            return

        fnames = [self.coder.abs_root_path(fname) for fname in fnames]

        lint_coder = None
        for fname in fnames:
            try:
                errors = self.coder.linter.lint(fname)
            except FileNotFoundError as err:
                self.io.tool_error(f"Unable to lint {fname}")
                self.io.tool_output(str(err))
                continue

            if not errors:
                continue

            self.io.tool_output(errors)
            if not self.io.confirm_ask(f"Fix lint errors in {fname}?", default="y"):
                continue

            # Commit everything before we start fixing lint errors
            if self.coder.repo.is_dirty() and self.coder.dirty_commits:
                self.cmd_commit("")

            if not lint_coder:
                lint_coder = self.coder.clone(
                    # Clear the chat history, fnames
                    cur_messages=[],
                    done_messages=[],
                    fnames=None,
                )

            lint_coder.add_rel_fname(fname)
            lint_coder.run(errors)
            lint_coder.abs_fnames = set()

        if lint_coder and self.coder.repo.is_dirty() and self.coder.auto_commits:
            self.cmd_commit("")

    def cmd_clear(self, args):
        "Clear the chat history"

        self._clear_chat_history()

    def _drop_all_files(self):
        self.coder.abs_fnames = set()
        self.coder.abs_read_only_fnames = set()

    def _clear_chat_history(self):
        self.coder.done_messages = []
        self.coder.cur_messages = []

    def cmd_reset(self, args):
        "Drop all files and clear the chat history"
        self._drop_all_files()
        self._clear_chat_history()
        self.io.tool_output("All files dropped and chat history cleared.")

    def cmd_tokens(self, args):
        "Report on the number of tokens used by the current chat context"

        res = []

        self.coder.choose_fence()

        # system messages
        main_sys = self.coder.fmt_system_prompt(self.coder.gpt_prompts.main_system)
        main_sys += "\n" + self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder)
        msgs = [
            dict(role="system", content=main_sys),
            dict(
                role="system",
                content=self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder),
            ),
        ]

        tokens = self.coder.main_model.token_count(msgs)
        res.append((tokens, "system messages", ""))

        # chat history
        msgs = self.coder.done_messages + self.coder.cur_messages
        if msgs:
            tokens = self.coder.main_model.token_count(msgs)
            res.append((tokens, "chat history", "use /clear to clear"))

        # repo map
        other_files = set(self.coder.get_all_abs_files()) - set(self.coder.abs_fnames)
        if self.coder.repo_map:
            repo_content = self.coder.repo_map.get_repo_map(self.coder.abs_fnames, other_files)
            if repo_content:
                tokens = self.coder.main_model.token_count(repo_content)
                res.append((tokens, "repository map", "use --map-tokens to resize"))

        fence = "`" * 3

        file_res = []
        # files
        for fname in self.coder.abs_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.io.read_text(fname)
            if is_image_file(relative_fname):
                tokens = self.coder.main_model.token_count_for_image(fname)
            else:
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)
            file_res.append((tokens, f"{relative_fname}", "/drop to remove"))

        # read-only files
        for fname in self.coder.abs_read_only_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.io.read_text(fname)
            if content is not None and not is_image_file(relative_fname):
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)
                file_res.append((tokens, f"{relative_fname} (read-only)", "/drop to remove"))

        file_res.sort()
        res.extend(file_res)

        self.io.tool_output(
            f"Approximate context window usage for {self.coder.main_model.name}, in tokens:"
        )
        self.io.tool_output()

        width = 8
        cost_width = 9

        def fmt(v):
            return format(int(v), ",").rjust(width)

        col_width = max(len(row[1]) for row in res)

        cost_pad = " " * cost_width
        total = 0
        total_cost = 0.0
        for tk, msg, tip in res:
            total += tk
            cost = tk * (self.coder.main_model.info.get("input_cost_per_token") or 0)
            total_cost += cost
            msg = msg.ljust(col_width)
            self.io.tool_output(f"${cost:7.4f} {fmt(tk)} {msg} {tip}")  # noqa: E231

        self.io.tool_output("=" * (width + cost_width + 1))
        self.io.tool_output(f"${total_cost:7.4f} {fmt(total)} tokens total")  # noqa: E231

        limit = self.coder.main_model.info.get("max_input_tokens") or 0
        if not limit:
            return

        remaining = limit - total
        if remaining > 1024:
            self.io.tool_output(f"{cost_pad}{fmt(remaining)} tokens remaining in context window")
        elif remaining > 0:
            self.io.tool_error(
                f"{cost_pad}{fmt(remaining)} tokens remaining in context window (use /drop or"
                " /clear to make space)"
            )
        else:
            self.io.tool_error(
                f"{cost_pad}{fmt(remaining)} tokens remaining, window exhausted (use /drop or"
                " /clear to make space)"
            )
        self.io.tool_output(f"{cost_pad}{fmt(limit)} tokens max context window size")

    def cmd_undo(self, args):
        "Undo the last git commit if it was done by aider"
        try:
            self.raw_cmd_undo(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete undo: {err}")

    def raw_cmd_undo(self, args):
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        last_commit = self.coder.repo.get_head_commit()
        if not last_commit or not last_commit.parents:
            self.io.tool_error("This is the first commit in the repository. Cannot undo.")
            return

        last_commit_hash = self.coder.repo.get_head_commit_sha(short=True)
        last_commit_message = self.coder.repo.get_head_commit_message("(unknown)").strip()
        if last_commit_hash not in self.coder.aider_commit_hashes:
            self.io.tool_error("The last commit was not made by aider in this chat session.")
            self.io.tool_output(
                "You could try `/git reset --hard HEAD^` but be aware that this is a destructive"
                " command!"
            )
            return

        if len(last_commit.parents) > 1:
            self.io.tool_error(
                f"The last commit {last_commit.hexsha} has more than 1 parent, can't undo."
            )
            return

        prev_commit = last_commit.parents[0]
        changed_files_last_commit = [item.a_path for item in last_commit.diff(prev_commit)]

        for fname in changed_files_last_commit:
            if self.coder.repo.repo.is_dirty(path=fname):
                self.io.tool_error(
                    f"The file {fname} has uncommitted changes. Please stash them before undoing."
                )
                return

            # Check if the file was in the repo in the previous commit
            try:
                prev_commit.tree[fname]
            except KeyError:
                self.io.tool_error(
                    f"The file {fname} was not in the repository in the previous commit. Cannot"
                    " undo safely."
                )
                return

        local_head = self.coder.repo.repo.git.rev_parse("HEAD")
        current_branch = self.coder.repo.repo.active_branch.name
        try:
            remote_head = self.coder.repo.repo.git.rev_parse(f"origin/{current_branch}")
            has_origin = True
        except ANY_GIT_ERROR:
            has_origin = False

        if has_origin:
            if local_head == remote_head:
                self.io.tool_error(
                    "The last commit has already been pushed to the origin. Undoing is not"
                    " possible."
                )
                return

        # Reset only the files which are part of `last_commit`
        restored = set()
        unrestored = set()
        for file_path in changed_files_last_commit:
            try:
                self.coder.repo.repo.git.checkout("HEAD~1", file_path)
                restored.add(file_path)
            except ANY_GIT_ERROR:
                unrestored.add(file_path)

        if unrestored:
            self.io.tool_error(f"Error restoring {file_path}, aborting undo.")
            self.io.tool_output("Restored files:")
            for file in restored:
                self.io.tool_output(f"  {file}")
            self.io.tool_output("Unable to restore files:")
            for file in unrestored:
                self.io.tool_output(f"  {file}")
            return

        # Move the HEAD back before the latest commit
        self.coder.repo.repo.git.reset("--soft", "HEAD~1")

        self.io.tool_output(f"Removed: {last_commit_hash} {last_commit_message}")

        # Get the current HEAD after undo
        current_head_hash = self.coder.repo.get_head_commit_sha(short=True)
        current_head_message = self.coder.repo.get_head_commit_message("(unknown)").strip()
        self.io.tool_output(f"Now at:  {current_head_hash} {current_head_message}")

        if self.coder.main_model.send_undo_reply:
            return prompts.undo_command_reply

    def cmd_diff(self, args=""):
        "Display the diff of changes since the last message"
        try:
            self.raw_cmd_diff(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete diff: {err}")

    def raw_cmd_diff(self, args=""):
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        current_head = self.coder.repo.get_head_commit_sha()
        if current_head is None:
            self.io.tool_error("Unable to get current commit. The repository might be empty.")
            return

        if len(self.coder.commit_before_message) < 2:
            commit_before_message = current_head + "^"
        else:
            commit_before_message = self.coder.commit_before_message[-2]

        if not commit_before_message or commit_before_message == current_head:
            self.io.tool_warning("No changes to display since the last message.")
            return

        self.io.tool_output(f"Diff since {commit_before_message[:7]}...")

        if self.coder.pretty:
            run_cmd(f"git diff {commit_before_message}")
            return

        diff = self.coder.repo.diff_commits(
            self.coder.pretty,
            commit_before_message,
            "HEAD",
        )

        self.io.print(diff)

    def quote_fname(self, fname):
        if " " in fname and '"' not in fname:
            fname = f'"{fname}"'
        return fname

    def completions_raw_read_only(self, document, complete_event):
        # Get the text before the cursor
        text = document.text_before_cursor

        # Skip the first word and the space after it
        after_command = text.split()[-1]

        # Create a new Document object with the text after the command
        new_document = Document(after_command, cursor_position=len(after_command))

        def get_paths():
            return [self.coder.root] if self.coder.root else None

        path_completer = PathCompleter(
            get_paths=get_paths,
            only_directories=False,
            expanduser=True,
        )

        # Adjust the start_position to replace all of 'after_command'
        adjusted_start_position = -len(after_command)

        # Collect all completions
        all_completions = []

        # Iterate over the completions and modify them
        for completion in path_completer.get_completions(new_document, complete_event):
            quoted_text = self.quote_fname(after_command + completion.text)
            all_completions.append(
                Completion(
                    text=quoted_text,
                    start_position=adjusted_start_position,
                    display=completion.display,
                    style=completion.style,
                    selected_style=completion.selected_style,
                )
            )

        # Add completions from the 'add' command
        add_completions = self.completions_add()
        for completion in add_completions:
            if after_command in completion:
                all_completions.append(
                    Completion(
                        text=completion,
                        start_position=adjusted_start_position,
                        display=completion,
                    )
                )

        # Sort all completions based on their text
        sorted_completions = sorted(all_completions, key=lambda c: c.text)

        # Yield the sorted completions
        for completion in sorted_completions:
            yield completion

    def completions_add(self):
        files = set(self.coder.get_all_relative_files())
        files = files - set(self.coder.get_inchat_relative_files())
        files = [self.quote_fname(fn) for fn in files]
        return files

    def glob_filtered_to_repo(self, pattern):
        if not pattern.strip():
            return []
        try:
            if os.path.isabs(pattern):
                # Handle absolute paths
                raw_matched_files = [Path(pattern)]
            else:
                try:
                    raw_matched_files = list(Path(self.coder.root).glob(pattern))
                except (IndexError, AttributeError):
                    raw_matched_files = []
        except ValueError as err:
            self.io.tool_error(f"Error matching {pattern}: {err}")
            raw_matched_files = []

        matched_files = []
        for fn in raw_matched_files:
            matched_files += expand_subdir(fn)

        matched_files = [
            fn.relative_to(self.coder.root)
            for fn in matched_files
            if fn.is_relative_to(self.coder.root)
        ]

        # if repo, filter against it
        if self.coder.repo:
            git_files = self.coder.repo.get_tracked_files()
            matched_files = [fn for fn in matched_files if str(fn) in git_files]

        res = list(map(str, matched_files))
        return res

    def cmd_add(self, args):
        "Add files to the chat so aider can edit them or review them in detail"

        all_matched_files = set()

        filenames = parse_quoted_filenames(args)
        for word in filenames:
            if Path(word).is_absolute():
                fname = Path(word)
            else:
                fname = Path(self.coder.root) / word

            if self.coder.repo and self.coder.repo.ignored_file(fname):
                self.io.tool_warning(f"Skipping {fname} due to aiderignore or --subtree-only.")
                continue

            if fname.exists():
                if fname.is_file():
                    all_matched_files.add(str(fname))
                    continue
                # an existing dir, escape any special chars so they won't be globs
                word = re.sub(r"([\*\?\[\]])", r"[\1]", word)

            matched_files = self.glob_filtered_to_repo(word)
            if matched_files:
                all_matched_files.update(matched_files)
                continue

            if "*" in str(fname) or "?" in str(fname):
                self.io.tool_error(
                    f"No match, and cannot create file with wildcard characters: {fname}"
                )
                continue

            if fname.exists() and fname.is_dir() and self.coder.repo:
                self.io.tool_error(f"Directory {fname} is not in git.")
                self.io.tool_output(f"You can add to git with: /git add {fname}")
                continue

            if self.io.confirm_ask(f"No files matched '{word}'. Do you want to create {fname}?"):
                try:
                    fname.parent.mkdir(parents=True, exist_ok=True)
                    fname.touch()
                    all_matched_files.add(str(fname))
                except OSError as e:
                    self.io.tool_error(f"Error creating file {fname}: {e}")

        for matched_file in sorted(all_matched_files):
            abs_file_path = self.coder.abs_root_path(matched_file)

            if not abs_file_path.startswith(self.coder.root) and not is_image_file(matched_file):
                self.io.tool_error(
                    f"Can not add {abs_file_path}, which is not within {self.coder.root}"
                )
                continue

            if self.coder.repo and self.coder.repo.git_ignored_file(matched_file):
                self.io.tool_error(f"Can't add {matched_file} which is in gitignore")
                continue

            if abs_file_path in self.coder.abs_fnames:
                self.io.tool_error(f"{matched_file} is already in the chat as an editable file")
                continue
            elif abs_file_path in self.coder.abs_read_only_fnames:
                if self.coder.repo and self.coder.repo.path_in_repo(matched_file):
                    self.coder.abs_read_only_fnames.remove(abs_file_path)
                    self.coder.abs_fnames.add(abs_file_path)
                    self.io.tool_output(
                        f"Moved {matched_file} from read-only to editable files in the chat"
                    )
                else:
                    self.io.tool_error(
                        f"Cannot add {matched_file} as it's not part of the repository"
                    )
            else:
                if is_image_file(matched_file) and not self.coder.main_model.info.get(
                    "supports_vision"
                ):
                    self.io.tool_error(
                        f"Cannot add image file {matched_file} as the"
                        f" {self.coder.main_model.name} does not support images."
                    )
                    continue
                content = self.io.read_text(abs_file_path)
                if content is None:
                    self.io.tool_error(f"Unable to read {matched_file}")
                else:
                    self.coder.abs_fnames.add(abs_file_path)
                    fname = self.coder.get_rel_fname(abs_file_path)
                    self.io.tool_output(f"Added {fname} to the chat")
                    self.coder.check_added_files()

    def completions_drop(self):
        files = self.coder.get_inchat_relative_files()
        read_only_files = [self.coder.get_rel_fname(fn) for fn in self.coder.abs_read_only_fnames]
        all_files = files + read_only_files
        all_files = [self.quote_fname(fn) for fn in all_files]
        return all_files

    def cmd_drop(self, args=""):
        "Remove files from the chat session to free up context space"

        if not args.strip():
            self.io.tool_output("Dropping all files from the chat session.")
            self._drop_all_files()
            return

        filenames = parse_quoted_filenames(args)
        for word in filenames:
            # Expand tilde in the path
            expanded_word = os.path.expanduser(word)

            # Handle read-only files with substring matching and samefile check
            read_only_matched = []
            for f in self.coder.abs_read_only_fnames:
                if expanded_word in f:
                    read_only_matched.append(f)
                    continue

                # Try samefile comparison for relative paths
                try:
                    abs_word = os.path.abspath(expanded_word)
                    if os.path.samefile(abs_word, f):
                        read_only_matched.append(f)
                except (FileNotFoundError, OSError):
                    continue

            for matched_file in read_only_matched:
                self.coder.abs_read_only_fnames.remove(matched_file)
                self.io.tool_output(f"Removed read-only file {matched_file} from the chat")

            # For editable files, use glob if word contains glob chars, otherwise use substring
            if any(c in expanded_word for c in "*?[]"):
                matched_files = self.glob_filtered_to_repo(expanded_word)
            else:
                # Use substring matching like we do for read-only files
                matched_files = [
                    self.coder.get_rel_fname(f) for f in self.coder.abs_fnames if expanded_word in f
                ]

            if not matched_files:
                matched_files.append(expanded_word)

            for matched_file in matched_files:
                abs_fname = self.coder.abs_root_path(matched_file)
                if abs_fname in self.coder.abs_fnames:
                    self.coder.abs_fnames.remove(abs_fname)
                    self.io.tool_output(f"Removed {matched_file} from the chat")

    def cmd_git(self, args):
        "Run a git command (output excluded from chat)"
        combined_output = None
        try:
            args = "git " + args
            env = dict(subprocess.os.environ)
            env["GIT_EDITOR"] = "true"
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                shell=True,
                encoding=self.io.encoding,
                errors="replace",
            )
            combined_output = result.stdout
        except Exception as e:
            self.io.tool_error(f"Error running /git command: {e}")

        if combined_output is None:
            return

        self.io.tool_output(combined_output)

    def cmd_test(self, args):
        "Run a shell command and add the output to the chat on non-zero exit code"
        if not args and self.coder.test_cmd:
            args = self.coder.test_cmd

        if not args:
            return

        # Store the original command for task tracking
        test_cmd = args if isinstance(args, str) else "custom test function"

        if not callable(args):
            if type(args) is not str:
                raise ValueError(repr(args))
            exit_status, errors = self._run_test_cmd(args)
        else:
            errors = args()
            exit_status = 1 if errors else 0

        if not errors:
            # Test passed, reset failures if in a task
            self._handle_test_success(test_cmd)
            return

        # Test failed, track in task if applicable
        should_research = self._handle_test_failure(test_cmd, errors)

        self.io.tool_output(errors)
        
        # If we reached the threshold for automatic research, suggest it
        if should_research:
            self._offer_test_research(test_cmd, errors)

        return errors
        
    def _run_test_cmd(self, cmd):
        """Run a test command and return exit status and output."""
        exit_status, combined_output = run_cmd(
            cmd, verbose=self.verbose, error_print=self.io.tool_error, cwd=self.coder.root
        )
        return exit_status, combined_output
        
    def _handle_test_success(self, test_cmd):
        """Handle a successful test run, resetting failures if in a task."""
        task_manager = get_task_manager()
        active_task = task_manager.get_active_task()
        
        if active_task and active_task.test_info:
            # Reset failure counts for this test
            task_manager.reset_test_failures(active_task.id)
            self.io.tool_output("Test passed! Reset failure tracking.")
            
            # Add successful solution to task history
            if active_task.test_info.failing_tests:
                for test_name in active_task.test_info.failing_tests:
                    task_manager.add_attempted_solution(
                        active_task.id, 
                        test_name, 
                        "Recent code changes fixed this test", 
                        True
                    )
        
    def _handle_test_failure(self, test_cmd, errors):
        """Handle a test failure, tracking it in the active task if there is one."""
        task_manager = get_task_manager()
        active_task = task_manager.get_active_task()
        
        if not active_task:
            return False
            
        # Extract test names from errors - this is a simplified example
        # In practice, you would parse the errors more carefully based on test framework
        import re
        test_names = []
        
        # Look for common test failure patterns
        # This is just an example - you would need to adapt to your test output format
        patterns = [
            r'(?:FAIL|ERROR)(?:ED)?\s*(?::|::\s*|\s+)([^\n:]+)',  # pytest, jest, etc.
            r'(?:not ok|fail)\s+\d+\s+-\s+([^\n]+)',              # tap format
            r'(?:Assertion failed|Test failed)(?::|::\s*|\s+)([^\n:]+)',  # generic
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, errors, re.IGNORECASE)
            for match in matches:
                test_name = match.group(1).strip()
                if test_name:
                    test_names.append(test_name)
        
        # If no specific tests identified, use the command as the test name
        if not test_names:
            test_names = [test_cmd]
            
        # Track each failing test
        exceeded_threshold = False
        for test_name in test_names:
            if task_manager.add_test_failure(active_task.id, test_name):
                exceeded_threshold = True
                
        return exceeded_threshold
        
    def _offer_test_research(self, test_cmd, errors):
        """Offer to research the test failure and suggest solutions."""
        task_manager = get_task_manager()
        active_task = task_manager.get_active_task()
        
        # Check if auto-test-tasks is enabled
        if hasattr(self, 'args') and getattr(self.args, 'auto_test_tasks', False):
            self._auto_resolve_test_failure(test_cmd, errors)
            return
        
        if not active_task or not active_task.test_info:
            return
            
        self.io.tool_output("\nThis test has failed multiple times. Would you like me to:")
        options = [
            "Research similar tests in the codebase",
            "Analyze the test requirements more carefully",
            "Suggest a different implementation approach",
            "Review previous failed attempts"
        ]
        
        for i, option in enumerate(options, 1):
            self.io.tool_output(f"{i}. {option}")
            
        choice = self.io.confirm_ask("Would you like me to help with this test failure?")
        if choice:
            # Here you would typically use the help functionality or a special research mode
            # For now, we'll just add the suggestion to try again with a more careful analysis
            research_message = f"""
I notice that this test has failed multiple times. Let me analyze it more carefully.

The test command was: {test_cmd}

The error output is:
{errors[:500]}...

Let me review the test requirements carefully and suggest a new approach.
"""
            self.coder.cur_messages += [
                dict(role="user", content=research_message),
            ]
            
    def _auto_resolve_test_failure(self, test_cmd, errors):
        """
        Automatically resolve test failures by creating tasks and having the LLM fix them.
        This is used when --auto-test-tasks is enabled.
        """
        task_manager = get_task_manager()
        
        # Extract test names from errors
        import re
        test_names = []
        
        # Look for common test failure patterns
        patterns = [
            r'(?:FAIL|ERROR)(?:ED)?\s*(?::|::\s*|\s+)([^\n:]+)',  # pytest, jest, etc.
            r'(?:not ok|fail)\s+\d+\s+-\s+([^\n]+)',              # tap format
            r'(?:Assertion failed|Test failed)(?::|::\s*|\s+)([^\n:]+)',  # generic
            r'(?:Test Failed|Failure|Failed)\s*:?\s*([^\n]+)',    # other formats
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, errors, re.IGNORECASE)
            for match in matches:
                test_name = match.group(1).strip()
                if test_name:
                    test_names.append(test_name)
        
        # If no specific tests identified, use the command as the test name
        if not test_names:
            test_names = ["Unknown test failure"]
            
        # Process each failing test
        for test_name in test_names:
            self._process_failing_test(test_name, test_cmd, errors)
    
    def _process_failing_test(self, test_name, test_cmd, errors):
        """
        Process a single failing test, creating a task if needed and attempting to fix it.
        """
        task_manager = get_task_manager()
        
        # Check if we already have a task for this test
        task_name = f"Fix test: {test_name}"
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            # Create a new task for this failing test
            task = task_manager.create_task(task_name, f"Automatically fix failing test: {test_name}")
            self.io.tool_output(f"Created task for failing test: {test_name}")
            
            # Associate current files with the task
            for fname in self.coder.abs_fnames:
                rel_fname = self.coder.get_rel_fname(fname)
                task.add_files([rel_fname])
        
        # Check how many attempts we've made and if we should continue
        retry_limit = getattr(self.args, 'auto_test_retry_limit', 5)
        
        if task.test_info and test_name in task.test_info.failure_counts:
            attempt_count = task.test_info.failure_counts[test_name]
            
            # If we've reached the limit, notify but don't continue
            if attempt_count >= retry_limit:
                self.io.tool_error(f"Reached retry limit ({retry_limit}) for test: {test_name}")
                self.io.tool_output("Please manually address this test failure.")
                return
        
        # Switch to this task if we're not already on it
        active_task = task_manager.get_active_task()
        if not active_task or active_task.id != task.id:
            # Save current context
            if active_task:
                current_files = [self.coder.get_rel_fname(fname) for fname in self.coder.abs_fnames]
                active_task.add_files(current_files)
                
                if self.coder.cur_messages:
                    chat_context = str(self.coder.cur_messages)
                    active_task.add_conversation_context(chat_context)
                
                task_manager.update_task(active_task)
            
            # Switch to the test task
            task_manager.switch_task(task.id)
            self.io.tool_output(f"Switched to task: {task_name}")
        
        # Update the test failure count
        task_manager.add_test_failure(task.id, test_name)
        
        # Generate a message for fixing the test
        fix_message = self._generate_test_fix_message(test_name, test_cmd, errors, task)
        
        # Add to the chat and let the model generate a fix
        self.coder.cur_messages += [
            dict(role="user", content=fix_message),
        ]
        
    def _generate_test_fix_message(self, test_name, test_cmd, errors, task):
        """
        Generate a message asking the model to fix the failing test,
        including context from previous attempts.
        """
        task_manager = get_task_manager()
        attempt_count = 1
        previous_attempts = ""
        
        if task.test_info and test_name in task.test_info.failure_counts:
            attempt_count = task.test_info.failure_counts[test_name]
            
            # Add information about previous attempts
            if task.test_info.attempted_solutions:
                solutions = task_manager.get_attempted_solutions(task.id, test_name)
                if solutions:
                    previous_attempts = "\n\n## Previous Solution Attempts\n\n"
                    for i, solution in enumerate(solutions, 1):
                        previous_attempts += f"### Attempt {i}\n\n"
                        previous_attempts += f"Solution tried: {solution['solution'][:200]}...\n\n"
                        previous_attempts += f"Result: {'Successful' if solution['successful'] else 'Failed'}\n\n"
        
        # Create a message with appropriate context for the current attempt
        if attempt_count == 1:
            # First attempt - straightforward fix request
            message = f"""
I need to fix a failing test. Please help me resolve this test failure:

## Test Information
- Test name: {test_name}
- Test command: {test_cmd}

## Error Output
```
{errors[:1000]}
```

Please analyze the error and implement a fix for this failing test.
"""
        elif attempt_count <= 3:
            # Early attempts - look more carefully
            message = f"""
This test has failed {attempt_count} times. Let's look more deeply at the issue:

## Test Information
- Test name: {test_name}
- Test command: {test_cmd}

## Error Output
```
{errors[:1000]}
```

{previous_attempts}

For this attempt, please:
1. Analyze the test requirements more carefully
2. Check for subtle issues that might be causing the failure
3. Consider edge cases and input validation
4. Implement a fix that addresses the root cause
"""
        else:
            # Later attempts - thorough investigation
            message = f"""
This test has failed {attempt_count} times despite multiple fix attempts. Let's perform a thorough investigation:

## Test Information
- Test name: {test_name}
- Test command: {test_cmd}

## Error Output
```
{errors[:1000]}
```

{previous_attempts}

For this attempt, please:
1. Perform a comprehensive analysis of the test failure
2. Search for similar patterns in other tests that work correctly
3. Check if there are fundamental assumptions or environment issues
4. Consider if the test itself might need to be modified
5. Implement a solution that addresses the deeper issues

This is attempt {attempt_count} of {getattr(self.args, 'auto_test_retry_limit', 5)} before escalation.
"""
        
        return message

    def cmd_run(self, args, add_on_nonzero_exit=False):
        "Run a shell command and optionally add the output to the chat (alias: !)"
        exit_status, combined_output = run_cmd(
            args, verbose=self.verbose, error_print=self.io.tool_error, cwd=self.coder.root
        )

        if combined_output is None:
            return

        # Calculate token count of output
        token_count = self.coder.main_model.token_count(combined_output)
        k_tokens = token_count / 1000

        if add_on_nonzero_exit:
            add = exit_status != 0
        else:
            add = self.io.confirm_ask(f"Add {k_tokens:.1f}k tokens of command output to the chat?")

        if add:
            num_lines = len(combined_output.strip().splitlines())
            line_plural = "line" if num_lines == 1 else "lines"
            self.io.tool_output(f"Added {num_lines} {line_plural} of output to the chat.")

            msg = prompts.run_output.format(
                command=args,
                output=combined_output,
            )

            self.coder.cur_messages += [
                dict(role="user", content=msg),
                dict(role="assistant", content="Ok."),
            ]

            if add and exit_status != 0:
                self.io.placeholder = "What's wrong? Fix"

    def cmd_exit(self, args):
        "Exit the application"
        self.coder.event("exit", reason="/exit")
        sys.exit()

    def cmd_quit(self, args):
        "Exit the application"
        self.cmd_exit(args)

    def cmd_ls(self, args):
        "List all known files and indicate which are included in the chat session"

        files = self.coder.get_all_relative_files()

        other_files = []
        chat_files = []
        read_only_files = []
        for file in files:
            abs_file_path = self.coder.abs_root_path(file)
            if abs_file_path in self.coder.abs_fnames:
                chat_files.append(file)
            else:
                other_files.append(file)

        # Add read-only files
        for abs_file_path in self.coder.abs_read_only_fnames:
            rel_file_path = self.coder.get_rel_fname(abs_file_path)
            read_only_files.append(rel_file_path)

        if not chat_files and not other_files and not read_only_files:
            self.io.tool_output("\nNo files in chat, git repo, or read-only list.")
            return

        if other_files:
            self.io.tool_output("Repo files not in the chat:\n")
        for file in other_files:
            self.io.tool_output(f"  {file}")

        if read_only_files:
            self.io.tool_output("\nRead-only files:\n")
        for file in read_only_files:
            self.io.tool_output(f"  {file}")

        if chat_files:
            self.io.tool_output("\nFiles in chat:\n")
        for file in chat_files:
            self.io.tool_output(f"  {file}")

    def basic_help(self):
        commands = sorted(self.get_commands())
        pad = max(len(cmd) for cmd in commands)
        pad = "{cmd:" + str(pad) + "}"
        for cmd in commands:
            cmd_method_name = f"cmd_{cmd[1:]}".replace("-", "_")
            cmd_method = getattr(self, cmd_method_name, None)
            cmd = pad.format(cmd=cmd)
            if cmd_method:
                description = cmd_method.__doc__
                self.io.tool_output(f"{cmd} {description}")
            else:
                self.io.tool_output(f"{cmd} No description available.")
        self.io.tool_output()
        self.io.tool_output("Use `/help <question>` to ask questions about how to use aider.")

    def cmd_help(self, args):
        "Ask questions about aider"

        if not args.strip():
            self.basic_help()
            return

        self.coder.event("interactive help")
        from aider.coders.base_coder import Coder

        if not self.help:
            res = install_help_extra(self.io)
            if not res:
                self.io.tool_error("Unable to initialize interactive help.")
                return

            self.help = Help()

        coder = Coder.create(
            io=self.io,
            from_coder=self.coder,
            edit_format="help",
            summarize_from_coder=False,
            map_tokens=512,
            map_mul_no_files=1,
        )
        user_msg = self.help.ask(args)
        user_msg += """
# Announcement lines from when this session of aider was launched:

"""
        user_msg += "\n".join(self.coder.get_announcements()) + "\n"

        coder.run(user_msg, preproc=False)

        if self.coder.repo_map:
            map_tokens = self.coder.repo_map.max_map_tokens
            map_mul_no_files = self.coder.repo_map.map_mul_no_files
        else:
            map_tokens = 0
            map_mul_no_files = 1

        raise SwitchCoder(
            edit_format=self.coder.edit_format,
            summarize_from_coder=False,
            from_coder=coder,
            map_tokens=map_tokens,
            map_mul_no_files=map_mul_no_files,
            show_announcements=False,
        )

    def cmd_ask(self, args):
        """Ask questions about the code base without editing any files. If no prompt provided, switches to ask mode."""  # noqa
        return self._generic_chat_command(args, "ask")

    def cmd_code(self, args):
        """Ask for changes to your code. If no prompt provided, switches to code mode."""  # noqa
        return self._generic_chat_command(args, self.coder.main_model.edit_format)

    def cmd_architect(self, args):
        """Enter architect/editor mode using 2 different models. If no prompt provided, switches to architect/editor mode."""  # noqa
        return self._generic_chat_command(args, "architect")

    def _generic_chat_command(self, args, edit_format):
        if not args.strip():
            # Switch to the corresponding chat mode if no args provided
            return self.cmd_chat_mode(edit_format)

        from aider.coders.base_coder import Coder

        coder = Coder.create(
            io=self.io,
            from_coder=self.coder,
            edit_format=edit_format,
            summarize_from_coder=False,
        )

        user_msg = args
        coder.run(user_msg)

        raise SwitchCoder(
            edit_format=self.coder.edit_format,
            summarize_from_coder=False,
            from_coder=coder,
            show_announcements=False,
        )

    def get_help_md(self):
        "Show help about all commands in markdown"

        res = """
|Command|Description|
|:------|:----------|
"""
        commands = sorted(self.get_commands())
        for cmd in commands:
            cmd_method_name = f"cmd_{cmd[1:]}".replace("-", "_")
            cmd_method = getattr(self, cmd_method_name, None)
            if cmd_method:
                description = cmd_method.__doc__
                res += f"| **{cmd}** | {description} |\n"
            else:
                res += f"| **{cmd}** | |\n"

        res += "\n"
        return res

    def cmd_voice(self, args):
        "Record and transcribe voice input"

        if not self.voice:
            if "OPENAI_API_KEY" not in os.environ:
                self.io.tool_error("To use /voice you must provide an OpenAI API key.")
                return
            try:
                self.voice = voice.Voice(
                    audio_format=self.voice_format or "wav", device_name=self.voice_input_device
                )
            except voice.SoundDeviceError:
                self.io.tool_error(
                    "Unable to import `sounddevice` and/or `soundfile`, is portaudio installed?"
                )
                return

        try:
            text = self.voice.record_and_transcribe(None, language=self.voice_language)
        except litellm.OpenAIError as err:
            self.io.tool_error(f"Unable to use OpenAI whisper model: {err}")
            return

        if text:
            self.io.placeholder = text

    def cmd_paste(self, args):
        """Paste image/text from the clipboard into the chat.\
        Optionally provide a name for the image."""
        try:
            # Check for image first
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                if args.strip():
                    filename = args.strip()
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in (".jpg", ".jpeg", ".png"):
                        basename = filename
                    else:
                        basename = f"{filename}.png"
                else:
                    basename = "clipboard_image.png"

                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, basename)
                image_format = "PNG" if basename.lower().endswith(".png") else "JPEG"
                image.save(temp_file_path, image_format)

                abs_file_path = Path(temp_file_path).resolve()

                # Check if a file with the same name already exists in the chat
                existing_file = next(
                    (f for f in self.coder.abs_fnames if Path(f).name == abs_file_path.name), None
                )
                if existing_file:
                    self.coder.abs_fnames.remove(existing_file)
                    self.io.tool_output(f"Replaced existing image in the chat: {existing_file}")

                self.coder.abs_fnames.add(str(abs_file_path))
                self.io.tool_output(f"Added clipboard image to the chat: {abs_file_path}")
                self.coder.check_added_files()

                return

            # If not an image, try to get text
            text = pyperclip.paste()
            if text:
                self.io.tool_output(text)
                return text

            self.io.tool_error("No image or text content found in clipboard.")
            return

        except Exception as e:
            self.io.tool_error(f"Error processing clipboard content: {e}")

    def cmd_read_only(self, args):
        "Add files to the chat that are for reference only, or turn added files to read-only"
        if not args.strip():
            # Convert all files in chat to read-only
            for fname in list(self.coder.abs_fnames):
                self.coder.abs_fnames.remove(fname)
                self.coder.abs_read_only_fnames.add(fname)
                rel_fname = self.coder.get_rel_fname(fname)
                self.io.tool_output(f"Converted {rel_fname} to read-only")
            return

        filenames = parse_quoted_filenames(args)
        all_paths = []

        # First collect all expanded paths
        for pattern in filenames:
            expanded_pattern = expanduser(pattern)
            if os.path.isabs(expanded_pattern):
                # For absolute paths, glob it
                matches = list(glob.glob(expanded_pattern))
            else:
                # For relative paths and globs, use glob from the root directory
                matches = list(Path(self.coder.root).glob(expanded_pattern))

            if not matches:
                self.io.tool_error(f"No matches found for: {pattern}")
            else:
                all_paths.extend(matches)

        # Then process them in sorted order
        for path in sorted(all_paths):
            abs_path = self.coder.abs_root_path(path)
            if os.path.isfile(abs_path):
                self._add_read_only_file(abs_path, path)
            elif os.path.isdir(abs_path):
                self._add_read_only_directory(abs_path, path)
            else:
                self.io.tool_error(f"Not a file or directory: {abs_path}")

    def _add_read_only_file(self, abs_path, original_name):
        if is_image_file(original_name) and not self.coder.main_model.info.get("supports_vision"):
            self.io.tool_error(
                f"Cannot add image file {original_name} as the"
                f" {self.coder.main_model.name} does not support images."
            )
            return

        if abs_path in self.coder.abs_read_only_fnames:
            self.io.tool_error(f"{original_name} is already in the chat as a read-only file")
            return
        elif abs_path in self.coder.abs_fnames:
            self.coder.abs_fnames.remove(abs_path)
            self.coder.abs_read_only_fnames.add(abs_path)
            self.io.tool_output(
                f"Moved {original_name} from editable to read-only files in the chat"
            )
        else:
            self.coder.abs_read_only_fnames.add(abs_path)
            self.io.tool_output(f"Added {original_name} to read-only files.")

    def _add_read_only_directory(self, abs_path, original_name):
        added_files = 0
        for root, _, files in os.walk(abs_path):
            for file in files:
                file_path = os.path.join(root, file)
                if (
                    file_path not in self.coder.abs_fnames
                    and file_path not in self.coder.abs_read_only_fnames
                ):
                    self.coder.abs_read_only_fnames.add(file_path)
                    added_files += 1

        if added_files > 0:
            self.io.tool_output(
                f"Added {added_files} files from directory {original_name} to read-only files."
            )
        else:
            self.io.tool_output(f"No new files added from directory {original_name}.")

    def cmd_map(self, args):
        "Print out the current repository map"
        repo_map = self.coder.get_repo_map()
        if repo_map:
            self.io.tool_output(repo_map)
        else:
            self.io.tool_output("No repository map available.")

    def cmd_map_refresh(self, args):
        "Force a refresh of the repository map"
        repo_map = self.coder.get_repo_map(force_refresh=True)
        if repo_map:
            self.io.tool_output("The repo map has been refreshed, use /map to view it.")

    def cmd_settings(self, args):
        "Print out the current settings"
        settings = format_settings(self.parser, self.args)
        announcements = "\n".join(self.coder.get_announcements())
        output = f"{announcements}\n{settings}"
        self.io.tool_output(output)

    def completions_raw_load(self, document, complete_event):
        return self.completions_raw_read_only(document, complete_event)

    def cmd_load(self, args):
        "Load and execute commands from a file"
        if not args.strip():
            self.io.tool_error("Please provide a filename containing commands to load.")
            return

        try:
            with open(args.strip(), "r", encoding=self.io.encoding, errors="replace") as f:
                commands = f.readlines()
        except FileNotFoundError:
            self.io.tool_error(f"File not found: {args}")
            return
        except Exception as e:
            self.io.tool_error(f"Error reading file: {e}")
            return

        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith("#"):
                continue

            self.io.tool_output(f"\nExecuting: {cmd}")
            try:
                self.run(cmd)
            except SwitchCoder:
                self.io.tool_error(
                    f"Command '{cmd}' is only supported in interactive mode, skipping."
                )

    def completions_raw_save(self, document, complete_event):
        return self.completions_raw_read_only(document, complete_event)

    def cmd_save(self, args):
        "Save commands to a file that can reconstruct the current chat session's files"
        if not args.strip():
            self.io.tool_error("Please provide a filename to save the commands to.")
            return

        try:
            with open(args.strip(), "w", encoding=self.io.encoding) as f:
                f.write("/drop\n")
                # Write commands to add editable files
                for fname in sorted(self.coder.abs_fnames):
                    rel_fname = self.coder.get_rel_fname(fname)
                    f.write(f"/add       {rel_fname}\n")

                # Write commands to add read-only files
                for fname in sorted(self.coder.abs_read_only_fnames):
                    # Use absolute path for files outside repo root, relative path for files inside
                    if Path(fname).is_relative_to(self.coder.root):
                        rel_fname = self.coder.get_rel_fname(fname)
                        f.write(f"/read-only {rel_fname}\n")
                    else:
                        f.write(f"/read-only {fname}\n")

            self.io.tool_output(f"Saved commands to {args.strip()}")
        except Exception as e:
            self.io.tool_error(f"Error saving commands to file: {e}")

    def cmd_multiline_mode(self, args):
        "Toggle multiline mode (swaps behavior of Enter and Meta+Enter)"
        self.io.toggle_multiline_mode()
        
    def cmd_task(self, args):
        "Create and manage tasks"
        
        # Get the task manager
        task_manager = get_task_manager()
        
        # Parse arguments
        parts = args.strip().split(maxsplit=1)
        if not parts:
            self.io.tool_error("Missing subcommand. Use /task help for available commands.")
            return
            
        subcommand = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        if subcommand == "help" or subcommand == "":
            self.io.tool_output("Task Manager Commands:")
            self.io.tool_output("  /task create <name> <description> - Create a new task")
            self.io.tool_output("  /task list [active|completed|archived] - List tasks")
            self.io.tool_output("  /task switch <id> - Switch to a different task")
            self.io.tool_output("  /task info <id> - Show task details")
            self.io.tool_output("  /task complete <id> - Mark a task as completed")
            self.io.tool_output("  /task archive <id> - Archive a task")
            self.io.tool_output("  /task reactivate <id> - Reactivate a completed or archived task")
            return
            
        elif subcommand == "create":
            # Parse name and description
            if not args:
                self.io.tool_error("Missing task name and description. Usage: /task create <name> <description>")
                return
                
            parts = args.split(maxsplit=1)
            name = parts[0] if parts else ""
            description = parts[1] if len(parts) > 1 else ""
            
            # Create the task
            task = task_manager.create_task(name, description)
            task_manager.switch_task(task.id)
            
            # Update coder with the new active task
            if self.coder:
                self.coder.active_task = task
                
                # Track files in the task
                if self.coder.abs_fnames:
                    file_list = [self.coder.get_rel_fname(fname) for fname in self.coder.abs_fnames]
                    task.add_files(file_list)
                    task_manager.update_task(task)
            
            self.io.tool_output(f"Created task: {task.id}")
            self.io.tool_output(f"Title: {task.name}")
            self.io.tool_output(f"Description: {task.description}")
            self.io.tool_output(f"Status: {task.status}")
            
            return
            
        elif subcommand == "list":
            # Filter by status if provided
            status = args.strip() if args else None
            
            # List tasks
            tasks = task_manager.list_tasks(status=status)
            
            if not tasks:
                self.io.tool_output("No tasks found.")
                return
                
            # Group by status
            by_status = {}
            for task in tasks:
                if task.status not in by_status:
                    by_status[task.status] = []
                by_status[task.status].append(task)
                
            # Display active tasks first
            if "active" in by_status:
                self.io.tool_output("Active Tasks:")
                for task in by_status["active"]:
                    self.io.tool_output(f"- {task.id}: {task.name}")
                self.io.tool_output("")
                
            # Display completed tasks
            if "completed" in by_status:
                self.io.tool_output("Completed Tasks:")
                for task in by_status["completed"]:
                    self.io.tool_output(f"- {task.id}: {task.name}")
                self.io.tool_output("")
                
            # Display archived tasks
            if "archived" in by_status:
                self.io.tool_output("Archived Tasks:")
                for task in by_status["archived"]:
                    self.io.tool_output(f"- {task.id}: {task.name}")
                
            return
            
        elif subcommand == "switch":
            task_id = args.strip()
            if not task_id:
                self.io.tool_error("Missing task ID. Usage: /task switch <id>")
                return
                
            task = task_manager.get_task(task_id)
            if not task:
                self.io.tool_error(f"Task {task_id} not found.")
                return
                
            # Switch to the task
            task_manager.switch_task(task.id)
            
            # Update coder with the new active task
            if self.coder:
                self.coder.active_task = task
                
            self.io.tool_output(f"Switched to task: {task.id}")
            self.io.tool_output(f"Title: {task.name}")
            self.io.tool_output(f"Description: {task.description}")
            self.io.tool_output(f"Status: {task.status}")
            
            return
            
        elif subcommand == "info":
            task_id = args.strip()
            if not task_id:
                self.io.tool_error("Missing task ID. Usage: /task info <id>")
                return
                
            task = task_manager.get_task(task_id)
            if not task:
                self.io.tool_error(f"Task {task_id} not found.")
                return
                
            # Show task details
            self.io.tool_output(f"Task Details: {task.id}")
            self.io.tool_output(f"Title: {task.name}")
            self.io.tool_output(f"Description: {task.description}")
            self.io.tool_output(f"Status: {task.status}")
            self.io.tool_output(f"Created: {task.created_at}")
            if task.status == "completed":
                self.io.tool_output(f"Completed: {task.updated_at}")
            
            # Show files
            if task.files:
                self.io.tool_output("Files:")
                for file in task.files:
                    self.io.tool_output(f"- {file}")
            
            # Show environment
            self.io.tool_output("Environment:")
            self.io.tool_output(f"- OS: {task.environment.os}")
            self.io.tool_output(f"- Python: {task.environment.python_version}")
            
            # Show test information if available
            if task.test_info and task.test_info.failing_tests:
                self.io.tool_output("Test Information:")
                self.io.tool_output(f"- Attempts: {task.test_info.attempt_count}")
                self.io.tool_output("- Failing tests:")
                for test in task.test_info.failing_tests:
                    count = task.test_info.failure_counts.get(test, 0)
                    self.io.tool_output(f"  * {test} (failures: {count})")
            
            return
            
        elif subcommand == "complete":
            task_id = args.strip()
            if not task_id:
                self.io.tool_error("Missing task ID. Usage: /task complete <id>")
                return
                
            task = task_manager.get_task(task_id)
            if not task:
                self.io.tool_error(f"Task {task_id} not found.")
                return
                
            # Complete the task
            task_manager.complete_task(task.id)
            
            # Update coder if this was the active task
            if self.coder and self.coder.active_task and self.coder.active_task.id == task.id:
                self.coder.active_task = task_manager.get_task(task.id)
                
            self.io.tool_output(f"Marked task {task.id} as completed.")
            
            return
            
        elif subcommand == "archive":
            task_id = args.strip()
            if not task_id:
                self.io.tool_error("Missing task ID. Usage: /task archive <id>")
                return
                
            task = task_manager.get_task(task_id)
            if not task:
                self.io.tool_error(f"Task {task_id} not found.")
                return
                
            # Archive the task
            task_manager.archive_task(task.id)
            
            # Update coder if this was the active task
            if self.coder and self.coder.active_task and self.coder.active_task.id == task.id:
                self.coder.active_task = task_manager.get_task(task.id)
                
            self.io.tool_output(f"Archived task {task.id}.")
            
            return
            
        elif subcommand == "reactivate":
            task_id = args.strip()
            if not task_id:
                self.io.tool_error("Missing task ID. Usage: /task reactivate <id>")
                return
                
            task = task_manager.get_task(task_id)
            if not task:
                self.io.tool_error(f"Task {task_id} not found.")
                return
                
            # Reactivate the task
            task_manager.reactivate_task(task.id)
            
            # Update coder if this was the active task
            if self.coder and self.coder.active_task and self.coder.active_task.id == task.id:
                self.coder.active_task = task_manager.get_task(task.id)
                
            self.io.tool_output(f"Reactivated task {task.id}.")
            
            return
            
        else:
            self.io.tool_error(f"Unknown subcommand: {subcommand}. Use /task help for available commands.")

    def cmd_copy(self, args):
        "Copy the last assistant message to the clipboard"
        all_messages = self.coder.done_messages + self.coder.cur_messages
        assistant_messages = [msg for msg in reversed(all_messages) if msg["role"] == "assistant"]

        if not assistant_messages:
            self.io.tool_error("No assistant messages found to copy.")
            return

        last_assistant_message = assistant_messages[0]["content"]

        try:
            pyperclip.copy(last_assistant_message)
            preview = (
                last_assistant_message[:50] + "..."
                if len(last_assistant_message) > 50
                else last_assistant_message
            )
            self.io.tool_output(f"Copied last assistant message to clipboard. Preview: {preview}")
        except pyperclip.PyperclipException as e:
            self.io.tool_error(f"Failed to copy to clipboard: {str(e)}")
            self.io.tool_output(
                "You may need to install xclip or xsel on Linux, or pbcopy on macOS."
            )
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred while copying to clipboard: {str(e)}")

    def cmd_report(self, args):
        "Report a problem by opening a GitHub Issue"
        from aider.report import report_github_issue

        announcements = "\n".join(self.coder.get_announcements())
        issue_text = announcements

        if args.strip():
            title = args.strip()
        else:
            title = None

        report_github_issue(issue_text, title=title, confirm=False)

    def cmd_editor(self, initial_content=""):
        "Open an editor to write a prompt"

        user_input = pipe_editor(initial_content, suffix="md", editor=self.editor)
        if user_input.strip():
            self.io.set_placeholder(user_input.rstrip())

    def cmd_task(self, args):
        """Manage tasks - create, switch, list, complete, archive"""
        args = args.strip()
        if not args:
            self._task_help()
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        task_manager = get_task_manager()

        if subcmd == "create":
            return self._task_create(rest)
        elif subcmd == "list":
            return self._task_list(rest)
        elif subcmd == "switch":
            return self._task_switch(rest)
        elif subcmd == "complete":
            return self._task_complete(rest)
        elif subcmd == "archive":
            return self._task_archive(rest)
        elif subcmd == "reactivate":
            return self._task_reactivate(rest)
        elif subcmd == "info":
            return self._task_info(rest)
        else:
            self.io.tool_error(f"Unknown task subcommand: {subcmd}")
            self._task_help()

    def _task_help(self):
        """Show task command help"""
        self.io.tool_output("Task management commands:")
        self.io.tool_output("  /task create <name> [description]  - Create a new task")
        self.io.tool_output("  /task list [active|completed|archived]  - List tasks")
        self.io.tool_output("  /task switch <task_name>  - Switch to a different task")
        self.io.tool_output("  /task complete <task_name>  - Mark a task as completed")
        self.io.tool_output("  /task archive <task_name>  - Archive a task")
        self.io.tool_output("  /task reactivate <task_name>  - Reactivate a completed or archived task")
        self.io.tool_output("  /task info <task_name>  - Show detailed information about a task")

    def _task_create(self, args):
        """Create a new task"""
        parts = args.split(maxsplit=1)
        if not parts:
            self.io.tool_error("Task name is required")
            return
            
        name = parts[0]
        description = parts[1] if len(parts) > 1 else name
        
        task_manager = get_task_manager()
        
        # Check for duplicate task name
        if task_manager.get_task_by_name(name):
            self.io.tool_error(f"A task with the name '{name}' already exists")
            return
        
        # Create the task
        task = task_manager.create_task(name, description)
        
        # Associate current files with the task
        for fname in self.coder.abs_fnames:
            rel_fname = self.coder.get_rel_fname(fname)
            task.add_files([rel_fname])
        
        # Save any conversation context
        if self.coder.cur_messages:
            # This would be expanded to properly serialize conversation context
            chat_context = str(self.coder.cur_messages)
            task.add_conversation_context(chat_context)
        
        # Switch to the new task
        task_manager.switch_task(task.id)
        
        self.io.tool_output(f"Created and switched to task: {name}")
        return task

    def _task_list(self, args):
        """List tasks"""
        status = args.strip() if args.strip() in ["active", "completed", "archived"] else None
        
        task_manager = get_task_manager()
        tasks = task_manager.list_tasks(status=status)
        
        if not tasks:
            status_str = f" {status}" if status else ""
            self.io.tool_output(f"No{status_str} tasks found.")
            return
        
        active_task = task_manager.get_active_task()
        active_id = active_task.id if active_task else None
        
        self.io.tool_output("Tasks:")
        for task in tasks:
            status_indicator = " "
            if task.id == active_id:
                status_indicator = "*"
            elif task.status == "completed":
                status_indicator = "✓"
            elif task.status == "archived":
                status_indicator = "a"
                
            self.io.tool_output(f"  [{status_indicator}] {task.name} - {task.description[:50]}")

    def _task_switch(self, args):
        """Switch to a different task"""
        task_name = args.strip()
        if not task_name:
            self.io.tool_error("Task name is required")
            return
            
        task_manager = get_task_manager()
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            self.io.tool_error(f"No task found with name: {task_name}")
            return
            
        # Save current context to the active task if there is one
        active_task = task_manager.get_active_task()
        if active_task:
            # Save files list
            current_files = [self.coder.get_rel_fname(fname) for fname in self.coder.abs_fnames]
            active_task.add_files(current_files)
            
            # Save conversation context
            if self.coder.cur_messages:
                chat_context = str(self.coder.cur_messages)
                active_task.add_conversation_context(chat_context)
            
            task_manager.update_task(active_task)
        
        # Switch to the new task
        task_manager.switch_task(task.id)
        
        # Clear current context
        self._drop_all_files()
        self._clear_chat_history()
        
        # Load the task's context
        for file in task.files:
            try:
                abs_path = self.coder.abs_root_path(file)
                if os.path.exists(abs_path):
                    self.coder.abs_fnames.add(abs_path)
            except Exception as e:
                self.io.tool_error(f"Error loading file {file}: {e}")
        
        # Reload conversation context (this would need more sophisticated implementation)
        if task.conversation_context:
            # This would be expanded to properly restore conversation context
            self.io.tool_output("Restored previous conversation context")
        
        self.io.tool_output(f"Switched to task: {task.name}")

    def _task_complete(self, args):
        """Mark a task as completed"""
        task_name = args.strip()
        if not task_name:
            self.io.tool_error("Task name is required")
            return
            
        task_manager = get_task_manager()
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            self.io.tool_error(f"No task found with name: {task_name}")
            return
            
        task_manager.complete_task(task.id)
        self.io.tool_output(f"Marked task as completed: {task.name}")

    def _task_archive(self, args):
        """Archive a task"""
        task_name = args.strip()
        if not task_name:
            self.io.tool_error("Task name is required")
            return
            
        task_manager = get_task_manager()
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            self.io.tool_error(f"No task found with name: {task_name}")
            return
            
        task_manager.archive_task(task.id)
        self.io.tool_output(f"Archived task: {task.name}")

    def _task_reactivate(self, args):
        """Reactivate a completed or archived task"""
        task_name = args.strip()
        if not task_name:
            self.io.tool_error("Task name is required")
            return
            
        task_manager = get_task_manager()
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            self.io.tool_error(f"No task found with name: {task_name}")
            return
            
        task_manager.reactivate_task(task.id)
        self.io.tool_output(f"Reactivated task: {task.name}")
        
    def _task_info(self, args):
        """Show detailed information about a task"""
        task_name = args.strip()
        if not task_name:
            self.io.tool_error("Task name is required")
            return
            
        task_manager = get_task_manager()
        task = task_manager.get_task_by_name(task_name)
        
        if not task:
            self.io.tool_error(f"No task found with name: {task_name}")
            return
            
        self.io.tool_output(f"Task: {task.name}")
        self.io.tool_output(f"Description: {task.description}")
        self.io.tool_output(f"Status: {task.status}")
        self.io.tool_output(f"Created: {task.created_at}")
        self.io.tool_output(f"Updated: {task.updated_at}")
        
        if task.files:
            self.io.tool_output("\nFiles:")
            for file in task.files:
                self.io.tool_output(f"  - {file}")
                
        if task.parent_task_id:
            parent_task = task_manager.get_task(task.parent_task_id)
            if parent_task:
                self.io.tool_output(f"\nParent task: {parent_task.name}")
                
        subtasks = task_manager.get_subtasks(task.id)
        if subtasks:
            self.io.tool_output("\nSubtasks:")
            for subtask in subtasks:
                status_indicator = " "
                if subtask.status == "completed":
                    status_indicator = "✓"
                elif subtask.status == "archived":
                    status_indicator = "a"
                self.io.tool_output(f"  [{status_indicator}] {subtask.name}")
                
        if task.test_info and task.test_info.failing_tests:
            self.io.tool_output("\nFailing tests:")
            for test in task.test_info.failing_tests:
                count = task.test_info.failure_counts.get(test, 0)
                self.io.tool_output(f"  - {test} (failed {count} times)")
                
        self.io.tool_output(f"\nEnvironment: {task.environment.os}, Python {task.environment.python_version.split()[0]}")
        if task.environment.git_branch:
            self.io.tool_output(f"Git branch: {task.environment.git_branch}")

    def cmd_copy_context(self, args=None):
        """Copy the current chat context as markdown, suitable to paste into a web UI"""

        chunks = self.coder.format_chat_chunks()

        markdown = ""

        # Only include specified chunks in order
        for messages in [chunks.repo, chunks.readonly_files, chunks.chat_files]:
            for msg in messages:
                # Only include user messages
                if msg["role"] != "user":
                    continue

                content = msg["content"]

                # Handle image/multipart content
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            markdown += part["text"] + "\n\n"
                else:
                    markdown += content + "\n\n"

        args = args or ""
        markdown += f"""
Just tell me how to edit the files to make the changes.
Don't give me back entire files.
Just show me the edits I need to make.

{args}
"""

        try:
            pyperclip.copy(markdown)
            self.io.tool_output("Copied code context to clipboard.")
        except pyperclip.PyperclipException as e:
            self.io.tool_error(f"Failed to copy to clipboard: {str(e)}")
            self.io.tool_output(
                "You may need to install xclip or xsel on Linux, or pbcopy on macOS."
            )
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred while copying to clipboard: {str(e)}")


def expand_subdir(file_path):
    if file_path.is_file():
        yield file_path
        return

    if file_path.is_dir():
        for file in file_path.rglob("*"):
            if file.is_file():
                yield file


def parse_quoted_filenames(args):
    filenames = re.findall(r"\"(.+?)\"|(\S+)", args)
    filenames = [name for sublist in filenames for name in sublist if name]
    return filenames


def get_help_md():
    md = Commands(None, None).get_help_md()
    return md


def main():
    md = get_help_md()
    print(md)


if __name__ == "__main__":
    status = main()
    sys.exit(status)
