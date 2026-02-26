import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.environments import BaseEnv, SupportsPersistence, get_environment
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils.exceptions import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.utils.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys
from rlm.utils.token_utils import count_tokens, get_context_limit


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        compaction_threshold_pct: float = 0.85,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None] | None = None,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
        agent_definition: dict[str, Any] | None = None,
        agent_allowed_tools: list[str] | None = None,
        agent_permission_mode: str = "bypassPermissions",
        agent_hooks: dict[str, Any] | None = None,
    ):
        """
        Args:
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of recursion. When depth >= max_depth, falls back to plain LM completion.
            max_iterations: The maximum number of iterations of the RLM.
            max_budget: Maximum budget in USD. Execution stops if exceeded. Requires cost-tracking backend (e.g., OpenRouter).
            max_timeout: Maximum execution time in seconds. Execution stops if exceeded, returning best answer if available.
            max_tokens: Maximum total tokens (input + output). Execution stops if exceeded, returning best answer if available.
            max_errors: Maximum consecutive errors before stopping. Execution stops if exceeded, returning best answer if available.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends that the environments can use to make sub-calls.
            other_backend_kwargs: The kwargs to pass to the other client backends (ordered to match other_backends).
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output in rich to console.
            persistent: If True, reuse the environment across completion() calls for multi-turn conversations.
            custom_tools: Dict of custom functions/tools available in the REPL. Keys are function names,
                values are callable functions. These are injected into the REPL globals.
            custom_sub_tools: Dict of custom tools for sub-agents (llm_query calls). If None, inherits
                from custom_tools. Pass an empty dict {} to disable tools for sub-agents.
            compaction: If True, keep full root model history in REPL variable `history` and compact
                when root context reaches compaction_threshold_pct of the model's context limit.
            compaction_threshold_pct: When compaction is on, trigger summarization when root
                message token count reaches this fraction of the model context limit (default 0.85).
            on_subcall_start: Callback fired when a child RLM starts. Args: (depth, model, prompt_preview).
            on_subcall_complete: Callback fired when a child RLM completes. Args: (depth, model, duration, error_or_none).
            on_iteration_start: Callback fired when an iteration starts. Args: (depth, iteration_num).
            on_iteration_complete: Callback fired when an iteration completes. Args: (depth, iteration_num, duration).
            agent_definition: Dict of named AgentDefinition objects for Claude Agent SDK execution.
                When provided with backend="vertex_anthropic", RLM bypasses REPL execution and uses
                Claude Agent SDK with specialized sub-agents. Each key is an agent name, each value
                is an AgentDefinition specifying tools, prompt, and model.
                Requires: pip install claude-agent-sdk
                Example:
                    from claude_agent_sdk import AgentDefinition
                    agents = {
                        "researcher": AgentDefinition(
                            description="Web research specialist",
                            tools=["WebSearch", "Write"],
                            prompt="You are a research agent...",
                            model="claude-opus-4-1"
                        )
                    }
                    rlm = RLM(
                        backend="vertex_anthropic",
                        agent_definition=agents,
                    )
            agent_allowed_tools: List of tool names available to the lead agent (not sub-agents).
                Defaults to ["Task"] when agent_definition is provided.
                Sub-agents get tools from their individual AgentDefinition.tools.
            agent_permission_mode: Permission mode for Claude Agent SDK.
                Options: "bypassPermissions" (default), "requirePermissions".
            agent_hooks: Hooks for tracking agent SDK tool use.
                Structure: {"PreToolUse": [HookMatcher(...)], "PostToolUse": [...]}
                Enables observability for agent execution.
        """
        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        # Validate other_backends: currently only support one additional backend
        if other_backends is not None:
            if len(other_backends) != 1:
                raise ValueError(
                    "We currently only support one additional backend for the recursive sub-calls! "
                    "This model will be the model used for recursive sub-calls, but this will change in the future"
                )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        # Custom tools: functions available in the REPL environment
        self.custom_tools = custom_tools
        # Sub-tools: if None, inherit from custom_tools; if {}, no tools for sub-agents
        self.custom_sub_tools = custom_sub_tools if custom_sub_tools is not None else custom_tools

        self.compaction = compaction
        self.compaction_threshold_pct = compaction_threshold_pct

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_budget = max_budget
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Agent SDK configuration
        self.agent_definition = agent_definition
        self.agent_allowed_tools = agent_allowed_tools
        self.agent_permission_mode = agent_permission_mode
        self.agent_hooks = agent_hooks

        # Event callbacks for live tree display
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

        # Tracking (cumulative across all calls including children)
        self._cumulative_cost: float = 0.0
        self._consecutive_errors: int = 0
        self._last_error: str | None = None
        self._best_partial_answer: str | None = None
        self._completion_start_time: float | None = None  # Set when completion() starts

        # Persistence support
        self.persistent = persistent
        self._persistent_env: SupportsPersistence | None = None

        # Validate persistence support at initialization
        if self.persistent:
            self._validate_persistent_environment_support()

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        Spawn an LM handler and environment for a single completion call.

        When persistent=True, the environment is reused across calls.
        When persistent=False (default), creates fresh environment each call.
        """
        # Create client and wrap in handler
        client: BaseLM = get_client(self.backend, self.backend_kwargs)

        # Create other_backend_client if provided (for depth=1 routing)
        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(self.other_backends[0], self.other_backend_kwargs[0])

        lm_handler = LMHandler(client, other_backend_client=other_backend_client)

        # Register other clients to be available as sub-call options (by model name)
        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs, strict=True):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        # Environment: reuse if persistent, otherwise create fresh
        if self.persistent and self._persistent_env is not None:
            environment = self._persistent_env
            # Defensive check: ensure environment supports persistence methods
            if not self._env_supports_persistence(environment):
                raise RuntimeError(
                    f"Persistent environment of type '{type(environment).__name__}' does not "
                    f"implement required methods (update_handler_address, add_context, get_context_count). "
                    f"This should have been caught at initialization."
                )
            environment.update_handler_address((lm_handler.host, lm_handler.port))
            environment.add_context(prompt)
        else:
            env_kwargs = self.environment_kwargs.copy()
            env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
            env_kwargs["context_payload"] = prompt
            env_kwargs["depth"] = self.depth + 1  # Environment depth is RLM depth + 1
            # For local environment with max_depth > 1, pass subcall callback for recursive RLM calls
            if self.environment_type == "local" and self.max_depth > 1:
                env_kwargs["subcall_fn"] = self._subcall
            # Pass custom tools to the environment
            if self.custom_tools is not None:
                env_kwargs["custom_tools"] = self.custom_tools
            if self.custom_sub_tools is not None:
                env_kwargs["custom_sub_tools"] = self.custom_sub_tools
            if self.compaction and self.environment_type == "local":
                env_kwargs["compaction"] = True
            environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

            if self.persistent:
                self._persistent_env = environment

        try:
            yield lm_handler, environment
        finally:
            lm_handler.stop()
            if not self.persistent and hasattr(environment, "cleanup"):
                environment.cleanup()

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
            custom_tools=self.custom_tools,
        )
        if self.compaction:
            message_history[0]["content"] += (
                "\n\nThe full conversation history (trajectory segments and any summaries) "
                "is available in the REPL variable `history` as a list."
            )
        return message_history

    def _should_use_agent_sdk(self) -> bool:
        """Return True when Agent SDK should be used instead of REPL.

        Conditions: agent_definition is set, backend is "vertex_anthropic", and depth == 0.
        Sub-RLMs (depth > 0) always fall back to REPL even if agent_definition exists.

        Returns:
            True if Agent SDK should be used, False for REPL execution.
        """
        return (
            self.agent_definition is not None
            and self.backend == "vertex_anthropic"
            and self.depth == 0
        )

    def _validate_agent_sdk_config(self) -> None:
        """Validate Agent SDK configuration and warn about incompatibilities.

        Raises:
            ImportError: If claude-agent-sdk is not installed but agent_definition is provided.
        """
        if self.agent_definition is None:
            return

        # Validate SDK is installed
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "agent_definition requires 'claude-agent-sdk' package.\n"
                "Install with: pip install claude-agent-sdk\n"
                "Or: uv pip install claude-agent-sdk"
            ) from e

        # Warn if backend doesn't support Agent SDK
        if self.backend not in ["vertex_anthropic"]:
            import warnings

            warnings.warn(
                f"agent_definition provided but backend='{self.backend}' doesn't support Agent SDK. "
                f"Falling back to REPL execution. Use backend='vertex_anthropic' for Agent SDK support.",
                UserWarning,
                stacklevel=3,
            )

        # Warn about incompatible features
        self._warn_incompatible_features()

    def _warn_incompatible_features(self) -> None:
        """Warn if features incompatible with Agent SDK are set."""
        if not self._should_use_agent_sdk():
            return

        import warnings

        if self.custom_tools is not None:
            non_repl_tools = [k for k in self.custom_tools if k != "execute_in_repl"]
            if non_repl_tools:
                warnings.warn(
                    f"custom_tools keys {non_repl_tools} are ignored when using Agent SDK. "
                    "Define tools in AgentDefinition.tools instead. "
                    "Note: 'execute_in_repl' is auto-promoted to an MCP server tool.",
                    UserWarning,
                    stacklevel=4,
                )
            elif "execute_in_repl" in self.custom_tools:
                pass  # execute_in_repl is handled via MCP server; no warning needed

        if self.max_depth > 1:
            warnings.warn(
                "max_depth > 1 is ignored when using Agent SDK. "
                "Sub-agents are defined via agent_definition dict instead.",
                UserWarning,
                stacklevel=4,
            )

        if self.persistent:
            warnings.warn(
                "persistent mode is not supported with Agent SDK (stateless execution).",
                UserWarning,
                stacklevel=4,
            )

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """Run a completion, routing to Agent SDK or REPL based on configuration.

        Args:
            prompt: A single string or dictionary of messages to pass as context to the model.
            root_prompt: Optional small prompt visible to root LM (e.g., user's question).

        Returns:
            A final answer as RLMChatCompletion.
        """
        time_start = time.perf_counter()
        self._completion_start_time = time_start

        # Validate agent SDK config if specified
        if self.agent_definition is not None:
            self._validate_agent_sdk_config()

        # Route to appropriate execution path
        if self._should_use_agent_sdk():
            return self._completion_via_agent_sdk(prompt)

        # REPL execution path (original code below)
        # Reset tracking state for this completion
        self._consecutive_errors = 0
        self._last_error = None
        self._best_partial_answer = None
        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        if self.logger:
            self.logger.clear_iterations()

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            compaction_count = 0
            try:
                for i in range(self.max_iterations):
                    # Check timeout before each iteration
                    self._check_timeout(i, time_start)

                    # Compaction: check if context needs summarization
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        current_tokens, threshold_tokens, max_tokens = self._get_compaction_status(
                            message_history
                        )
                        self.verbose.print_compaction_status(
                            current_tokens, threshold_tokens, max_tokens
                        )
                        if current_tokens >= threshold_tokens:
                            compaction_count += 1
                            self.verbose.print_compaction()
                            message_history = self._compact_history(
                                lm_handler, environment, message_history, compaction_count
                            )

                    # Current prompt = message history + additional prompt suffix
                    context_count = (
                        environment.get_context_count()
                        if isinstance(environment, SupportsPersistence)
                        else 1
                    )
                    history_count = (
                        environment.get_history_count()
                        if isinstance(environment, SupportsPersistence)
                        else 0
                    )
                    current_prompt = message_history + [
                        build_user_prompt(root_prompt, i, context_count, history_count)
                    ]

                    iteration: RLMIteration = self._completion_turn(
                        prompt=current_prompt,
                        lm_handler=lm_handler,
                        environment=environment,
                    )

                    # Check error/budget/token limits after each iteration
                    self._check_iteration_limits(iteration, i, lm_handler)

                    # Check if RLM is done and has a final answer.
                    # Prefer FINAL_VAR result from REPL execution.
                    final_answer = None
                    for block in iteration.code_blocks:
                        if getattr(block.result, "final_answer", None):
                            final_answer = block.result.final_answer
                            break
                    if final_answer is None:
                        final_answer = find_final_answer(
                            iteration.response, environment=environment
                        )
                    iteration.final_answer = final_answer

                    # Store as best partial answer (most recent response with content)
                    if iteration.response and iteration.response.strip():
                        self._best_partial_answer = iteration.response

                    # If logger is used, log the iteration.
                    if self.logger:
                        self.logger.log(iteration)

                    # Verbose output for this iteration
                    self.verbose.print_iteration(iteration, i + 1)

                    if final_answer is not None:
                        time_end = time.perf_counter()
                        usage = lm_handler.get_usage_summary()
                        self.verbose.print_final_answer(final_answer)
                        self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                        # Store message history in persistent environment
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)

                        return RLMChatCompletion(
                            root_model=self.backend_kwargs.get("model_name", "unknown")
                            if self.backend_kwargs
                            else "unknown",
                            prompt=prompt,
                            response=final_answer,
                            usage_summary=usage,
                            execution_time=time_end - time_start,
                            metadata=self.logger.get_trajectory() if self.logger else None,
                        )

                    # Format the iteration for the next prompt.
                    new_messages = format_iteration(iteration)

                    # Update message history with the new messages.
                    message_history.extend(new_messages)
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        environment.append_compaction_entry(new_messages)

            except KeyboardInterrupt:
                self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
                raise CancellationError(
                    partial_answer=self._best_partial_answer,
                    message="Execution cancelled by user (Ctrl+C)",
                ) from None

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())

            # Store message history in persistent environment
            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
                metadata=self.logger.get_trajectory() if self.logger else None,
            )

    def _check_timeout(self, iteration: int, time_start: float) -> None:
        """Raise TimeoutExceededError if the timeout has been exceeded."""
        if self.max_timeout is None:
            return
        elapsed = time.perf_counter() - time_start
        if elapsed > self.max_timeout:
            self.verbose.print_limit_exceeded(
                "timeout",
                f"{elapsed:.1f}s of {self.max_timeout:.1f}s",
            )
            raise TimeoutExceededError(
                elapsed=elapsed,
                timeout=self.max_timeout,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Timeout exceeded after iteration {iteration}: "
                    f"{elapsed:.1f}s of {self.max_timeout:.1f}s limit"
                ),
            )

    def _check_iteration_limits(
        self, iteration: RLMIteration, iteration_num: int, lm_handler: LMHandler
    ) -> None:
        """Check error tracking, budget, and token limits after an iteration.

        Raises ErrorThresholdExceededError, BudgetExceededError, or TokenLimitExceededError
        if the respective limits are exceeded.
        """
        # Track errors from code execution (check stderr for errors)
        iteration_had_error = False
        for code_block in iteration.code_blocks:
            if code_block.result and code_block.result.stderr:
                iteration_had_error = True
                self._last_error = code_block.result.stderr
                break

        if iteration_had_error:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0  # Reset on success

        # Check error threshold
        if self.max_errors is not None and self._consecutive_errors >= self.max_errors:
            self.verbose.print_limit_exceeded(
                "errors",
                f"{self._consecutive_errors} consecutive errors (limit: {self.max_errors})",
            )
            raise ErrorThresholdExceededError(
                error_count=self._consecutive_errors,
                threshold=self.max_errors,
                last_error=self._last_error,
                partial_answer=self._best_partial_answer,
                message=(
                    "Error threshold exceeded: "
                    f"{self._consecutive_errors} consecutive errors "
                    f"(limit: {self.max_errors})"
                ),
            )

        # Check budget
        if self.max_budget is not None:
            current_usage = lm_handler.get_usage_summary()
            current_cost = current_usage.total_cost or 0.0
            self._cumulative_cost = current_cost
            if self._cumulative_cost > self.max_budget:
                self.verbose.print_budget_exceeded(self._cumulative_cost, self.max_budget)
                raise BudgetExceededError(
                    spent=self._cumulative_cost,
                    budget=self.max_budget,
                    message=(
                        f"Budget exceeded after iteration {iteration_num + 1}: "
                        f"spent ${self._cumulative_cost:.6f} "
                        f"of ${self.max_budget:.6f} budget"
                    ),
                )

        # Check token limit
        if self.max_tokens is not None:
            current_usage = lm_handler.get_usage_summary()
            total_tokens = current_usage.total_input_tokens + current_usage.total_output_tokens
            if total_tokens > self.max_tokens:
                self.verbose.print_limit_exceeded(
                    "tokens",
                    f"{total_tokens:,} of {self.max_tokens:,} tokens",
                )
                raise TokenLimitExceededError(
                    tokens_used=total_tokens,
                    token_limit=self.max_tokens,
                    partial_answer=self._best_partial_answer,
                    message=(
                        f"Token limit exceeded after iteration {iteration_num + 1}: "
                        f"{total_tokens:,} of {self.max_tokens:,} tokens"
                    ),
                )

    def _get_compaction_status(self, message_history: list[dict[str, Any]]) -> tuple[int, int, int]:
        """Return (current_tokens, threshold_tokens, max_tokens) for compaction."""
        model_name = (
            self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"
        )
        max_tokens = get_context_limit(model_name)
        current_tokens = count_tokens(message_history, model_name)
        threshold_tokens = int(self.compaction_threshold_pct * max_tokens)
        return current_tokens, threshold_tokens, max_tokens

    def _should_compact(self, message_history: list[dict[str, Any]]) -> bool:
        """True when root message history is at or over the compaction threshold."""
        current_tokens, threshold_tokens, _ = self._get_compaction_status(message_history)
        return current_tokens >= threshold_tokens

    def _compact_history(
        self,
        lm_handler: LMHandler,
        environment: BaseEnv,
        message_history: list[dict[str, Any]],
        compaction_count: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Summarize current trajectory, append summary to REPL history, and return
        a short message_history with the summary as the new starting point.
        """
        summary_prompt = message_history + [
            {
                "role": "user",
                "content": (
                    "Summarize your progress so far. Include:\n"
                    "1. Which steps/sub-tasks you have completed and which remain.\n"
                    "2. Any concrete intermediate results (numbers, values, variable names) "
                    "you computed — preserve these exactly.\n"
                    "3. What your next action should be.\n"
                    "Be concise (1–3 paragraphs) but preserve all key results and your "
                    "current position in the task."
                ),
            }
        ]
        summary = lm_handler.completion(summary_prompt)
        if hasattr(environment, "append_compaction_entry"):
            environment.append_compaction_entry({"type": "summary", "content": summary})
        # Keep system + initial assistant (metadata), then summary + continue
        new_history = message_history[:2] + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    f"Your conversation has been compacted {compaction_count} time(s). "
                    "Continue from the above summary. Do NOT repeat work you have already "
                    "completed. Use SHOW_VARS() to check which REPL variables exist, "
                    "and check `history` for full context. "
                    "Your next action:"
                ),
            },
        ]
        return new_history

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        response = client.completion(message)
        return response

    def _subcall(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        """Handle a subcall, spawning a child RLM or falling back to plain LM at max depth.

        Args:
            prompt: The prompt to process.
            model: Optional model override for the child RLM.
        Returns:
            RLMChatCompletion from the child RLM or plain LM; error as response on failure.
        """
        next_depth = self.depth + 1

        # Determine which backend/kwargs to use (model override or parent's default)
        if model is not None:
            child_backend_kwargs = (self.backend_kwargs or {}).copy()
            child_backend_kwargs["model_name"] = model
        else:
            child_backend_kwargs = self.backend_kwargs
        resolved_model = model or (child_backend_kwargs or {}).get("model_name", "unknown")

        # If we'd hit/exceed the cap, do a normal LM completion (no REPL)
        if next_depth >= self.max_depth:
            # Use other_backend if available, otherwise use main backend
            if self.other_backends and self.other_backend_kwargs:
                client = get_client(self.other_backends[0], self.other_backend_kwargs[0])
            else:
                client = get_client(self.backend, child_backend_kwargs or {})
            root_model = model or client.model_name
            start_time = time.perf_counter()
            try:
                response = client.completion(prompt)
                end_time = time.perf_counter()
                model_usage = client.get_last_usage()
                usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=response,
                    usage_summary=usage_summary,
                    execution_time=end_time - start_time,
                )
            except Exception as e:
                end_time = time.perf_counter()
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=f"Error: LM query failed at max depth - {e}",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=end_time - start_time,
                )

        # Calculate remaining budget for child (if budget tracking enabled)
        remaining_budget = None
        if self.max_budget is not None:
            remaining_budget = self.max_budget - self._cumulative_cost
            if remaining_budget <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=(
                        "Error: Budget exhausted "
                        f"(spent ${self._cumulative_cost:.6f} of ${self.max_budget:.6f})"
                    ),
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Calculate remaining timeout for child (if timeout tracking enabled)
        remaining_timeout = None
        if self.max_timeout is not None and self._completion_start_time is not None:
            elapsed = time.perf_counter() - self._completion_start_time
            remaining_timeout = self.max_timeout - elapsed
            if remaining_timeout <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=f"Error: Timeout exhausted ({elapsed:.1f}s of {self.max_timeout:.1f}s)",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Resolve the model name for callbacks
        prompt_preview = prompt[:80] if len(prompt) > 80 else prompt

        # Fire subcall start callback
        if self.on_subcall_start:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt_preview)
            except Exception:
                pass  # Don't let callback errors break execution

        subcall_start = time.perf_counter()
        error_msg: str | None = None

        # Spawn a child RLM with its own LocalREPL
        child = RLM(
            backend=self.backend,
            backend_kwargs=child_backend_kwargs,
            environment=self.environment_type,
            environment_kwargs=self.environment_kwargs,
            depth=next_depth,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_budget=remaining_budget,
            max_timeout=remaining_timeout,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            custom_system_prompt=self.system_prompt,
            other_backends=self.other_backends,
            other_backend_kwargs=self.other_backend_kwargs,
            # Give child its own logger so its trajectory is captured in metadata
            logger=RLMLogger() if self.logger else None,
            verbose=False,
            # Propagate custom tools to children (sub_tools become the child's tools)
            custom_tools=self.custom_sub_tools,
            custom_sub_tools=self.custom_sub_tools,
            # Propagate callbacks to children for nested tracking
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
        )
        try:
            result = child.completion(prompt, root_prompt=None)
            # Track child's cost in parent's cumulative cost
            if result.usage_summary and result.usage_summary.total_cost:
                self._cumulative_cost += result.usage_summary.total_cost
            return result
        except BudgetExceededError as e:
            # Propagate child's spending to parent
            self._cumulative_cost += e.spent
            error_msg = f"Budget exceeded - {e}"
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM budget exceeded - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        except Exception as e:
            error_msg = str(e)
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM completion failed - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        finally:
            # Ensure child resources are cleaned up
            child.close()
            # Fire subcall complete callback
            if self.on_subcall_complete:
                try:
                    duration = time.perf_counter() - subcall_start
                    self.on_subcall_complete(next_depth, str(resolved_model), duration, error_msg)
                except Exception:
                    pass  # Don't let callback errors break execution

    def _validate_persistent_environment_support(self) -> None:
        """Raise ValueError if the environment type does not support persistent mode.

        Currently only 'local' (LocalREPL) supports update_handler_address, add_context,
        and get_context_count.
        """
        # Known environments that support persistence
        persistent_supported_environments = {"local"}

        if self.environment_type not in persistent_supported_environments:
            raise ValueError(
                f"persistent=True is not supported for environment type '{self.environment_type}'. "
                f"Persistent mode requires environments that implement update_handler_address(), "
                f"add_context(), and get_context_count(). "
                f"Supported environments: {sorted(persistent_supported_environments)}"
            )

    @staticmethod
    def _env_supports_persistence(env: BaseEnv) -> bool:
        """Check if an environment instance supports persistent mode methods."""
        return isinstance(env, SupportsPersistence)

    def _completion_via_agent_sdk(self, prompt: str | dict[str, Any]) -> RLMChatCompletion:
        """Bypass REPL and delegate to Agent SDK sub-agents (depth=0 only).

        Args:
            prompt: User prompt (string or message dict).
        Returns:
            RLMChatCompletion with response from Agent SDK.
        Raises:
            RuntimeError: If Agent SDK execution fails.
        """
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
        from claude_agent_sdk import TextBlock as AgentTextBlock
        from claude_agent_sdk.types import HookContext, HookInput, HookJSONOutput

        from rlm.clients.vertex_patch import patch_anthropic_for_vertex

        time_start = time.perf_counter()

        # Patch Anthropic SDK to use Vertex AI
        patch_anthropic_for_vertex()

        # Accumulate tool call records: each entry is a dict with
        # tool_name, tool_input, tool_output, tool_use_id, and status.
        captured_tool_calls: list[dict[str, Any]] = []

        async def _pre_tool_hook(
            hook_input: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> HookJSONOutput:
            captured_tool_calls.append(
                {
                    "tool_name": hook_input.get("tool_name", ""),
                    "tool_input": hook_input.get("tool_input", {}),
                    "tool_output": None,
                    "tool_use_id": tool_use_id,
                    "status": "pending",
                }
            )
            return {"continue_": True}

        async def _post_tool_hook(
            hook_input: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> HookJSONOutput:
            tool_use_id_val = tool_use_id
            tool_name = hook_input.get("tool_name", "")
            tool_response = hook_input.get("tool_response")

            # For execute_in_repl (called directly or via MCP server), extract nested
            # RLM metadata from the structured dict returned by
            # create_instrumented_repl_executor().
            _repl_tool_names = {"execute_in_repl", "mcp__repl-executor__execute_in_repl"}
            nested_rlm_calls: list[Any] = []
            nested_metadata: dict[str, Any] | None = None
            if tool_name in _repl_tool_names and isinstance(tool_response, dict):
                nested_metadata = tool_response.get("metadata")
                nested_rlm_calls = tool_response.get("rlm_calls") or []

            # Update the matching pending entry with its output
            for entry in reversed(captured_tool_calls):
                if entry.get("tool_use_id") == tool_use_id_val and entry["status"] == "pending":
                    entry["tool_output"] = tool_response
                    entry["status"] = "completed"
                    if tool_name in _repl_tool_names:
                        entry["rlm_calls"] = nested_rlm_calls
                        entry["rlm_metadata"] = nested_metadata
                    break
            else:
                # No matching pre-hook entry: record as standalone completed call
                new_entry: dict[str, Any] = {
                    "tool_name": tool_name,
                    "tool_input": hook_input.get("tool_input", {}),
                    "tool_output": tool_response,
                    "tool_use_id": tool_use_id_val,
                    "status": "completed",
                }
                if tool_name in _repl_tool_names:
                    new_entry["rlm_calls"] = nested_rlm_calls
                    new_entry["rlm_metadata"] = nested_metadata
                captured_tool_calls.append(new_entry)
            return {"continue_": True}

        # Build internal hook matchers that capture every tool call
        internal_hooks: dict[str, list[HookMatcher]] = {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[_pre_tool_hook])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
        }

        # Merge with any user-provided hooks (user hooks appended after internal ones)
        merged_hooks: dict[str, list[HookMatcher]] = dict(internal_hooks)
        if self.agent_hooks:
            for event, matchers in self.agent_hooks.items():
                if event in merged_hooks:
                    merged_hooks[event] = merged_hooks[event] + list(matchers)
                else:
                    merged_hooks[event] = list(matchers)

        # Auto-promote execute_in_repl from custom_tools to an in-process MCP server so the
        # Agent SDK orchestrator can call it as a real tool (custom_tools is not visible to the
        # Agent SDK subprocess; MCP servers are the correct extension point).
        mcp_servers: dict[str, Any] = {}
        allowed_tools = list(self.agent_allowed_tools or ["Task"])
        if self.custom_tools and "execute_in_repl" in self.custom_tools:
            from rlm.utils.repl_executor import create_repl_mcp_server

            repl_mcp = create_repl_mcp_server(self.custom_tools["execute_in_repl"])
            mcp_servers["repl-executor"] = {
                "type": "sdk",
                "name": "repl-executor",
                "instance": repl_mcp._mcp_server,
            }
            # Allow the orchestrator to call execute_in_repl via the MCP server.
            # Claude Code MCP tool names are prefixed: mcp__<server>__<tool>.
            mcp_tool_name = "mcp__repl-executor__execute_in_repl"
            if mcp_tool_name not in allowed_tools:
                allowed_tools.append(mcp_tool_name)

        # Build ClaudeAgentOptions
        options = ClaudeAgentOptions(
            permission_mode=self.agent_permission_mode,
            setting_sources=["project"],
            system_prompt=self.system_prompt,
            allowed_tools=allowed_tools,
            agents=self.agent_definition,
            hooks=merged_hooks,  # type: ignore[arg-type]
            mcp_servers=mcp_servers,  # type: ignore[arg-type]
            model=self.backend_kwargs.get("model_name", "claude-opus-4-1")
            if self.backend_kwargs
            else "claude-opus-4-1",
        )

        # Convert prompt to string if dict/list
        if isinstance(prompt, dict):
            prompt_str = prompt.get("content", str(prompt))
        elif isinstance(prompt, list):
            prompt_str = "\n".join(
                msg.get("content", str(msg)) for msg in prompt if isinstance(msg, dict)
            )
        else:
            prompt_str = str(prompt)

        # Execute via Agent SDK
        response_text = ""

        async def _run_agent():
            nonlocal response_text
            try:
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(prompt=prompt_str)

                    # Stream and collect response
                    async for msg in client.receive_response():
                        if type(msg).__name__ == "AssistantMessage":
                            # Extract text from message, filtering out ThinkingBlocks
                            if hasattr(msg, "content"):
                                content = msg.content
                                if isinstance(content, list):
                                    text = "\n\n".join(
                                        block.text
                                        for block in content
                                        if isinstance(block, AgentTextBlock)
                                    )
                                else:
                                    text = str(content)
                            elif hasattr(msg, "text"):
                                text = msg.text
                            else:
                                text = str(msg)

                            response_text += text

                            if self.verbose:
                                print(text, end="", flush=True)
            except Exception as e:
                raise RuntimeError(
                    f"Claude Agent SDK query failed: {str(e)}\n\n"
                    f"Troubleshooting:\n"
                    f"  - Verify Vertex AI API is enabled\n"
                    f"  - Check IAM permissions (roles/aiplatform.user)\n"
                    f"  - Verify model availability in region\n"
                    f"  - Review agent_definition for tool/prompt errors"
                ) from e

        # Run async agent in sync context
        asyncio.run(_run_agent())

        # Store captured tool calls for downstream use (e.g., US-002 code_block conversion)
        self._last_agent_tool_calls = captured_tool_calls

        time_end = time.perf_counter()

        # Build CodeBlock entries for execute_in_repl tool calls so nested RLM
        # iterations appear under code_block.result.rlm_calls in the trajectory.
        _repl_tool_names_outer = {"execute_in_repl", "mcp__repl-executor__execute_in_repl"}
        agent_sdk_code_blocks: list[CodeBlock] = []
        for tc in captured_tool_calls:
            if tc.get("tool_name") in _repl_tool_names_outer and tc.get("status") == "completed":
                task_input = tc.get("tool_input") or {}
                task_text = (
                    task_input.get("task", str(task_input))
                    if isinstance(task_input, dict)
                    else str(task_input)
                )
                tool_output = tc.get("tool_output") or {}
                response_str = (
                    tool_output.get("response", "")
                    if isinstance(tool_output, dict)
                    else str(tool_output)
                )
                # Reconstruct sub-RLM calls as RLMChatCompletion objects for REPLResult
                raw_rlm_calls = tc.get("rlm_calls") or []
                sub_calls: list[RLMChatCompletion] = []
                for raw_call in raw_rlm_calls:
                    if isinstance(raw_call, dict):
                        try:
                            sub_calls.append(RLMChatCompletion.from_dict(raw_call))
                        except Exception:
                            pass

                repl_result = REPLResult(
                    stdout=response_str,
                    stderr="",
                    locals={},
                    execution_time=0.0,
                    rlm_calls=sub_calls,
                    final_answer=response_str,
                )
                agent_sdk_code_blocks.append(CodeBlock(code=task_text, result=repl_result))

        # Log iteration to logger if available, matching the guard at rlm.py:601
        if self.logger:
            iteration = RLMIteration(
                prompt=[{"role": "user", "content": prompt_str}],
                response=response_text,
                code_blocks=agent_sdk_code_blocks,
                final_answer=response_text,
                iteration_time=time_end - time_start,
            )
            self.logger.log(iteration)

        # Build RLMChatCompletion
        # Note: Agent SDK doesn't provide usage tracking, so we use empty summary
        usage = UsageSummary(model_usage_summaries={})

        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "claude-opus-4-1")
            if self.backend_kwargs
            else "claude-opus-4-1",
            prompt=prompt,
            response=response_text,
            usage_summary=usage,
            execution_time=time_end - time_start,
            metadata=self.logger.get_trajectory() if self.logger else None,
        )

    def close(self) -> None:
        """Clean up persistent environment. Call when done with multi-turn conversations."""
        if self._persistent_env is not None:
            if hasattr(self._persistent_env, "cleanup"):
                self._persistent_env.cleanup()
            self._persistent_env = None

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
