"""REPL Execution Tool for Agent SDK.

Factory for creating callables that delegate tasks from an Agent SDK orchestrator
to a REPL-mode RLM instance.
"""

import time
from typing import Any

from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.logger import RLMLogger


def create_repl_executor(
    backend: str,
    backend_kwargs: dict[str, Any],
    logger: RLMLogger | None = None,
    verbose: bool = False,
    max_depth: int = 2,
    max_iterations: int = 30,
    max_timeout: float | None = None,
    max_budget: float | None = None,
    verification_suffix: str | None = None,
):
    """Create and return a callable that runs a task in a REPL-mode RLM instance.

    The returned callable returns a full RLMChatCompletion object. When injected
    into a LocalREPL via custom_tools, the LocalREPL wrapper intercepts the
    RLMChatCompletion, appends it to _pending_llm_calls (so it appears in the
    parent code block's rlm_calls metadata), and returns the plain response string
    to the REPL code.
    """

    def execute_computation(task: str) -> RLMChatCompletion:
        """Execute a computational task in a new REPL-mode RLM instance.

        Args:
            task: Computational task prompt to run in REPL mode

        Returns:
            RLMChatCompletion with the final answer, usage metadata, and full
            trajectory (when a logger is configured). On failure, returns an
            RLMChatCompletion with a formatted error message as the response.
        """
        start_time = time.perf_counter()
        root_model = (backend_kwargs or {}).get("model_name", backend)

        try:
            # Import RLM here to avoid circular imports
            from rlm import RLM

            # Append verification suffix to guarantee rlm_query() is called
            effective_task = task
            if verification_suffix:
                effective_task = f"{task}\n\n{verification_suffix}"

            # Create a separate logger for the sub-RLM so its clear_iterations()
            # call does not wipe the parent logger's in-progress iterations.
            # If the parent provided a log_dir, write sub-RLM iterations to the
            # same directory (separate file) so the JSONL is visible in the
            # visualizer. The sub-RLM's trajectory is returned as part of the
            # RLMChatCompletion.metadata and surfaced in the parent code block's
            # rlm_calls via _wrap_custom_tool.
            from rlm.logger import RLMLogger as _RLMLogger

            sub_logger = _RLMLogger(log_dir=logger.log_dir) if logger else None

            # Create sub-RLM with REPL mode forced
            sub_rlm = RLM(
                backend=backend,
                backend_kwargs=backend_kwargs,
                environment="local",  # Force REPL mode
                max_depth=max_depth,
                max_iterations=max_iterations,
                max_timeout=max_timeout,
                max_budget=max_budget,
                logger=sub_logger,
                verbose=verbose,
            )

            # Execute computation in REPL mode
            result = sub_rlm.completion(effective_task)

            # Clean up resources
            sub_rlm.close()

            # Return the full RLMChatCompletion so LocalREPL can append it to
            # _pending_llm_calls, making the sub-RLM trajectory visible in the
            # parent code block's rlm_calls metadata.
            return result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            error_response = f"Error during REPL execution [{error_type}]: {error_msg}"
            return RLMChatCompletion(
                root_model=root_model,
                prompt=task,
                response=error_response,
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - start_time,
            )

    # Add metadata to the tool for introspection
    execute_computation.__doc__ = (
        "Execute a computational task in REPL mode with full iteration logging. "
        "Use this tool for tasks requiring Python code execution, recursion, or "
        "step-by-step computation tracking."
    )

    return execute_computation


def create_instrumented_repl_executor(
    backend: str,
    backend_kwargs: dict[str, Any],
    logger: RLMLogger | None = None,
    verbose: bool = False,
    max_depth: int = 2,
    max_iterations: int = 30,
    max_timeout: float | None = None,
    max_budget: float | None = None,
    verification_suffix: str | None = None,
):
    """Create a wrapped REPL executor that returns a structured dict for Agent SDK hooks.

    Wraps create_repl_executor() so that when execute_in_repl is called by an Agent
    SDK orchestrator, the PostToolUse hook can extract nested RLM metadata (iterations,
    sub-calls) from the tool response and store them as code_block.result.rlm_calls.
    """
    repl_tool = create_repl_executor(
        backend=backend,
        backend_kwargs=backend_kwargs,
        logger=logger,
        verbose=verbose,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=max_timeout,
        max_budget=max_budget,
        verification_suffix=verification_suffix,
    )

    def instrumented_execute(task: str) -> dict[str, Any]:
        """Execute a REPL task and return structured metadata for Agent SDK hooks.

        Args:
            task: Computational task prompt to run in REPL mode

        Returns:
            Dict with response, metadata, and rlm_calls for downstream extraction.
        """
        result = repl_tool(task)

        rlm_calls: list[dict[str, Any]] = []
        metadata = result.metadata

        if metadata:
            for iteration in metadata.get("iterations", []):
                for code_block in iteration.get("code_blocks", []):
                    repl_result = code_block.get("result", {})
                    for sub_call in repl_result.get("rlm_calls", []):
                        rlm_calls.append(sub_call)

        return {
            "response": result.response,
            "metadata": metadata,
            "rlm_calls": rlm_calls,
        }

    instrumented_execute.__doc__ = (
        "Execute a computational task in REPL mode with full iteration logging. "
        "Use this tool for tasks requiring Python code execution, recursion, or "
        "step-by-step computation tracking. Returns structured metadata for "
        "Agent SDK hook extraction."
    )

    return instrumented_execute


def create_repl_mcp_server(executor_callable: Any) -> Any:
    """Wrap a REPL executor callable in a FastMCP in-process server.

    Creates an MCP server with a single ``execute_in_repl`` tool backed by the
    provided callable.  The server's underlying ``mcp.server.Server`` instance
    can be passed directly to ``ClaudeAgentOptions.mcp_servers`` as a
    ``McpSdkServerConfig`` so the Agent SDK orchestrator can call the tool.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as err:
        raise ImportError(
            "mcp package is required to create an MCP server for execute_in_repl. "
            "Install with: pip install mcp"
        ) from err

    mcp = FastMCP("repl-executor")

    @mcp.tool(name="execute_in_repl")
    def _execute_in_repl(task: str) -> dict[str, Any]:
        """Execute a computational task in REPL mode with full iteration logging.

        Use this tool for tasks requiring Python code execution, recursion, or
        step-by-step computation tracking. Each call creates an isolated REPL
        environment and returns the final computed answer.
        """
        result = executor_callable(task)
        if isinstance(result, dict):
            return result
        return {"response": str(result), "metadata": None, "rlm_calls": []}

    return mcp


def get_repl_tool_metadata() -> dict[str, Any]:
    """Get metadata about the REPL executor tool for documentation.

    Returns:
        Dict containing tool metadata (name, description, use cases, constraints)
    """
    return {
        "name": "execute_in_repl",
        "description": (
            "Execute computational tasks in REPL mode with full iteration logging. "
            "This tool creates a Python REPL environment where code can be executed "
            "step-by-step with visible iterations."
        ),
        "use_cases": [
            "Mathematical computations requiring Python code",
            "Recursive algorithms (fibonacci, tree traversal, etc.)",
            "Step-by-step calculations with intermediate results",
            "Tasks requiring code execution and variable tracking",
        ],
        "constraints": [
            "Limited to max_depth recursion levels (default: 2)",
            "Subject to max_iterations limit (default: 30)",
            "Requires parent RLM to have logger for trajectory tracking",
            "Authentication context inherited from parent backend_kwargs",
        ],
        "returns": "Final answer string from REPL execution or error message",
    }
