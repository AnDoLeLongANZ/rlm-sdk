"""Agent SDK + REPL Hybrid Example.

Agent SDK orchestration (depth=0) delegates to REPL sub-agents (depth=1+).
Prerequisites: uv pip install -e ".[claude_agent]", set GOOGLE_CLOUD_PROJECT /
GOOGLE_CLOUD_LOCATION, run gcloud auth application-default login.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

from rlm import RLM  # noqa: E402
from rlm.clients.vertex_auth import verify_vertex_auth  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402

# Agent SDK tool calls (execute_in_repl) are captured via PreToolUse/PostToolUse callback
# hooks registered inside _completion_via_agent_sdk(). The instrumented executor returns a
# structured dict {"response", "metadata", "rlm_calls"} so the PostToolUse hook can extract
# nested RLM sub-call trajectories and reconstruct CodeBlock entries for trajectory logging.
try:
    from rlm.utils.repl_executor import (
        create_instrumented_repl_executor as _create_executor,  # noqa: E402
    )

    _using_instrumented = True
except ImportError:
    from rlm.utils.repl_executor import create_repl_executor as _create_executor  # noqa: E402

    _using_instrumented = False

auth_ok, auth_message = verify_vertex_auth()
if not auth_ok:
    print(f"\nError: {auth_message}")
    print(
        "Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION and run: gcloud auth application-default login\n"
    )
    exit(1)

print(f"Authentication: {auth_message}\n")

try:
    from claude_agent_sdk import AgentDefinition
except ImportError:
    print('\nError: claude-agent-sdk not installed. Run: uv pip install -e ".[claude_agent]"\n')
    exit(1)

print("=" * 60)
print("Multi-Series Mathematical Analysis - Agent SDK + REPL Hybrid")
print("=" * 60)
print()

logger = RLMLogger(log_dir="./logs")

# The Agent SDK uses ANTHROPIC_VERTEX_PROJECT_ID for its Vertex AI project.
# The sub-RLM (execute_in_repl) uses the same project and a versioned model name.
# ANTHROPIC_VERTEX_PROJECT_ID takes precedence over GOOGLE_CLOUD_PROJECT so both
# the Agent SDK orchestrator and the sub-RLM call the same Vertex AI endpoint.
project_id = os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")
# Vertex AI requires a versioned model ID; "claude-sonnet-4-5@20250929" is the
# stable Sonnet 4.5 endpoint. Override via REPL_MODEL_NAME env var if needed.
repl_model_name = os.getenv("REPL_MODEL_NAME", "claude-sonnet-4-5@20250929")
backend_kwargs = {"project_id": project_id, "location": location, "model_name": repl_model_name}

repl_tool = _create_executor(
    backend="vertex_anthropic",
    backend_kwargs=backend_kwargs,
    logger=logger,
    verbose=True,
    max_depth=3,
    max_iterations=30,
    verification_suffix=(
        "After computing the result, use rlm_query() to verify the last value "
        "in the sequence by asking a sub-LM to confirm it independently. "
        "Store the sub-LM response and compare it to your computed value. "
        "Report whether the sub-LM answer matches."
    ),
)

agents = {
    "orchestrator": AgentDefinition(
        description="Coordinate multi-series mathematical workflows: plan, delegate each series to REPL, synthesize.",
        tools=["Task"],
        prompt=(
            "You are a workflow orchestrator. Plan your approach for all three series first, "
            "then delegate each to execute_in_repl as a separate call (one per series). "
            "Each call must include: 'Use rlm_query() to verify the last value independently.' "
            "Synthesize a comparative summary after all three series are computed."
        ),
        model="sonnet",
    ),
    "coder": AgentDefinition(
        description="Write and execute Python for mathematical series (Fibonacci, primes, perfect squares).",
        tools=["Bash", "Write", "Read"],
        prompt=(
            "You are a Python code specialist. Write and execute code for the assigned series, "
            "use rlm_query() to verify the last value independently, and report any discrepancy."
        ),
        model="sonnet",
    ),
    "analyst": AgentDefinition(
        description="Analyze and compare results across Fibonacci, primes, and perfect squares series.",
        tools=["Read", "Write"],
        prompt=(
            "You are a mathematical analyst. Read results from all three series, "
            "identify patterns within each, and compare cross-series relationships."
        ),
        model="haiku",
    ),
}

rlm = RLM(
    backend="vertex_anthropic",
    backend_kwargs=backend_kwargs,
    agent_definition=agents,
    agent_allowed_tools=["Task"],
    agent_permission_mode="bypassPermissions",
    custom_tools={"execute_in_repl": repl_tool},
    logger=logger,
    verbose=True,
)

_executor_mode = (
    "instrumented (structured dict output)"
    if _using_instrumented
    else "standard (RLMChatCompletion output)"
)

print("Configuration:")
print(f"  - Project: {project_id}, Region: {location}")
print("  - Agents: orchestrator (Sonnet), coder (Sonnet), analyst (Haiku)")
print(
    f"  - REPL executor: max_depth=3, max_iterations=30, mode={_executor_mode}, model={repl_model_name}"
)
print(f"  - Logger: {logger.log_file_path}")
print()


def print_separator(char="─", width=60):
    print(char * width)


def print_metadata_tree(result, depth=0):
    """Recursively print metadata from an RLMChatCompletion and its sub-calls."""
    indent = "  " * depth
    prefix = f"{'└─ ' if depth > 0 else ''}"

    print(
        f"{indent}{prefix}[Depth {depth}] model={result.root_model}  "
        f"time={result.execution_time:.2f}s  response_len={len(result.response)}"
    )

    # Print usage
    usage = result.usage_summary
    if usage and usage.model_usage_summaries:
        for _model, summary in usage.model_usage_summaries.items():
            print(
                f"{indent}   tokens: in={summary.total_input_tokens} out={summary.total_output_tokens} "
                f"calls={summary.total_calls}"
                + (f" cost=${summary.total_cost:.6f}" if summary.total_cost else "")
            )

    # Print trajectory metadata
    if result.metadata:
        traj = result.metadata
        n_iters = len(traj.get("iterations", []))
        print(f"{indent}   metadata: {n_iters} iteration(s) captured")

        # Dig into iterations to find sub-calls with their own metadata
        for i, iteration in enumerate(traj.get("iterations", [])):
            for cb in iteration.get("code_blocks", []):
                repl_result = cb.get("result", {})
                for j, sub_call in enumerate(repl_result.get("rlm_calls", [])):
                    sub_response = sub_call.get("response", "")[:80]
                    print(
                        f"{indent}   iter {i + 1} sub-call {j + 1}: "
                        f"model={sub_call.get('root_model', '?')}  "
                        f"response={sub_response!r}..."
                    )
                    if sub_call.get("metadata"):
                        sub_n = len(sub_call["metadata"].get("iterations", []))
                        print(f"{indent}     ^ has nested metadata: {sub_n} iteration(s)")
    else:
        print(f"{indent}   metadata: (none)")

    print()


print_separator("=")
print("EXECUTION START")
print_separator("=")

task = (
    "Analyze 3 mathematical series. First, outline your plan, then execute each series "
    "separately using execute_in_repl (one call per series). Each call MUST include: "
    "'Use rlm_query() to double-check the last value by asking a sub-LM to verify it.'\n"
    "(a) Fibonacci: first 20 numbers, analyze convergence to golden ratio (~1.618).\n"
    "(b) Primes: first 20 primes, analyze distribution gaps between consecutive primes.\n"
    "(c) Perfect squares: first 20, analyze differences between consecutive squares.\n"
    "After all three, synthesize a comparative summary highlighting patterns across series."
)

print("Task: Compute 3 mathematical series (Fibonacci, Primes, Perfect Squares)\n")

result = rlm.completion(task)

# Debug summary: verify metadata structure and print iteration/code_block/rlm_call counts.
print()
print_separator("=")
print("DEBUG SUMMARY")
print_separator("=")
print()
assert result.metadata is not None, "Expected result.metadata to be set after Agent SDK execution"
iterations = result.metadata.get("iterations", [])
assert len(iterations) >= 1, f"Expected at least 1 iteration in metadata, got {len(iterations)}"
print(f"Metadata verified: {len(iterations)} iteration(s) captured")
for i, iteration in enumerate(iterations):
    code_blocks = iteration.get("code_blocks", [])
    print(f"  Iteration {i + 1}: {len(code_blocks)} code_block(s)")
    for j, cb in enumerate(code_blocks):
        repl_result = cb.get("result", {})
        rlm_calls = repl_result.get("rlm_calls", [])
        print(f"    code_block {j + 1}: {len(rlm_calls)} rlm_call(s)")
print()

print()
print_separator("=")
print("EXECUTION COMPLETE")
print_separator("=")
print(f"\nExecution Time: {result.execution_time:.2f}s  Root Model: {result.root_model}")

if result.usage_summary.total_input_tokens > 0:
    tokens_in = result.usage_summary.total_input_tokens
    tokens_out = result.usage_summary.total_output_tokens
    cost = result.usage_summary.total_cost
    cost_str = f"  Cost: ${cost:.4f}" if cost > 0 else ""
    print(f"Tokens: {tokens_in} in, {tokens_out} out{cost_str}")

print()
print_separator("=")
print("FINAL ANSWER")
print_separator("=")
print(f"\n{result.response}\n")

print_separator("=")
print("METADATA TREE")
print_separator("=")
print()
print_metadata_tree(result, depth=0)

print_separator("=")
print("SUB-CALL METADATA DETAIL")
print_separator("=")
print()

if result.metadata:
    found_subcalls = False
    for i, iteration in enumerate(result.metadata.get("iterations", [])):
        for cb in iteration.get("code_blocks", []):
            repl_result_meta = cb.get("result", {})
            for j, sub_call in enumerate(repl_result_meta.get("rlm_calls", [])):
                found_subcalls = True
                print(f"Iteration {i + 1}, Sub-call {j + 1}:")
                print(f"  Model: {sub_call.get('root_model', '?')}")
                print(f"  Response: {sub_call.get('response', '')[:200]}")
                print(f"  Execution time: {sub_call.get('execution_time', 0):.2f}s")
                if sub_call.get("metadata"):
                    meta = sub_call["metadata"]
                    n_nested = len(meta.get("iterations", []))
                    print(f"  Depth: {meta.get('run_metadata', {}).get('depth', '?')}")
                    print(f"  Nested iterations: {n_nested}")
                else:
                    print("  Trajectory: (none - leaf LM call, no REPL)")
                print()
    if not found_subcalls:
        print("No rlm_calls entries found in metadata iterations.")
else:
    print("No metadata to inspect.\n")

print_separator("=")
print("TRAJECTORY LOGGING")
print_separator("=")
print(f"\nLog file: {logger.log_file_path}")
print("Load in visualizer: cd visualizer && npm run dev\n")
