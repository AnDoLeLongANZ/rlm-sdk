"""Agent SDK with Vertex AI Example

This example demonstrates using RLM with Claude Agent SDK for Vertex AI.
When agent_definition is provided with backend="vertex_anthropic", RLM
bypasses REPL execution and uses Agent SDK's multi-agent orchestration.

Architecture:
  Load .env
      ↓
  Define agent_definition sub-agents (coder, analyst)
      ↓
  Create RLM with agent_definition + backend="vertex_anthropic"
      ↓
  RLM.completion() → routes to Agent SDK (not REPL)
      ↓
  patch_anthropic_for_vertex() → AnthropicVertex client
      ↓
  ClaudeSDKClient with specialized agents
      ↓
  Vertex AI API: /models/model-name:rawPredict
"""

import os

from dotenv import load_dotenv

# Step 1: Load environment variables FIRST
load_dotenv(override=True)

# Step 2: Import RLM and verify auth
from rlm import RLM
from rlm.clients.vertex_auth import verify_vertex_auth
from rlm.logger import RLMLogger

auth_ok, auth_message = verify_vertex_auth()
if not auth_ok:
    print(f"\nError: {auth_message}")
    print("\nRequired environment variables:")
    print("  - GOOGLE_CLOUD_PROJECT: Your Google Cloud project ID")
    print("  - GOOGLE_CLOUD_LOCATION: GCP region (default: us-east5)")
    print("\nAuthentication:")
    print("  Run: gcloud auth application-default login")
    print("\nSet these in a .env file or export them in your shell.\n")
    exit(1)

print(f"Authentication: {auth_message}\n")

# Step 3: Import Claude Agent SDK for AgentDefinition
try:
    from claude_agent_sdk import AgentDefinition
except ImportError:
    print("\nError: claude-agent-sdk not installed")
    print("Install with: uv pip install -e \".[claude_agent]\"")
    print("Or: pip install claude-agent-sdk\n")
    exit(1)

# Step 4: Define specialized sub-agents using AgentDefinition
agents = {
    "coder": AgentDefinition(
        description=(
            "Use this agent for Python computation tasks. The coder writes "
            "and executes Python code to solve mathematical problems."
        ),
        tools=["Bash", "Write"],
        prompt=(
            "You are a Python computation specialist. Your role is to:\n"
            "1. Write clear, correct Python code to solve the given problem\n"
            "2. Execute the code using Bash and capture the output\n"
            "3. Write the raw results to a file called results.txt\n"
            "Show your work step by step."
        ),
        model="sonnet"  # Sub-agent uses Sonnet
    ),
    "analyst": AgentDefinition(
        description=(
            "Use this agent for result analysis tasks. The analyst reads "
            "computed output and explains the mathematical significance."
        ),
        tools=["Read", "Write"],
        prompt=(
            "You are a mathematical analyst. Your role is to:\n"
            "1. Read the computation results from results.txt\n"
            "2. Identify patterns and mathematical properties\n"
            "3. Explain the significance of the results clearly\n"
            "Be concise and precise."
        ),
        model="haiku"  # Sub-agent uses Haiku (faster)
    )
}

# Step 5: Create RLM with Agent SDK configuration
logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="vertex_anthropic",
    backend_kwargs={
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5"),
        "model_name": "sonnet",
    },
    agent_definition=agents,
    agent_allowed_tools=["Task"],
    agent_permission_mode="bypassPermissions",  # Auto-approve tool use
    logger=logger,
    verbose=True,
)

# Step 6: Use RLM - it will use Agent SDK execution (not REPL)
print("=" * 60)
print("RLM with Agent SDK Mode")
print("=" * 60)
print("\nAsking RLM to delegate to coder and analyst agents...\n")

result = rlm.completion(
    "Using Python code, compute the first 20 Fibonacci numbers and their "
    "ratios to consecutive terms. Show your work. "
    "Then analyze the results and explain the golden ratio connection."
)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"\nExecution Time: {result.execution_time:.2f}s")
print(f"Model: {result.root_model}")

# Note: Agent SDK doesn't provide token usage, so this will be empty
if result.usage_summary.total_input_tokens > 0:
    print(f"Tokens: {result.usage_summary.total_input_tokens} in, "
          f"{result.usage_summary.total_output_tokens} out")
else:
    print("Token usage: Not available (Agent SDK mode)")
