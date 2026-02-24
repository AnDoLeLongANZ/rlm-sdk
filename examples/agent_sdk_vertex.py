"""Agent SDK with Vertex AI Example

This example demonstrates using RLM with Claude Agent SDK for Vertex AI.
When agent_definition is provided with backend="vertex_anthropic", RLM
bypasses REPL execution and uses Agent SDK's multi-agent orchestration.

Prerequisites:
1. Install dependencies: uv pip install -e ".[vertex_ai,claude_agent]"
2. Set up GCP authentication: gcloud auth application-default login
3. Set environment variables in .env:
   - GOOGLE_CLOUD_PROJECT=your-project-id
   - GOOGLE_CLOUD_LOCATION=us-east5 (or your preferred region)
4. Enable Vertex AI API: gcloud services enable aiplatform.googleapis.com

Architecture:
  Load .env
      ↓
  Define agent_definition sub-agents (researcher, analyst)
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
from rlm.logger import RLMLogger
from rlm.clients.vertex_auth import verify_vertex_auth

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
    "researcher": AgentDefinition(
        description=(
            "Use this agent for web research tasks. The researcher gathers "
            "information, articles, and sources from across the internet."
        ),
        tools=["WebSearch", "Write"],
        prompt=(
            "You are a research specialist. Your role is to:\n"
            "1. Search the web for relevant, credible information\n"
            "2. Synthesize findings into clear summaries\n"
            "3. Write research notes for later analysis\n"
            "Be thorough and cite sources."
        ),
        model="sonnet"  # Sub-agent uses Sonnet
    ),
    "analyst": AgentDefinition(
        description=(
            "Use this agent for data analysis tasks. The analyst processes "
            "information and generates insights."
        ),
        tools=["Read", "Write"],
        prompt=(
            "You are a data analyst. Your role is to:\n"
            "1. Read research findings\n"
            "2. Identify key trends and patterns\n"
            "3. Generate actionable insights\n"
            "Be concise and data-driven."
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
print("\nAsking RLM to delegate to researcher and analyst agents...\n")

result = rlm.completion(
    "Research the top 3 trends in AI for 2026. "
    "Then analyze the findings and provide a brief summary."
)

print("\n" + "=" * 60)
print("RESULT:")
print("=" * 60)
print(f"\nExecution Time: {result.execution_time:.2f}s")
print(f"Model: {result.root_model}")

# Note: Agent SDK doesn't provide token usage, so this will be empty
if result.usage_summary.total_input_tokens > 0:
    print(f"Tokens: {result.usage_summary.total_input_tokens} in, "
          f"{result.usage_summary.total_output_tokens} out")
else:
    print("Token usage: Not available (Agent SDK mode)")
