import os
import sys

from anthropic import AnthropicVertex

_original_anthropic_imported = False
_vertex_client = None


def patch_anthropic_for_vertex():
    global _vertex_client

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT must be set before patching Anthropic for Vertex AI")

    _vertex_client = AnthropicVertex(project_id=project_id, region=location)

    import anthropic

    class AnthropicVertexProxy:
        def __new__(cls, *args, **kwargs):
            return _vertex_client

        def __init__(self, *args, **kwargs):
            pass

    anthropic.Anthropic = AnthropicVertexProxy
    anthropic.AsyncAnthropic = AnthropicVertexProxy

    if "claude_agent_sdk" in sys.modules:
        import claude_agent_sdk

        if hasattr(claude_agent_sdk, "anthropic"):
            claude_agent_sdk.anthropic.Anthropic = AnthropicVertexProxy
            claude_agent_sdk.anthropic.AsyncAnthropic = AnthropicVertexProxy

    return _vertex_client


def get_vertex_client():
    return _vertex_client
