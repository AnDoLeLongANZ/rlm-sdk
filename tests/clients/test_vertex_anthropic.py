from rlm.clients.vertex_anthropic import VertexAnthropicClient


def test_vertex_client_initialization():
    """Test VertexAnthropicClient can be instantiated with valid params."""
    client = VertexAnthropicClient(
        project_id="test-project",
        location="us-east5",
        model_name="claude-opus-4-1"
    )
    assert client.project_id == "test-project"
    assert client.location == "us-east5"
    assert client.model_name == "claude-opus-4-1"
