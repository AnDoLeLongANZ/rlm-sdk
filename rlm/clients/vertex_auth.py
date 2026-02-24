import os
from typing import Optional
from anthropic import AnthropicVertex


def create_vertex_client(
    project_id: Optional[str] = None,
    location: Optional[str] = None
) -> AnthropicVertex:
    project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

    if not project_id:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT must be set in environment. "
            "Set it in .env file or export in your shell."
        )

    client = AnthropicVertex(
        project_id=project_id,
        region=location
    )

    return client


def verify_vertex_auth() -> tuple[bool, str]:
    try:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")

        if not project_id:
            return False, "GOOGLE_CLOUD_PROJECT not found in environment"

        client = create_vertex_client(project_id, location)
        return True, f"Vertex AI configured for project: {project_id} in {location}"

    except Exception as e:
        return False, f"Vertex AI authentication failed: {str(e)}"
