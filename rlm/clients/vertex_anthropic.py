import asyncio
import os
from pathlib import Path
from typing import Any

from anthropic import AnthropicVertex

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class VertexAnthropicClient(BaseLM):
    """
    Anthropic client using Vertex AI authentication.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model_name: str | None = None,
        timeout: float = 300.0,
        **kwargs,
    ):
        """Initialize Vertex AI client.

        Args:
            project_id: GCP project ID (from env: GOOGLE_CLOUD_PROJECT)
            location: GCP region (from env: GOOGLE_CLOUD_LOCATION, default: us-east5)
            model_name: Claude model name (e.g., "claude-opus-4-1")
            timeout: Request timeout in seconds
            **kwargs: Additional client kwargs

        Raises:
            ValueError: If project_id missing or credentials invalid
        """
        super().__init__(model_name=model_name or "claude-opus-4-1", timeout=timeout, **kwargs)

        # Load from environment with explicit precedence
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")
        self.model_name = model_name or "claude-opus-4-1"

        # Fail-fast validation (no silent fallbacks)
        if not self.project_id:
            raise ValueError(
                "GCP project ID required. Provide via:\n"
                "  1. project_id= constructor argument\n"
                "  2. GOOGLE_CLOUD_PROJECT environment variable\n"
                "  3. gcloud auth login --update-adc\n"
                "See: https://cloud.google.com/docs/authentication"
            )

        # Validate credentials file if specified
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not Path(creds_path).exists():
                raise ValueError(
                    f"GOOGLE_APPLICATION_CREDENTIALS points to non-existent file:\n"
                    f"  {creds_path}\n"
                    f"Solutions:\n"
                    f"  1. Verify file path: ls -la {creds_path}\n"
                    f"  2. Generate new key via gcloud iam service-accounts keys create\n"
                    f"  3. Use ADC instead: gcloud auth login --update-adc && unset GOOGLE_APPLICATION_CREDENTIALS"
                )

        # Create authenticated client
        try:
            self.client = AnthropicVertex(project_id=self.project_id, region=self.location)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Vertex AI client:\n{str(e)}\n\n"
                f"Ensure GCP credentials are available via:\n"
                f"  • Application Default Credentials (gcloud auth login --update-adc)\n"
                f"  • Service account key file (GOOGLE_APPLICATION_CREDENTIALS)\n"
                f"  • Workload Identity (if running on GCP)\n"
                f"Debug: gcloud auth list && gcloud config list"
            ) from e

        # Initialize usage tracking (per existing pattern)
        from collections import defaultdict

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Prepare messages and extract system prompt for Anthropic API.

        Anthropic requires system messages separate from conversation messages.

        Args:
            prompt: String or message list

        Returns:
            (messages list, system message or None)
        """
        system = None

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    system = msg.get("content")
                else:
                    messages.append(msg)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        return messages, system

    def _track_usage(self, response: Any, model: str):
        """Track usage from Vertex AI response.

        Args:
            response: Response object from AnthropicVertex
            model: Model name used
        """
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += response.usage.input_tokens
        self.model_output_tokens[model] += response.usage.output_tokens

        self.last_prompt_tokens = response.usage.input_tokens
        self.last_completion_tokens = response.usage.output_tokens

    def completion(self, prompt: str | dict[str, Any]) -> str:
        """Synchronous completion via Vertex AI.

        Args:
            prompt: String prompt or message list

        Returns:
            Response text from Claude

        Raises:
            ValueError: If API call fails
        """
        messages, system = self._prepare_messages(prompt)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                system=system,
                max_tokens=32768,
                timeout=self.timeout,
            )
        except Exception as e:
            raise ValueError(
                f"Vertex AI API call failed:\n{str(e)}\n\n"
                f"Troubleshooting:\n"
                f"  • Verify Vertex AI API is enabled: gcloud services list --enabled | grep aiplatform\n"
                f"  • Check IAM permissions: roles/aiplatform.user required\n"
                f"  • Verify model availability in region: {self.location}\n"
                f"  • Check network connectivity and firewall rules"
            ) from e

        self._track_usage(response, self.model_name)

        return "\n\n".join(block.text for block in response.content if block.type == "text")

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        """Async completion via Vertex AI.

        Note: AnthropicVertex doesn't support native async, so we wrap
        synchronous call with asyncio.to_thread to avoid blocking.

        Args:
            prompt: String prompt or message list dict

        Returns:
            Response text from Claude via Vertex AI
        """
        return await asyncio.to_thread(self.completion, prompt)

    def get_usage_summary(self) -> UsageSummary:
        """Aggregate usage across all models called by this client.

        Returns:
            UsageSummary with per-model token counts and total_cost=None
            (Vertex AI doesn't provide cost information)
        """
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=None,
            )

        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Track the last call's tokens/cost.

        Returns:
            ModelUsageSummary for the most recent API call with total_cost=None
            (Vertex AI doesn't provide cost information)
        """
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=None,
        )
