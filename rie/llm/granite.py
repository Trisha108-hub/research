from uuid import UUID

import requests

from rie.config import Settings, get_settings

class GraniteClient:
    """IBM WatsonX.ai text-generation client with SDK, REST, and offline modes."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        if not self.settings.watsonx_api_key:
            return self._offline_response(prompt)
        if not self.settings.watsonx_project_id.strip():
            raise RuntimeError(
                "WatsonX is enabled, but WATSONX_PROJECT_ID is empty. Add the project ID "
                "from IBM Cloud to your .env file, or clear WATSONX_API_KEY to use offline mode."
            )
        project_id = self.settings.watsonx_project_id.strip()
        try:
            parsed_project_id = UUID(project_id)
        except ValueError as exc:
            raise RuntimeError(
                "WATSONX_PROJECT_ID must be the project's UUIDv4 from IBM Cloud. "
                "It is not the WatsonX API key or an identifier beginning with 'ApiKey-'."
            ) from exc
        if parsed_project_id.version != 4 or str(parsed_project_id) != project_id.lower():
            raise RuntimeError(
                "WATSONX_PROJECT_ID must be a canonical UUIDv4 from the IBM Cloud project "
                "Manage page. Replace the configured value, then restart Streamlit."
            )
        return self._generate_rest(prompt, max_tokens)

    def _generate_rest(self, prompt: str, max_tokens: int) -> str:
        try:
            token_response = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self.settings.watsonx_api_key,
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            url = f"{self.settings.watsonx_url.rstrip('/')}/ml/v1/text/chat?version=2024-10-08"
            payload = {
                "model_id": self.settings.watsonx_model_id,
                "messages": [{"role": "user", "content": prompt}],
                "parameters": {"max_new_tokens": max_tokens, "temperature": 0.2},
                "project_id": self.settings.watsonx_project_id.strip(),
            }
            result = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": "Bearer " + access_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=60,
            )
            if not result.ok:
                raise RuntimeError(self._format_http_error(result.status_code, result.text))
            body = result.json()
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            generated = message.get("content")
            if not generated:
                finish_reason = choice.get("finish_reason")
                detail = f" Finish reason: {finish_reason}." if finish_reason else ""
                raise RuntimeError(
                    "WatsonX returned no chat content. Check that the selected model supports "
                    f"the chat API and try again.{detail}"
                )
            return generated.strip()
        except requests.Timeout as exc:
            raise RuntimeError("WatsonX request timed out. Check your connection and try again.") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError("Could not connect to WatsonX. Check the service URL and your internet connection.") from exc
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            raise self._friendly_error(exc) from exc

    @staticmethod
    def _format_http_error(status_code: int, body: str) -> str:
        detail = body[:1000]
        if "no_associated_service_instance_error" in body:
            return (
                "The IBM Cloud project exists, but it is not associated with a Watson Machine "
                "Learning (WML) service instance. In IBM Cloud, open the project settings and "
                "associate or attach an active WML service instance in the same account and "
                "region, then retry. Keep this project's UUID as WATSONX_PROJECT_ID. "
                f"Provider details: {detail}"
            )
        if status_code == 404 and "container_not_found" in body:
            return (
                "WatsonX could not find the configured project. WATSONX_PROJECT_ID must be an "
                "existing project ID in the IBM Cloud account that owns this API key. Copy the "
                "ID from that project's Manage page, update .env, then restart Streamlit. "
                f"Provider details: {detail}"
            )
        if status_code == 400 and "project_id should be a version 4 uuid" in body.lower():
            return (
                "WatsonX rejected WATSONX_PROJECT_ID because it is not a project UUIDv4. "
                "Copy the project ID from the IBM Cloud project's Manage page. Do not use "
                "the API key or a value prefixed with 'ApiKey-'. "
                f"Provider details: {detail}"
            )
        if status_code in (401, 403):
            return f"WatsonX rejected the API key or access permissions (HTTP {status_code}). Verify .env. Details: {detail}"
        if status_code == 404:
            return f"WatsonX endpoint or model was not found (HTTP 404). Check WATSONX_URL and WATSONX_MODEL_ID. Details: {detail}"
        if status_code == 429:
            return f"WatsonX rate limit reached (HTTP 429). Wait a moment and try again. Details: {detail}"
        return f"WatsonX request failed (HTTP {status_code}). Details: {detail}"

    @classmethod
    def _friendly_error(cls, exc: Exception) -> RuntimeError:
        message = str(exc)
        if "no_associated_service_instance_error" in message:
            return RuntimeError(cls._format_http_error(403, message))
        if "container_not_found" in message or "project_id" in message and "not found" in message.lower():
            return RuntimeError(cls._format_http_error(404, message))
        if isinstance(exc, requests.Timeout):
            return RuntimeError("WatsonX request timed out. Check your connection and try again.")
        if isinstance(exc, requests.ConnectionError):
            return RuntimeError("Could not connect to WatsonX. Check the service URL and your internet connection.")
        return RuntimeError(f"WatsonX generation failed: {message[:1000]}")

    @staticmethod
    def _offline_response(prompt: str) -> str:
        if prompt.startswith("Break this research question"):
            question = prompt.split("Question:", 1)[-1].strip()
            return "\n".join([question, f"{question} methods", f"{question} evidence"])
        if prompt.startswith("Extract evidence"):
            return (
                "Claim: The retrieved evidence is relevant to the research question.\n"
                "Summary: the source provides background context.\n"
                "Confidence: 0.50\n"
                "Stance: neutral"
            )
        if prompt.startswith("Write an evidence-grounded synthesis"):
            return (
                "No WatsonX credentials are configured. The structured sections below are "
                "generated from retrieved source metadata; configure WatsonX for narrative synthesis."
            )
        return (
            "## Executive Summary\n\nNo WatsonX credentials are configured. "
            "The report is based on retrieved source metadata only.\n\n"
            "## Findings\n\nSee the evidence section for source details.\n\n"
            "## Limitations\n\nNo language model synthesis was performed.\n\n"
            "## Conclusion\n\nConfigure WatsonX credentials for Granite-powered synthesis."
        )
