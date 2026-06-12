from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project configuration loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Providers ---
    # Two providers split by budget: OpenAI ($30) for GPT models, OpenRouter ($70)
    # for everything else. Routing is by model id: ids WITHOUT a "/" go to OpenAI
    # (e.g. "gpt-4.1-nano"); ids WITH a "/" go to OpenRouter (e.g.
    # "google/gemini-2.5-flash-lite"). At least one key must be set.
    openrouter_api_key: str = Field("", description="OpenRouter API key ($70 budget)")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = "paper5-mcts"
    openrouter_site_url: str = ""

    openai_api_key: str = Field("", description="OpenAI API key ($30 budget)")
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Models per role (see EXPERIMENTS.md §3.7) ---
    # No "/" -> OpenAI direct; with "/" -> OpenRouter.
    llm_dev: str = "gpt-4.1-nano"                       # OpenAI, weak end of alpha axis
    llm_cheap: str = "gpt-4.1-nano"                     # OpenAI
    llm_baseline: str = "google/gemini-2.5-flash-lite"  # OpenRouter
    llm_judge: str = "anthropic/claude-haiku-4.5"       # OpenRouter, external judge (RQ3)
    llm_premium: str = "anthropic/claude-sonnet-4.6"     # OpenRouter, headline strong end
    embed_model: str = "BAAI/bge-large-en-v1.5"

    # --- Inference defaults ---
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 60

    # --- Caching ---
    cache_dir: Path = Path(".cache/llm")
    cache_enabled: bool = True

    # --- Paths ---
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    cwe_tree_path: Path = Path("data/cwe_tree.json")

    # --- External APIs ---
    mitre_cwe_api: str = "https://cwe-api.mitre.org/api/v1"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def provider_for(model: str) -> str:
        """Route a model id to a provider: '/' in id -> openrouter, else openai."""
        return "openrouter" if "/" in model else "openai"

    def openrouter_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name
        return headers


settings = Settings()
