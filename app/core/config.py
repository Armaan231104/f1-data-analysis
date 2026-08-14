from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "F1 Data Analysis Platform"
    api_version: str = "v1"
    database_url: str = "sqlite:///./data/f1.db"
    fastf1_cache_dir: str = "data/raw/fastf1_cache"
    model_artifact_path: str = "ml/artifacts/model.pt"
    api_keys_seed: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
