from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "F1 Data Analysis Platform"
    api_version: str = "v1"
    api_key: str = "dev-api-key"
    database_url: str = "postgresql+psycopg://f1:f1@db:5432/f1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
