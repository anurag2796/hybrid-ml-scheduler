"""
Configuration Management using Pydantic Settings.

Supports environment variables and .env files for different environments.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, RedisDsn
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # Application
    app_name: str = "Hybrid ML Scheduler"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Database
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="Coloreal@1", env="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="hybrid_scheduler_db", env="POSTGRES_DB")
    
    @property
    def database_url(self) -> str:
        """Construct database URL."""
        from urllib.parse import quote_plus
        password = quote_plus(self.postgres_password)
        return f"postgresql://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async database URL for asyncpg."""
        from urllib.parse import quote_plus
        password = quote_plus(self.postgres_password)
        return f"postgresql+asyncpg://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    
    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Cache
    cache_ttl: int = 300  # 5 minutes default TTL
    recent_metrics_cache_size: int = 100
    
    # Simulation
    num_gpus: int = 4
    retrain_interval: int = 50
    batch_size: int = 50
    max_history: int = 1000
    
    # WebSocket
    websocket_heartbeat_interval: int = 30  # seconds
    websocket_message_queue_size: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Rate Limiting
    rate_limit_per_minute: int = 60


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export singleton instance
settings = get_settings()
