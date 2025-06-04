#from pydantic import BaseSettings
from pydantic-settings import BaseSettings

class BaseConfig(BaseSettings):
    """
    Base application configuration
    """

    port: int = 8080
    env: str = "DEVELOPMENT"
    debug: bool = True

    class Config:
        env_file = ".env"
