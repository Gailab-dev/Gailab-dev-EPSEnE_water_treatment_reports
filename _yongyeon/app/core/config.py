"""용연 정수장 AI API 설정 (골격).

각 정수장은 독립 서버로 운영되므로 PLANT_ID 는 이 사이트(yongyeon)로 고정된다.
설정 우선순위: 환경변수 > .env > configs/*.yaml > 아래 기본값.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError:  # pyyaml 미설치 시 YAML 로딩만 비활성
    yaml = None

from pydantic_settings import BaseSettings, SettingsConfigDict

# 사이트 루트(yongyeon/) — 이 파일은 yongyeon/app/core/config.py 에 위치.
SITE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = SITE_ROOT / "configs"

# 이 서버가 담당하는 정수장 (고정값)
PLANT_ID = "yongyeon"


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if yaml is None or not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """환경변수/.env 로 오버라이드 가능한 런타임 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    PLANT_ID: str = PLANT_ID
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8002
    DEBUG: bool = False

    # 사이트 로컬 경로 (독립 서버이므로 본인 트리 안에서 해석)
    MODEL_BASE_PATH: str = str(SITE_ROOT / "models")
    DATASET_PATH: str = str(SITE_ROOT / "dataset")

    DATABASE_URL: str = "mysql+aiomysql://root:password@localhost:3306/yongyeon"
    API_KEY: str = "change-me"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

    @property
    def site_config(self) -> dict:
        """configs/yongyeon.yaml (태그 매핑·안전 제약·모델 파라미터)."""
        return _load_yaml(f"{self.PLANT_ID}.yaml")

    @property
    def common_config(self) -> dict:
        """configs/common.yaml (DB·API·재학습 스케줄)."""
        return _load_yaml("common.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
