"""덕남 정수장 AI API 설정 (골격).

각 정수장은 독립 서버로 운영되므로 PLANT_ID 는 이 사이트(deoknam)로 고정된다.
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

# 사이트 루트(deoknam/) — 이 파일은 deoknam/app/core/config.py 에 위치.
SITE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = SITE_ROOT / "configs"

# 이 서버가 담당하는 정수장 (고정값)
PLANT_ID = "deoknam"


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
    SERVER_PORT: int = 8001
    DEBUG: bool = False

    # 사이트 로컬 경로 (독립 서버이므로 본인 트리 안에서 해석)
    MODEL_BASE_PATH: str = str(SITE_ROOT / "models")
    DATASET_PATH: str = str(SITE_ROOT / "dataset")

    DATABASE_URL: str = "mysql+aiomysql://root:password@localhost:3306/deoknam"
    API_KEY: str = "change-me"

    # 군집 실모델용 step_aiwtp DB — 값은 환경변수 WTP_DB_* 또는 .env로만 주입.
    # 소스에 자격증명을 두지 말 것(미설정 시 API는 mock 폴백). .env.example 참고.
    WTP_DB_HOST: str = ""
    WTP_DB_PORT: int = 44306
    WTP_DB_USER: str = ""
    WTP_DB_PASSWORD: str = ""
    WTP_DB_NAME: str = ""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

    @property
    def site_config(self) -> dict:
        """configs/deoknam.yaml (태그 매핑·안전 제약·모델 파라미터)."""
        return _load_yaml(f"{self.PLANT_ID}.yaml")

    @property
    def common_config(self) -> dict:
        """configs/common.yaml (DB·API·재학습 스케줄)."""
        return _load_yaml("common.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# --- 군집 실모델 연동 상수 (덕남) ---
_MODEL_ROOT = SITE_ROOT / "덕남 모델 개발" / "01_clustering" / "output"
CLUSTER_MODEL_PATH = _MODEL_ROOT / "model" / "덕남_kmeans_k3.joblib"
CLUSTER_CENTERS_PATH = _MODEL_ROOT / "cluster_centers.csv"

MEASR_TABLE = "WTP_WIDE_MEASR_DK"      # 실측 원수성상(1분)
SMLT_TABLE = "WTP_SMLT_HSTRY"          # 알칼리도·전기전도도(현재 0행 — 채워지면 우선 사용)
LCLGV_CD = "2915"                       # 덕남 지자체코드

# DB 계측코드 → 군집 raw_cols(학습 컬럼명)
DB2RAW = {
    "INT_TRBD": "RCS_6.AI.착수정_TB",
    "INT_PH": "RCS_6.AI.착수정_PH",
    "INT_WTMP": "RCS_6.AI.착수정_온도",
}
# WIDE에 없는 2성상: 학습 중앙값(parquet p50)
MISSING_FILL = {"RCS_6.AI.착수정_AL": 21.45, "RCS_6.AI.착수정_전기전도도": 75.67}
# SMLT_HSTRY 대응 컬럼(있으면 우선)
SMLT_COLS = {"RCS_6.AI.착수정_AL": "ALKLNT", "RCS_6.AI.착수정_전기전도도": "ELCDT"}
# 군집 의미(01_clustering cluster0=저탁도 규약, 온도로 계절 분리)
CLUSTER_MEANINGS = {0: "저탁도·저수온(겨울형)", 1: "중간(전이기)", 2: "고탁도·고수온(여름형)"}
