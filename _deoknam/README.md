# 덕남 정수장 AI 운영 서버

광주광역시 **덕남 정수장** 전용 독립 서버 트리입니다. 이 트리는 자기 루트(`deoknam/`)를
기준으로 동작하며, 용연 트리와 완전히 독립적입니다. 그대로 별도 git 저장소로 분리할 수 있습니다.

상위 플랫폼 개요는 [최상위 README](../README.md) 참고.

## 덕남 공정 특성

- **혼화 방식**: 기계식 급속혼화 단일 방식 (Letterman 산식 기반)
- **응집플록분석장치**: 없음
- **초기 응집제 전략**: 침전수 탁도 피드백 기반 역산
- **군집 데이터셋**: `water_quality_only` (원수 수질 4개 피처)

## 디렉토리

| 경로 | 설명 |
|---|---|
| `app/` | FastAPI API 서버 (`app/main.py`, `app/core/config.py` — `PLANT_ID=deoknam` 고정) |
| `configs/` | `deoknam.yaml` (태그·안전제약·모델), `common.yaml` (DB·API·재학습) |
| `data_pipeline/` | 덕남 수집기 + 공통 전처리·피처·검증 |
| `ml/` | 학습 작업 공간 — `01_clustering`(구현됨), `02~06`(군집별 스캐폴드) |
| `models/` | 학습된 모델 산출물 (`cluster/classifier`, `coagulant/cluster_c1~c5` 등) |
| `dataset/` | 덕남 학습 데이터 (`덕남_*.parquet`, .gitignore 대상) |
| `docker/` | `Dockerfile.api` + `docker-compose.yml` (서비스 `epsene-deoknam`, 포트 8001) |
| `mlops/` `simulation/` `tests/` | 공통 인프라 (복제본) |

## 실행

학습 스크립트는 **사이트 루트에서** 실행합니다.

```bash
cd deoknam

# 군집분류기 학습 (입력: dataset/덕남_응집제공정_1분.parquet)
python ml/01_clustering/train_cluster_models.py

# API 서버 (골격)
python -m app.main          # 또는: uvicorn app.main:app --port 8001
```

군집분류 상세는 [ml/01_clustering/deoknam_README.md](ml/01_clustering/deoknam_README.md) 참고.

## 컨테이너 배포

```bash
cd deoknam
docker compose -f docker/docker-compose.yml up -d
```
