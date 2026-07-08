# 용연 정수장 AI 운영 서버

광주광역시 **용연 정수장** 전용 독립 서버 트리입니다. 이 트리는 자기 루트(`yongyeon/`)를
기준으로 동작하며, 덕남 트리와 완전히 독립적입니다. 그대로 별도 git 저장소로 분리할 수 있습니다.

상위 플랫폼 개요는 [최상위 README](../README.md) 참고.

## 용연 공정 특성

- **혼화 방식**: 복수 혼화방식 선택 가능 + 플록 분석 장치 피드백
- **응집플록분석장치**: 있음
- **초기 응집제 전략**: Jar-Test 기반 직접 예측
- **군집 데이터셋**: `water_quality_only` / `water_quality_with_flow` 두 변형

## 디렉토리

| 경로 | 설명 |
|---|---|
| `app/` | FastAPI API 서버 (`app/main.py`, `app/core/config.py` — `PLANT_ID=yongyeon` 고정) |
| `configs/` | `yongyeon.yaml` (태그·안전제약·모델), `common.yaml` (DB·API·재학습) |
| `data_pipeline/` | 용연 수집기 + 공통 전처리·피처·검증 |
| `ml/` | 학습 작업 공간 — `01_clustering`(구현됨), `02~06`(군집별 스캐폴드) |
| `models/` | 학습된 모델 산출물 (`cluster/classifier`, `coagulant/cluster_c1~c5` 등) |
| `dataset/` | 용연 학습 데이터 (`용연_*.parquet`, .gitignore 대상) |
| `docker/` | `Dockerfile.api` + `docker-compose.yml` (서비스 `epsene-yongyeon`, 포트 8002) |
| `mlops/` `simulation/` `tests/` | 공통 인프라 (복제본) |

## 실행

학습 스크립트는 **사이트 루트에서** 실행합니다.

```bash
cd yongyeon

# 군집분류기 학습 (전처리된 시간 단위 parquet 입력)
python ml/01_clustering/train_cluster_models.py

# 최종 군집 데이터셋 생성 / 리포트
python ml/01_clustering/create_final_clustered_dataset.py
python ml/01_clustering/make_clustering_report.py

# API 서버 (골격)
python -m app.main          # 또는: uvicorn app.main:app --port 8002
```

군집분류 상세는 [ml/01_clustering/yongyeon_README.md](ml/01_clustering/yongyeon_README.md) 참고.

## 컨테이너 배포

```bash
cd yongyeon
docker compose -f docker/docker-compose.yml up -d
```
