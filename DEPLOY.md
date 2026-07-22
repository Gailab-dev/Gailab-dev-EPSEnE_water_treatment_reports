# 배포 가이드 — 덕남·용연 mock API를 회사 리눅스 서버에 Docker로 올리기

발주처가 `http://<서버 공인IP>:8001`(덕남) / `:8002`(용연)으로 API를 호출할 수 있게 하는 절차.
전제: 서버에 Docker + docker compose 설치됨, 본 레포는 **private**.

## 1. 서버 접속

```bash
ssh <계정>@<회사서버IP>
```

## 2. 레포 clone (private — PAT 인증)

GitHub → Settings → Developer settings → **Fine-grained personal access token** 발급
(Repository access: `Gailab-dev/EPSEnE_water_treatment`, Permissions: Contents **Read-only**)

```bash
git clone https://github.com/Gailab-dev/EPSEnE_water_treatment.git
# Username: <GitHub ID>,  Password: <PAT> 입력
cd EPSEnE_water_treatment
```

> URL에 PAT를 직접 넣는 방식(`https://<ID>:<PAT>@github.com/...`)은 shell 히스토리와
> `.git/config`에 토큰이 남으므로 권장하지 않는다.

## 3. 빌드·기동 (두 정수장 각각)

```bash
docker compose version   # v2.24 미만이면: touch _deoknam/.env _yongyeon/.env

cd _deoknam/docker  && docker compose up -d --build
cd ../../_yongyeon/docker && docker compose up -d --build
```

- 컨테이너: `epsene-deoknam`(8001), `epsene-yongyeon`(8002)
- `restart: unless-stopped` → 서버 재부팅 시 자동 재기동
- 상태 확인: `docker ps` — healthcheck가 붙어 있어 `(healthy)` 표시가 정상

## 4. 방화벽/네트워크 개방

서버 방화벽 (둘 중 사용하는 것):

```bash
# ufw (Ubuntu 계열)
sudo ufw allow 8001/tcp && sudo ufw allow 8002/tcp

# firewalld (RHEL/CentOS 계열)
sudo firewall-cmd --permanent --add-port=8001/tcp --add-port=8002/tcp
sudo firewall-cmd --reload
```

추가로 회사 라우터/클라우드 보안그룹에서 8001·8002 **인바운드 허용** 필요.
보안상 가능하면 **발주처 공인 IP만 허용**할 것 (예: `sudo ufw allow from <발주처IP> to any port 8001 proto tcp`).

## 5. 검증

```bash
# 서버 내부
curl http://localhost:8001/health   # → {"status":"ok","plant_id":"deoknam"}
curl http://localhost:8002/health   # → {"status":"ok","plant_id":"yongyeon"}

# 외부(내 PC 등)에서
curl http://<공인IP>:8001/api/v1/ai/deoknam/dashboard/recommendations
curl http://<공인IP>:8002/api/v1/ai/yongyeon/dashboard/recommendations
curl -N http://<공인IP>:8001/api/v1/ai/deoknam/events/stream   # SSE (Ctrl+C로 종료)
```

## 6. 운영 명령

| 작업 | 명령 (각 `*/docker` 디렉토리에서) |
| :--- | :--- |
| 로그 확인 | `docker compose logs -f` |
| 재시작 | `docker compose restart` |
| 중지/제거 | `docker compose down` |
| 코드 업데이트 반영 | `git pull` (레포 루트) 후 `docker compose up -d --build` |

## 7. 발주처 전달 정보

| 항목 | 내용 |
| :--- | :--- |
| Base URL | `http://<공인IP>:8001/api/v1/ai/deoknam` · `http://<공인IP>:8002/api/v1/ai/yongyeon` |
| API 문서(Swagger) | `http://<공인IP>:8001/docs` · `http://<공인IP>:8002/docs` |
| 인증 | 없음 (mock 단계 — 카탈로그의 Bearer 인증 미적용) |
| 응답 형식 | `{success, data, metadata:{generated_at, plant_id}}` — wtp-api-catalog v0.2.5 기준 61종 |
| CORS | 전체 허용 (브라우저에서 직접 호출 가능) |

## 주의사항

- **mock 데이터는 in-memory** — 컨테이너 재시작 시 승인/이벤트 등 변경 상태가 초기 시드로 돌아간다.
- `models/`·`dataset/` 볼륨 마운트는 mock 단계에서 사용되지 않는다 (호스트에 빈 폴더 자동 생성 — 무해).
- 운영 전환 시: CORS 허용 origin 축소(`app/main.py`), API 인증 적용, HTTPS(리버스 프록시) 검토.
