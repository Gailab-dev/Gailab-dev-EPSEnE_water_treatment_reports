# 🚩 [규칙] Git 사용 및 커밋 컨벤션 (팀 표준)

프로젝트 코드의 일관성과 효율적인 협업을 위해 아래 규칙을 반드시 준수해 주세요.

---

## 1. 🚨 최우선 원칙: 이슈 기반 개발 (Issue-Driven Development)
모든 작업은 반드시 GitHub Issues에 생성된 티켓을 기반으로 진행되어야 합니다.

작업 시작 전: 어떤 작업을 하든지, 가장 먼저 GitHub Issues에 새로운 이슈를 생성하고 자신을 Assignee로 지정하세요.

브랜치 명명: 생성한 이슈 번호를 브랜치 이름에 포함하는 것을 권장합니다. (예: feature/ISSUE-10-user-login)

PR 연결: Pull Request(PR) 생성 시, 해당 PR이 해결하는 이슈(Closes #10)를 본문에 명시하여 PR이 머지될 때 이슈가 자동으로 닫히도록 합니다.

---

## 2. 🌿 브랜치 전략 (Simplified GitHub Flow)

저희는 `main` 브랜치를 중심으로 `feature` 브랜치를 사용하는 단순화된 **GitHub Flow**를 따릅니다.

| 브랜치 명 | 목적 | 역할 |
| :--- | :--- | :--- |
| **`main`** | **배포 가능 코드** | 서비스에 운영되는 가장 안정적인 코드를 유지합니다. **절대 직접 커밋하지 않습니다.** |
| **`feature/[기능명]`** | **기능 개발** | 새로운 기능 구현 시 `main`에서 분기하여 사용합니다. (예: `feature/user-login`, `feature/capture-page`) |
| **`fix/[수정명]`** | **긴급 버그 수정** | `main`에서 발견된 버그를 수정할 때 사용합니다. |

---

## 3. 🔄 브랜치 사용 흐름 (Workflow)

1.  **시작:** 새 작업을 시작할 때, 항상 `main` 브랜치에서 최신 코드를 당겨(pull) 받은 후, 새 `feature` 브랜치를 생성합니다.

    ```bash
    git switch main
    git pull origin main
    git switch -c feature/[기능명] # 새 브랜치 생성
    ```

2.  **완료 및 병합 요청 (PR):**
    * 기능 구현이 완료되면, **`feature/[기능명]` $\to$ `main`** 브랜치로 **Pull Request (PR)**를 생성합니다.
    * PR 생성 시 담당 팀원(Assignee)을 지정하고, 리뷰를 요청(Reviewers)하세요.
    * **PR 제목**은 3번 커밋 컨벤션(`[Type]: 내용`)을 따릅니다.

3.  **머지:** PR이 승인되면 **Squash and Merge** 또는 **Merge**를 사용하여 `main`으로 병합합니다.

---

## 4. 📝 커밋 메시지 컨벤션 (Commit Convention)

모든 커밋 메시지는 **`[Type]: [내용 요약]`** 형식으로 작성해야 합니다.

| Type | 설명 | 예시 |
| :--- | :--- | :--- |
| **`feat`** | **새로운 기능** 구현 및 추가 | `feat: 사용자 로그인 API 구현 및 JWT 적용` |
| **`fix`** | **버그 수정** | `fix: history 상세 페이지의 오타 수정` |
| **`docs`** | 문서 수정 (`README.md`, 주석, 가이드 등) | `docs: README에 mysqlclient 설치 가이드 추가` |
| **`style`** | 코드 스타일 수정 (포맷팅, 세미콜론, 들여쓰기 등) | `style: settings.py PEP8 스타일 적용` |
| **`refactor`**| 코드 리팩토링 및 구조 개선 (기능 변경 없음) | `refactor: diagnosis/views.py의 중복 로직 제거` |
| **`chore`** | 빌드, 환경 설정, 라이브러리 업데이트 등 | `chore: package.json에 axios 라이브러리 추가` |

---

## 5. ⚠️ 기타 규칙 및 유의사항

* **Conflict 해결 책임:** PR 과정에서 **Conflict**가 발생하면 **PR을 생성한 본인**이 `main` 브랜치를 pull 받아 Conflict를 해결해야 합니다.
* **.gitignore 준수:** `.env`, `node_modules/`, 가상 환경 폴더는 **절대 커밋하지 않습니다.**
* **작은 단위 커밋:** 기능 단위로 쪼개서 자주 커밋합니다. 내용이 불분명하거나 포괄적인 커밋은 지양합니다.