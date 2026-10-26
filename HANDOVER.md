# food-kcal 인수인계 문서

> 작성일 2026-10-26 · 기준 커밋: `develop` `60e3dc8` (PR #7 merge 직후)
> 저장소: <https://github.com/bio-hyeok/food-kcal> · 위키: <https://github.com/bio-hyeok/food-kcal/wiki>

## 1. 한눈에 보기

음식 사진을 올리면 음식 종류를 인식하고 칼로리·영양성분을 보여주는 모바일 프로그램이다. 사용자가 먹은 양을 직접 입력해 칼로리를 다시 계산할 수 있다.

**현재 상태: 사진을 첨부하고 음식·양을 고르면 칼로리를 보여주는 모바일 앱(샘플 데이터)까지 동작한다. 사진 인식, 모델 학습, API 서버는 아직 없다.**

| 영역 | 상태 | 위치 |
| --- | --- | --- |
| 모바일 앱 | 동작 (화면 1개, 샘플 영양 데이터 11개) | `app/` |
| DB 스키마 | 완료 (001~003), CI로 검증 | `server/db/` |
| API 서버 | **미구현** (명세만 있음) | 위키 `API 문서` |
| 이미지 인식·모델 학습 | **미착수** (학습 스크립트 초안은 위키에만 있음) | 위키 `전체 흐름 문서` 6절, `ml/`(비어 있음) |
| 실제 영양 데이터 | **없음** (샘플 값만 있음) | `app/src/foods.sample.json` |
| 문서 | 위키 5페이지, v0.4 | 위키 |

## 2. 먼저 읽을 것

1. 이 문서
2. 위키 `Home` → `전체 흐름 문서` → `ERD 설계 문서` → `API 문서`
3. `app/README.md`, `server/db/README.md`

## 3. 저장소와 브랜치

### 3.1 브랜치 규칙

| 브랜치 | 역할 | 규칙 |
| --- | --- | --- |
| `develop` | **현재 기준 브랜치.** 앱과 DB가 여기서 동작해야 한다 | 직접 push 불가, Pull Request로만 변경 |
| `feature/*` | 작업 단위 | 항상 `develop`에서 만들고, 끝나면 `develop`으로 PR |
| `main` | 배포용 | **지금은 사용하지 않기로 했다.** 배포 방식이 정해지지 않았다 |

- `main`과 `develop`은 브랜치 보호가 켜져 있다. `develop`은 "Pull Request를 통해서만 변경"을 확인했다. 승인 수·필수 상태 검사 같은 세부 설정은 확인하지 못했다.
- 커밋 메시지는 `feat:` `fix:` `docs:` `chore:` `ci:` 접두어를 쓴다.
- `develop`에 합칠 때는 GitHub에서 PR을 만든다. `develop` → `main`은 "Create a merge commit"으로 합친다(squash 금지, 이력이 갈라진다).

### 3.2 현재 브랜치 상태

- `develop`에 지금까지의 모든 작업이 들어 있다 (PR #1, #2, #3, #6, #7).
- **`main`은 `develop`보다 뒤처져 있다.** `main`에는 폴더 구조와 DB 스키마 001만 있다. 또 `main`에만 있는 merge 커밋 2개(PR #4, #5)가 있어서, 나중에 `develop` → `main` PR을 만들 때 이 점을 알고 있어야 한다.
- 원격 `feature/mobile-app-kcal`은 이미 merge됐으므로 삭제해도 된다.

### 3.3 작업 순서

```bash
git clone https://github.com/bio-hyeok/food-kcal.git
cd food-kcal
git switch develop
git pull origin develop
git switch -c feature/<영역>-<내용>     # 예: feature/food-data
# 작업 → 커밋 → 푸시
git push -u origin feature/<영역>-<내용>
# GitHub에서 feature/... → develop 으로 PR 생성 후 merge
```

## 4. 폴더 구조 (실제)

```
food-kcal/
├─ .github/workflows/db-migrations.yml   # DB 마이그레이션 CI
├─ app/                                  # Expo(React Native) 모바일 앱
│  ├─ App.js                             # 화면 1개
│  ├─ src/calc.js, calc.test.js          # 칼로리 계산과 테스트
│  └─ src/foods.sample.json              # 샘플 음식·영양 데이터
├─ server/db/
│  ├─ migrations/                        # 001~003 + 각 down 파일
│  ├─ tests/                             # 제약·인덱스 테스트
│  └─ test-migrations.sh                 # 적용 → 테스트 → 되돌리기 검사
├─ ml/                                   # 비어 있음 (scripts/, configs/, label_map.csv 헤더만)
├─ data/, models/                        # 비어 있음 (내용물은 Git 제외, .gitkeep만 유지)
├─ .env.example                          # 환경 변수 예시 (.env는 Git 제외)
└─ HANDOVER.md                           # 이 문서
```

`server/`에는 DB 스키마만 있고 **API 서버 코드는 없다.**

## 5. 실행과 테스트

### 5.1 모바일 앱

필요: Node.js (개발은 Node 24, npm 11에서 했다)

```bash
cd app
npm install
npx expo start
```

- 기반: Expo SDK 58, React Native 0.88, `expo-image-picker`
- **휴대폰에서 열기**
  - Android: Play Store의 Expo Go를 설치하고, 카메라 앱이 아니라 **Expo Go 안의 "Scan QR code"**로 스캔한다. 휴대폰과 PC가 같은 Wi-Fi여야 한다.
  - iPhone: Expo 공식 문서 기준으로 App Store에서 Expo Go를 받을 수 없고, 직접 빌드하려면 유료 Apple Developer Program이 필요하다. 이 부분은 실제 단말로 확인하지 못했다.
  - 대안(둘 다 가능): `npx expo start --web --host lan` 후 휴대폰 브라우저에서 `http://<PC의 Wi-Fi IP>:8081`을 연다. Windows에서는 `ipconfig`의 Wi-Fi 어댑터 IPv4를 쓴다(WSL/Hyper-V 가상 어댑터 IP는 휴대폰에서 열리지 않는다).
- 테스트: `npm test` (계산 로직과 샘플 데이터 형식, 7개)
- 확인한 것: 웹 미리보기에서 사진 첨부·음식 선택·인분/g 입력·결과 표시, Android 번들 빌드, `expo-doctor` 통과
- **확인하지 못한 것**: 실제 휴대폰(카메라 권한, Expo Go 동작)

### 5.2 DB

- PostgreSQL 16 기준이다. 적용: `psql "$DATABASE_URL" -f server/db/migrations/001_init_schema.sql` (002, 003도 순서대로)
- 전체 검사(적용 → 테스트 → 되돌리기): PostgreSQL을 띄우고 `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`를 설정한 뒤 `bash server/db/test-migrations.sh`
- PR을 올리면 GitHub Actions `db-migrations`가 같은 검사를 자동으로 돌린다(`server/db/**`가 바뀔 때만). **앱 테스트는 CI에 없다.**
- 마이그레이션 작성 규칙: merge된 파일은 고치지 않고 번호를 올려 새 파일로 추가, down 파일과 테스트를 함께 둔다. 스키마를 바꾸면 위키 ERD 문서의 DDL도 함께 고친다.
- 003은 `pg_trgm` 확장을 쓴다. DB 계정에 확장 생성 권한이 필요하다.

### 5.3 모델 학습

아직 실행 가능한 코드가 저장소에 없다. 절차와 학습 스크립트 초안은 위키 `전체 흐름 문서` 6절에 있다. 스크립트는 **무작위 이미지 48장으로만** 끝까지 실행되는 것을 확인했고, 실제 음식 데이터에서는 돌려 보지 않았다.

## 6. 설계 결정 기록

| 결정 | 내용 | 이유·메모 |
| --- | --- | --- |
| 클라이언트 | Expo(React Native) | 개발 PC에 Flutter, Android SDK가 없고 Node만 있었다. Flutter를 원하면 다시 정해야 한다 |
| 서버 | 아직 없음 (Spring Boot 또는 FastAPI 후보) | 앱은 서버 없이 단독으로 동작하게 만들었다 |
| 칼로리 계산 | `kcal × 먹은 양(g) ÷ basis_g`, 인분은 `배율 × 1인분 중량` | ERD 문서 4절. 앱은 `basis_g`와 1인분 중량이 모두 있는 음식만 지원한다 |
| 영양정보 없는 음식 | 기록 불가 (`422 NUTRITION_NOT_AVAILABLE`) | `kcal_snapshot`이 NOT NULL이다. 바꾸려면 스키마 변경이 필요하다 |
| 음식이 아닌 사진 | 1순위 신뢰도가 `RECOGNITION_MIN_CONFIDENCE` 미만이면 인식 실패로 처리 | **임시값 0.5는 근거 없는 자리표시**다. 첫 모델 평가 후 정한다 |
| 이메일 | 대소문자 무시 유니크 (`lower(email)` 인덱스) | 조회도 `lower(email) = lower($1)`로 해야 한다 |
| 음식 검색 | 포함 검색 + `pg_trgm` 인덱스 | 검색어 3글자 이상일 때만 인덱스를 탄다. 1~2글자는 순차 스캔 |
| 시간 | `TIMESTAMP` + 한국 표준시(KST) | 해외 사용자를 받게 되면 `TIMESTAMPTZ`로 바꾼다 |
| 사진 첨부 | 앱에서는 **표시만** 하고 분석하지 않는다 | 인식·학습을 의도적으로 제외한 범위다 |

## 7. 알려진 한계와 미해결 사항

| # | 항목 | 내용 |
| --- | --- | --- |
| 1 | **영양 데이터가 샘플** | `foods.sample.json`의 11개 값은 일반적인 근사값으로 임의로 넣은 것이다. 실제 서비스 전에 교체해야 한다. 앱 상단에 경고가 표시된다 |
| 2 | 학습 데이터 미확보 | AI-Hub 74번 데이터는 마지막 확인 시점에 다운로드 전이었다. 영양정보 단위(100g당/1인분당), 양 라벨 형식, 어노테이션 형식은 받아 봐야 안다 |
| 3 | 데이터 이용 조건 | AI-Hub 데이터는 내국인만 신청 가능하고 승인이 필요하다. 모델·서비스의 상업적 이용 가능 여부는 신청 전에 확인해야 한다 |
| 4 | API 서버 없음 | 위키 API 문서는 명세일 뿐 구현이 없다 |
| 5 | 위키에 앱 구현 범위가 없음 | 위키 화면 설계는 9개 화면인데 앱은 화면 1개다. 위키에 "현재 구현 범위"를 반영해야 한다 |
| 6 | `server/db/README.md`가 옛 내용 | 마이그레이션 001만 설명한다. 002, 003을 반영해야 한다 |
| 7 | 배포 미정 | 서버 호스팅, 앱 배포 방식이 없다. `main`을 쓰지 않는 이유이기도 하다 |
| 8 | 사진 개인정보 | 업로드 사진의 EXIF 위치정보 제거, 보관 기간, 재학습 사용 동의 방식이 정해지지 않았다 |
| 9 | 브랜치 보호 세부 | 필수 승인 수, `db-migrations`를 필수 검사로 지정할지는 정하지 않았다 |
| 10 | 앱 CI·린트 없음 | 앱 테스트를 PR마다 돌리는 설정이 없다 |
| 11 | 학습 스크립트 | 합성 데이터로만 실행 확인. 실제 성능은 알 수 없다 |

위키 `전체 흐름 문서` 7절의 "확인 필요 사항" 표에도 같은 내용이 번호와 함께 정리돼 있다.

## 8. 다음 할 일 (우선순위)

1. **`feature/food-data`**: 샘플을 실제 영양 데이터(식약처 식품영양성분 DB 등)로 교체. 학습 없이 앱 품질이 가장 크게 오른다. `foodId`는 나중에 학습 모델이 내놓는 번호와 같아야 하므로 함부로 바꾸지 않는다
2. **AI-Hub 74번 신청·다운로드**: 필요한 파일만 받고(전체 1.76TB는 받지 않음) 영양정보 단위와 라벨 형식을 확인한다 → 위키 문서의 "확인 필요" 항목을 채운다
3. **`feature/train-classifier`**: 위키의 학습 스크립트를 `ml/scripts/`에 넣고 실제 데이터로 학습·평가. 배포 전에는 "모델의 모든 클래스에 영양정보가 있는지" 점검 SQL을 돌린다
4. **`feature/recognition-api`**: 서버에서 모델을 서비스 (위키 API 문서의 `POST /images`, `POST /recognitions`). 휴대폰 안에서 추론하려면 개발 빌드가 필요해서 Expo Go로는 안 된다
5. **`feature/app-recognition`**: 앱에서 후보 Top-N을 보여주고, 틀렸을 때는 지금의 검색을 쓴다
6. 위키 갱신(구현 범위), `server/db/README.md` 수정, 앱 테스트 CI 추가
7. 배포 방식이 정해지면 `develop` → `main`

## 9. 외부 의존과 주의

- **비밀 값**: API 키는 `.env`에만 둔다(`.env.example` 참고). 명령어나 스크립트에 직접 적지 않는다. `.env`, 데이터 압축 파일(`*.zip`, `*.zip.part*`), `aihubshell`, 모델 파일(`*.pt`, `*.onnx` 등)은 `.gitignore`로 제외돼 있다.
- **AI-Hub**: 내려받은 데이터는 저장소 밖이나 `data/`에 둔다. 전체 다운로드는 받을 용량의 2~3배 디스크가 필요하다. 분할 압축 병합은 Linux(또는 WSL) 명령이다.
- **Expo**: 앱 폴더의 `AGENTS.md`(AI 도구용 안내)와 `LICENSE`(Expo 템플릿의 MIT)는 템플릿에서 온 파일이다. Expo SDK는 API가 자주 바뀌므로 코드를 고치기 전에 해당 버전 공식 문서를 확인한다.

## 10. 작업 환경 메모 (Windows)

- 개발은 Windows에서 했다. 셸 스크립트는 Git Bash로 실행한다. 일부 도구(예: Python의 `subprocess`)에서 `bash`가 WSL의 bash로 잡히면 실패하므로 `C:\Program Files\Git\bin\bash.exe`를 직접 지정한다.
- `.gitattributes`가 `*.sh`를 LF로 고정한다(CRLF면 리눅스 CI에서 실행되지 않는다).
- `aihubshell`은 Linux용이라 Windows에서는 WSL이 필요하다.

## 11. 문서 안내와 갱신 규칙

| 문서 | 내용 |
| --- | --- |
| 위키 `Home` | 프로젝트 개요, 문서 목록, 현재 상태 |
| 위키 `전체 흐름 문서` | 처리 흐름, 브랜치 전략, DB 마이그레이션 규칙(4.7), 학습 데이터와 학습 방법(6), 확인 필요 사항(7) |
| 위키 `ERD 설계 문서` | 테이블 정의, 칼로리 계산 규칙(4), DDL(6) |
| 위키 `API 문서` | 엔드포인트 명세와 오류 코드 |

- 위키는 코드 저장소와 **별도의 git 저장소**(`food-kcal.wiki.git`)이고, 브랜치 보호나 PR 없이 `master`에 바로 올린다.
- 스키마를 바꾸면 위키 ERD 문서의 DDL을, 구현 범위가 바뀌면 위키 `Home`과 이 문서를 함께 고친다. 위키 문서는 코드와 어긋나지 않도록 `develop`에 merge된 내용을 기준으로 쓴다.
