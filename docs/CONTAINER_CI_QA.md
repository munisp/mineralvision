# CONTAINER_CI_QA.md — W4 closeout: container builds + CI validation

Branch: `gap/containers` (from main `f804aad`). Date: 2026-08-09.

## Sandbox capability

- **No container runtime**: `docker`, `podman`, `nerdctl`, `buildah` all absent;
  `act` (nektos) not installable (no Go toolchain, no release-binary fetch
  sanctioned in sandbox). → Docker builds were **statically verified** instead
  of executed; every build step was validated against the repo with real
  evidence where a non-docker equivalent exists.
- Available for real validation: `python3` (3.12), `pip`, `node`/`npm`
  (system node), `pytest`, `ruff`.

## 1. Dockerfile verification (static, honest)

### `Dockerfile` (API image) — PASS (static)
- Base images `python:3.12-slim` (builder + runtime): standard, pull not attempted (no daemon).
- `COPY requirements.txt` — exists; **validated for real**: `pip install -r requirements.txt` → exit 0 in this sandbox (82 lines of pins, all resolvable on py3.12).
- `COPY MineralVision_Final_Package / MineralVision_Enhanced / MineralVision_WALDO_Production_Package` — all three dirs exist at repo root (build context = `.`).
- `HEALTHCHECK http://localhost:8000/health` — **validated for real**: imported `src.api.main:app` (after installing requirements) and enumerated routes; `/health` is mounted (`@app.get("/health")`, main.py:171).
- `CMD ["uvicorn", "src.api.main:app", ...]` with `WORKDIR /app/MineralVision_Final_Package` — module path `src.api.main:app` verified importable with that working directory layout (`app = FastAPI(...)` at main.py:108). 278+ routes enumerated at import time.
- Unverifiable in-sandbox: base-image pull, layer caching, final image size, container runtime networking.

### `Dockerfile.ui` (React + Vite → nginx) — PASS (real partial build)
- `COPY .../mineralvision-app/package.json + package-lock.json` — both exist.
- `RUN npm ci` — **executed for real**: `npm ci --no-audit --no-fund` succeeded (lockfile in sync with package.json).
- `RUN npm run build` (`tsc && vite build`) — **executed for real**: build succeeded, `dist/index.html` + assets emitted, 0 TS errors. (An earlier failure — TS5102 `baseUrl` removed — was a sandbox artifact of a globally installed TypeScript 6; the project-pinned TS ^5.2 from the lockfile builds cleanly.)
- `COPY infrastructure/nginx/ui.conf` — exists; statically checked (see §3 for the one fix applied).
- Unverifiable: `node:20-alpine` / `nginx:alpine` pulls, `wget` healthcheck inside the nginx container.

### WALDO image — PASS (static); **naming drift noted**
- The mission brief references `Dockerfile.waldo`; there is **no such file at the repo root**. The WALDO image is `MineralVision_WALDO_Production_Package/deployment/cloud/Dockerfile` (build context = `MineralVision_WALDO_Production_Package`, per its cloud compose file and the root compose service added in this branch).
- `COPY requirements.txt` — exists (pins targeted at the image's Python 3.10; `gdown==6.1.0` present as the comment claims).
- `COPY src/ /app/src/` — exists; entrypoint sets `PYTHONPATH=/app/src` and runs `python3 -m api.server` → `src/api/server.py` exists (Flask app).
- `COPY deployment/cloud/entrypoint.sh` — exists, bash, `exec python3 -m api.server`, `PORT` default 8000 matches `EXPOSE 8000`.
- **Fix applied**: `src/api/server.py` had no `/health` route, but the API's `waldo_proxy.py` (`health_check()` → `GET {WALDO_SERVICE_URL}/health`) and any container healthcheck require it. Added a minimal `/health` alias (returns `{'status': 'healthy'}`). This is the one application-code change in this branch, justified by the proven proxy↔service contract break.
- Unverifiable: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` pull, GPU runtime, `gdown` model download at image-build time (external Google Drive fetch — inherently flaky in CI; documented risk), torch/ultralytics install on py3.10.

## 2. CI workflow validation (`.github/workflows/ci.yml`)

All four jobs validated by executing their steps in-sandbox:

| Job | Step | Result |
|---|---|---|
| lint | `ruff check tests MineralVision_Enhanced/lakehouse_architecture/tests` | **was RED on main** (101 findings), **fixed** → now green ("All checks passed!") |
| lint (informational) | full-repo ruff, `continue-on-error: true` | unchanged, informational by design |
| compile | `python -m compileall -q -x ... <dirs>` | **green** (exit 0) |
| test | `pip install -r requirements.txt -r requirements-dev.txt` | **green** (both files resolve/install) |
| test | `pytest tests/ MineralVision_Enhanced/lakehouse_architecture/tests/ --cov --cov-fail-under=40 -v` | **green**: `485 passed, 3 skipped`, coverage **48.65%** ≥ 40% gate |
| docker | `docker build -f Dockerfile .` | static verification only (no daemon) — see §1 |

- Python version `3.12` matches sandbox validation env.
- Requirements split (`requirements.txt`, `requirements-dev.txt`, `requirements-ml.txt`) exists; CI uses the correct two.
- Test paths in the pytest command exist.
- No `make` targets are referenced by CI; the Makefile's `docker-build`/`docker-up` targets were statically checked and are consistent (`docker build -f Dockerfile`, `docker compose up --build`).
- `act` run not possible — static validation + real step execution documented above.

### CI drift found & fixed
1. **Ruff gate red on main**: 101 findings in `tests/` + lakehouse tests (F401, I001, N806, B905, E402, E702, E741, B007, B008, B018, C416, F841, UP034). Autofixed 50, hand-fixed 52 (mechanical: explicit `strict=False` on zip, lowercase locals, `# noqa: E402/B008` where the pattern is intentional, split semicolons, removed dead assignments). No test semantics changed — full suite re-run after fixes.
2. **Real app bug proven by the CI test step** — `lakehouse_architecture/data_storage/delta_lake_storage.py`:
   - `create_table()` hand-wrote a fake `_delta_log/00000000000000000000.json` before calling `write_deltalake`, corrupting the Delta log ("Kernel error: No table metadata or protocol found"). Platform metadata moved to a `_mineralvision_metadata.json` sidecar; fallback readers updated.
   - `write_data()` appended pandas `datetime64[ns]` batches into `date32`-declared tables → `CommitFailedError: Table features must be specified: TimestampWithoutTimezone`. Now casts incoming batches to the existing table schema (pyarrow schema via `to_pyarrow_dataset()`, since deltalake 1.6.x exposes arro3 schemas).
   - `_schema_dict_to_pyarrow()` silently mapped `array<float>` → string; parameterized `array<T>`/`list<T>` types now map to `pa.list_(T)`.
3. **Test isolation** — `tests/test_lakehouse.py::TestDeltaLakeStorage` shared the session-scoped `temp_dir` and table name `test_table` across tests with conflicting schemas; each test now uses its own subdirectory.

## 3. docker-compose.yml consistency

Root `docker-compose.yml` verified against Dockerfiles, code, and `.env.example`; **three fixes applied**:

1. **Missing `POSTGRES_PASSWORD`** — compose uses `${POSTGRES_PASSWORD:?...}` guards but `.env.example` didn't define it → fresh checkouts couldn't `docker compose up`. Added to `.env.example`.
2. **WALDO service missing + `WALDO_SERVICE_URL` wiring** — `waldo_proxy.py` defaults to `http://localhost:8001`, but no `waldo` service existed in the root compose. Added a `waldo` service (build context `MineralVision_WALDO_Production_Package`, dockerfile `deployment/cloud/Dockerfile`, host port `8001:8000`, healthcheck against the new `/health`, named volumes for models/data) and set `WALDO_SERVICE_URL: http://waldo:8000` on the `api` service.
3. **nginx `/api/innovations/*` path drift** — UI calls `${base}/api/innovations/geotoolkit/...` but the API mounts innovation routers at `/innovations/*` (verified by runtime route enumeration: `/innovations/geotoolkit/tiles/features/{z}/{x}/{y}` etc., no `/api/innovations/*`). Fixed at the infrastructure layer: `infrastructure/nginx/ui.conf` now proxies `location /api/innovations/ → http://api:8000/innovations/`. No app code touched for this.

Consistent as-found (no change needed):
- `postgres` service uses `postgis/postgis:16-3.4` (per GEOSPATIAL_SPEC).
- postgres/redis healthchecks (`pg_isready`, `redis-cli ping`) reference binaries present in those images.
- api healthcheck comes from the image's Dockerfile `HEALTHCHECK` (compose `depends_on: service_healthy` honors image healthchecks).
- Build contexts/dockerfile paths for `api` and `ui` match the files verified in §1.
- Compose YAML re-validated with a YAML parser after edits.

## Commits on `gap/containers`
1. `a578938` fix(containers): waldo /health endpoint, nginx /api/innovations rewrite, compose waldo service + WALDO_SERVICE_URL, POSTGRES_PASSWORD in .env.example
2. `381b422` fix(ci): ruff gate green, delta_lake_storage log-corruption + schema-cast bugs, lakehouse test isolation
3. docs commit (this file)

## Remaining unverifiable in-sandbox
- Actual `docker build` of all three images (no daemon): base-image pulls, multi-stage layer behavior, final image sizes.
- WALDO image's build-time `gdown` model fetch (external Drive dependency — recommend moving to a registry artifact or runtime volume mount; the compose `waldo-models` volume already supports the mount pattern).
- GPU runtime behavior of the WALDO service (compose deploy GPU reservations intentionally omitted from the root compose's `waldo` service so CPU-only dev machines can start it; the image itself is CUDA-based).
- `act` local workflow execution.
- End-to-end `docker compose up` smoke (healthcheck convergence across services).
