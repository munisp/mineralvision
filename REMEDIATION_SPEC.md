# REMEDIATION SPEC — MineralVision (single source of truth for Wave 1)

All agents implement to these contracts. No unilateral deviations.

## C1. Canonical Application
- ONE FastAPI entry point: `MineralVision_Final_Package/src/api/main.py` exposing `app`.
  Run: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` from `MineralVision_Final_Package/`.
- DELETE after merging useful logic into `endpoints/` routers: `main_demo.py`, `main_simple.py`,
  `main_standalone.py`, `main_production.py`.
- ALL routers in `src/api/endpoints/` MUST be mounted in `main.py` via `include_router`.
- Fix all broken imports (13 known): align names with what source modules actually define
  (e.g. `geology/drillhole_database.py` defines `CollarData`/`SurveyData` — fix importers, not the source,
  unless the source is the broken one).
- App MUST import and start cleanly with only: fastapi, uvicorn, sqlalchemy, pydantic, PyJWT, bcrypt,
  python-multipart, numpy, pandas, scipy, scikit-learn installed (no torch/ultralytics needed for API boot —
  heavy ML imports must be lazy/optional with clear feature-flag degradation).

## C2. Security Contracts
- JWT: PyJWT (`import jwt`). `JWT_SECRET` env REQUIRED — app refuses to start if unset in production mode
  (`ENV=production`); dev mode may auto-generate a random ephemeral secret with a loud warning. NO hardcoded fallback.
- Passwords: `bcrypt` library directly (NOT passlib, NOT sha256). Work factor 12.
- Auth enforced globally: `JWTMiddleware` active; public paths only: `/auth/login`, `/auth/register`,
  `/health`, `/docs`, `/openapi.json`, `/redoc`. Every other route requires valid token; mutating routes
  require appropriate role via `require_role`.
- CORS: `CORS_ORIGINS` env, comma-separated; default `http://localhost:3000,http://localhost:5173`. Never `*` with credentials.
- No seeded credentials: demo seed only when `SEED_DEMO=true`; admin password from `ADMIN_INITIAL_PASSWORD`
  env (required when seeding), never a literal.
- SQLite allowed only when `DATABASE_URL` unset (dev fallback with warning); document Postgres DSN for prod.

## C3. Packaging & Containers
- `requirements.txt` = runtime deps actually imported, pinned `==`. `requirements-dev.txt` = test/lint tools.
  Remove python-jose and passlib. Add PyJWT, bcrypt. Heavy ML (torch, ultralytics) goes to
  `requirements-ml.txt` (optional install).
- `.gitignore`: `__pycache__/`, `*.pyc`, `.DS_Store`, `.env`, `*.db`, `node_modules/`.
- Root `Dockerfile` (multi-stage, python:3.12-slim) for the API; `Dockerfile.ui` for the React app
  (node:20 build → nginx serve); root `docker-compose.yml`: api + ui + postgres + redis, coherent paths.
  Fix WALDO `deployment/cloud/` refs (Dockerfile COPY path; compose service files) or point compose at root files.
- `.dockerignore` at root.

## C4. CI & Tests
- `.github/workflows/ci.yml`: on push/PR → ruff lint, py_compile all, pytest with coverage gate (>=40% initially),
  docker build check (api image). Paths must match repo layout.
- Makefile: `install`, `install-dev`, `lint`, `test`, `run`, `docker-build`, `verify`.
- Fix `MineralVision_Enhanced` lakehouse test collection (add missing `__init__.py` / fix import path).
- FORBIDDEN: `try/except ImportError → pytest.skip` blanket pattern. Tests import directly; missing optional
  heavy deps use `pytest.importorskip("torch")` ONLY for torch/ultralytics-dependent tests.

## C5. Decontamination
- Remove `src/api/crop_monitoring/` entirely, `CropMonitoringPage.tsx`, their routes/imports/menu entries.
- Any other agriculture-domain remnants (search: crop, palm, cocoa, ginger, harvest, agric) removed or justified.

## Quality Bar (all agents)
- No placeholders, no TODO-without-implementation, no mock-by-default in production paths.
- Every change compiles; every owned test passes before commit. Commit with conventional message.
- Do not touch files outside your assigned ownership list.
