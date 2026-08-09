.PHONY: install install-dev install-ml lint compile-check test test-cov run docker-build docker-up verify clean

PYTHON ?= python3
PIP ?= pip3
PKG_DIRS := MineralVision_Enhanced MineralVision_Final_Package MineralVision_WALDO_Production_Package
TEST_DIRS := tests MineralVision_Enhanced/lakehouse_architecture/tests

# Install runtime dependencies
install:
	$(PIP) install -r requirements.txt

# Install runtime + dev/test dependencies
install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# Install optional heavy ML dependencies (torch, ultralytics, ...)
install-ml:
	$(PIP) install -r requirements-ml.txt

# Ruff lint over the three package dirs and tests
lint:
	ruff check $(PKG_DIRS) tests

# Byte-compile every Python source (fails on syntax errors)
compile-check:
	$(PYTHON) -m compileall -q -x '(^|/)(\.venv|node_modules|dist|__pycache__)/' $(PKG_DIRS) tests scripts

# Run the full test suite
test:
	$(PYTHON) -m pytest $(TEST_DIRS)

# Run the test suite with coverage gate (mirrors CI)
test-cov:
	$(PYTHON) -m pytest $(TEST_DIRS) --cov --cov-fail-under=40 -v

# Run the canonical FastAPI app
run:
	cd MineralVision_Final_Package && uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Build the API Docker image
docker-build:
	docker build -f Dockerfile -t mineralvision-api .

# Bring up the full stack (api + ui + postgres + redis)
docker-up:
	docker compose up --build

# Production Readiness Baseline verification
verify:
	$(PYTHON) scripts/verify.py

# Clean generated files
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .coverage htmlcov 2>/dev/null || true
	@echo "Cleaned."
