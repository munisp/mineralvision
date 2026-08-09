.PHONY: verify setup clean

# Production Readiness Baseline verification
verify:
	@python3 scripts/verify.py

# Setup development environment
setup:
	@echo "Installing Python dependencies..."
	@pip3 install -r requirements.txt
	@echo "Setup complete."

# Clean generated files
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
