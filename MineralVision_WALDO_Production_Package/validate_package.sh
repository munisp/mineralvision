#!/bin/bash
# Validation script for MineralVision WALDO Production Package

echo "Starting MineralVision WALDO Production Package validation..."
echo "============================================================"

# Check directory structure
echo -e "\n[1/7] Checking directory structure..."
REQUIRED_DIRS=(
  "src/waldo_integration"
  "src/api"
  "src/ui"
  "src/database"
  "src/arcgis_integration"
  "deployment/cloud"
  "deployment/on-premise"
  "deployment/edge"
  "docs/installation"
  "docs/user_manual"
  "docs/api"
  "docs/admin"
  "docs/training"
  "sample_data/aerial_imagery"
  "sample_data/drone_video"
  "sample_data/historical_data"
)

MISSING_DIRS=0
for dir in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "❌ Missing directory: $dir"
    MISSING_DIRS=$((MISSING_DIRS+1))
  else
    echo "✅ Directory exists: $dir"
  fi
done

if [ $MISSING_DIRS -eq 0 ]; then
  echo "✅ All required directories are present."
else
  echo "❌ $MISSING_DIRS required directories are missing."
fi

# Check core files
echo -e "\n[2/7] Checking core files..."
REQUIRED_FILES=(
  "src/waldo_integration/detection.py"
  "src/waldo_integration/tracking.py"
  "src/waldo_integration/measurement.py"
  "src/waldo_integration/integration.py"
  "src/api/server.py"
  "src/api/static/index.html"
  "src/arcgis_integration/connector.py"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing file: $file"
    MISSING_FILES=$((MISSING_FILES+1))
  else
    echo "✅ File exists: $file"
  fi
done

if [ $MISSING_FILES -eq 0 ]; then
  echo "✅ All required core files are present."
else
  echo "❌ $MISSING_FILES required core files are missing."
fi

# Check deployment files
echo -e "\n[3/7] Checking deployment files..."
DEPLOYMENT_FILES=(
  "deployment/cloud/docker-compose.yml"
  "deployment/cloud/Dockerfile"
  "deployment/cloud/kubernetes/deploy.sh"
  "deployment/on-premise/install.sh"
  "deployment/edge/install.sh"
)

MISSING_DEPLOYMENT=0
for file in "${DEPLOYMENT_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing deployment file: $file"
    MISSING_DEPLOYMENT=$((MISSING_DEPLOYMENT+1))
  else
    echo "✅ Deployment file exists: $file"
  fi
done

if [ $MISSING_DEPLOYMENT -eq 0 ]; then
  echo "✅ All required deployment files are present."
else
  echo "❌ $MISSING_DEPLOYMENT required deployment files are missing."
fi

# Check documentation
echo -e "\n[4/7] Checking documentation..."
DOC_FILES=(
  "docs/installation/installation_guide.md"
  "docs/user_manual/user_manual.md"
  "docs/api/api_documentation.md"
  "docs/admin/administrator_guide.md"
  "docs/training/training_materials.md"
)

MISSING_DOCS=0
for file in "${DOC_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing documentation file: $file"
    MISSING_DOCS=$((MISSING_DOCS+1))
  else
    echo "✅ Documentation file exists: $file"
  fi
done

if [ $MISSING_DOCS -eq 0 ]; then
  echo "✅ All required documentation files are present."
else
  echo "❌ $MISSING_DOCS required documentation files are missing."
fi

# Check sample data
echo -e "\n[5/7] Checking sample data..."
SAMPLE_FILES=(
  "sample_data/README.md"
  "sample_data/aerial_imagery/aerial_001_annotations.json"
  "sample_data/drone_video/drone_flight_001_metadata.json"
  "sample_data/historical_data/detection_samples.geojson"
)

MISSING_SAMPLES=0
for file in "${SAMPLE_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing sample data file: $file"
    MISSING_SAMPLES=$((MISSING_SAMPLES+1))
  else
    echo "✅ Sample data file exists: $file"
  fi
done

if [ $MISSING_SAMPLES -eq 0 ]; then
  echo "✅ All required sample data files are present."
else
  echo "❌ $MISSING_SAMPLES required sample data files are missing."
fi

# Validate Python syntax
echo -e "\n[6/7] Validating Python syntax..."
PYTHON_FILES=(
  "src/waldo_integration/detection.py"
  "src/waldo_integration/tracking.py"
  "src/waldo_integration/measurement.py"
  "src/waldo_integration/integration.py"
  "src/api/server.py"
  "src/arcgis_integration/connector.py"
)

SYNTAX_ERRORS=0
for file in "${PYTHON_FILES[@]}"; do
  if [ -f "$file" ]; then
    if python3 -m py_compile "$file" 2>/dev/null; then
      echo "✅ Python syntax valid: $file"
    else
      echo "❌ Python syntax error in: $file"
      SYNTAX_ERRORS=$((SYNTAX_ERRORS+1))
    fi
  fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
  echo "✅ All Python files have valid syntax."
else
  echo "❌ $SYNTAX_ERRORS Python files have syntax errors."
fi

# Validate JSON syntax
echo -e "\n[7/7] Validating JSON syntax..."
JSON_FILES=(
  "sample_data/aerial_imagery/aerial_001_annotations.json"
  "sample_data/drone_video/drone_flight_001_metadata.json"
  "sample_data/historical_data/detection_samples.geojson"
)

JSON_ERRORS=0
for file in "${JSON_FILES[@]}"; do
  if [ -f "$file" ]; then
    if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
      echo "✅ JSON syntax valid: $file"
    else
      echo "❌ JSON syntax error in: $file"
      JSON_ERRORS=$((JSON_ERRORS+1))
    fi
  fi
done

if [ $JSON_ERRORS -eq 0 ]; then
  echo "✅ All JSON files have valid syntax."
else
  echo "❌ $JSON_ERRORS JSON files have syntax errors."
fi

# Summary
echo -e "\n============================================================"
echo "Validation Summary:"
echo "------------------------------------------------------------"
echo "Directory structure: $([ $MISSING_DIRS -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($MISSING_DIRS missing)")"
echo "Core files: $([ $MISSING_FILES -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($MISSING_FILES missing)")"
echo "Deployment files: $([ $MISSING_DEPLOYMENT -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($MISSING_DEPLOYMENT missing)")"
echo "Documentation: $([ $MISSING_DOCS -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($MISSING_DOCS missing)")"
echo "Sample data: $([ $MISSING_SAMPLES -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($MISSING_SAMPLES missing)")"
echo "Python syntax: $([ $SYNTAX_ERRORS -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($SYNTAX_ERRORS errors)")"
echo "JSON syntax: $([ $JSON_ERRORS -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL ($JSON_ERRORS errors)")"
echo "------------------------------------------------------------"

TOTAL_ERRORS=$((MISSING_DIRS + MISSING_FILES + MISSING_DEPLOYMENT + MISSING_DOCS + MISSING_SAMPLES + SYNTAX_ERRORS + JSON_ERRORS))
if [ $TOTAL_ERRORS -eq 0 ]; then
  echo "✅ VALIDATION PASSED: Package is ready for delivery."
else
  echo "❌ VALIDATION FAILED: $TOTAL_ERRORS issues found. Please fix before delivery."
fi
echo "============================================================"
