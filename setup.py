"""
MineralVision Platform - Setup Script

This setup.py is provided for backwards compatibility.
The primary configuration is in pyproject.toml.
Pinned runtime dependencies live in requirements.txt.
"""

from setuptools import setup, find_packages

setup(
    name="mineralvision",
    version="1.0.0",
    packages=find_packages(
        include=[
            "MineralVision_Enhanced*",
            "MineralVision_Final_Package*",
            "MineralVision_WALDO*",
        ]
    ),
    python_requires=">=3.10",
    install_requires=[
        "fastapi==0.141.1",
        "uvicorn[standard]==0.52.1",
        "pydantic==2.13.4",
        "sqlalchemy==2.0.51",
        "PyJWT==2.13.0",
        "bcrypt==5.0.0",
        "numpy==2.5.1",
        "pandas==2.3.3",
        "scipy==1.18.0",
        "scikit-learn==1.9.0",
        "httpx==0.28.1",
        "structlog==26.1.0",
        "python-dotenv==1.2.2",
        "pyyaml==6.0.3",
    ],
)
