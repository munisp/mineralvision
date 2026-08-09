"""
MineralVision Platform - Setup Script

This setup.py is provided for backwards compatibility.
The primary configuration is in pyproject.toml.
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
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "sqlalchemy>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
    ],
)
