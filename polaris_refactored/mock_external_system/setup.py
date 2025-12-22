"""
Setup script for Mock External System package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mock-external-system",
    version="0.1.0",
    author="POLARIS Team",
    author_email="polaris@example.com",
    description="A mock external system for POLARIS framework testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polaris/mock-external-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.9",
    install_requires=[
        req for req in requirements 
        if not req.startswith("pytest") and not req.startswith("hypothesis") 
        and not req.startswith("black") and not req.startswith("mypy")
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mock-system-start=scripts.start_mock_system:main",
            "mock-system-stop=scripts.stop_mock_system:main",
        ],
    },
)
