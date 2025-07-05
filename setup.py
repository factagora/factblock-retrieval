from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphrag-retrieval",
    version="0.1.0",
    author="Randy Baek",
    author_email="randy@factagora.com",
    description="A modular retrieval system for graph-based retrieval augmented generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphrag-retrieval",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "neo4j==5.14.0",
        "pydantic==2.5.0",
        "python-dotenv==1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
            "black",
            "flake8",
            "mypy",
        ],
    },
)