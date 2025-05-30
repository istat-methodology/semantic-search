from setuptools import setup, find_packages

setup(
    name="semantic-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "torch",
        "qdrant_client",
        "openai",
        "python-dotenv",
        "tqdm",
    ],
    include_package_data=True,
    description="Toolkit for building semantic search applications in Python.",
    author="Istat Methodology",
    license="MIT",
)
