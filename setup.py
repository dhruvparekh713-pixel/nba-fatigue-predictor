from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="nba-fatigue-predictor",
    version="0.1.0",
    author="Dhruv",
    description="NBA Q4 fatigue prediction from cumulative in-game workload",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "nba-fatigue=predict:main",
        ],
    },
)
