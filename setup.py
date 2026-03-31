"""Package setup for regime_rl_trading."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="regime_rl_trading",
    version="0.1.0",
    author="regime-rl-trading contributors",
    description="Regime-aware reinforcement learning for adaptive trading strategy selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/regime-rl-trading/regime-rl-trading",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "rrl-train=train:main",
            "rrl-evaluate=evaluate:main",
        ]
    },
)
