from setuptools import setup, find_packages

setup(
    name="statforge",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
    ],
    python_requires=">=3.9",
)
