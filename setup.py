from setuptools import setup, find_packages

src_packages = find_packages(include=["src", "src.*"])
statforge_packages = [p.replace("src", "statforge") for p in src_packages]

setup(
    name="statforge",
    version="0.1.0",
    packages=src_packages + statforge_packages,
    package_dir={
        "src": "src",
        "statforge": "src",
    },
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "psycopg2-binary>=2.9.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.9",
)
