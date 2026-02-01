from setuptools import setup, find_packages

# TOKENS (Placeholder: Ne jamais commiter de vrais tokens ici !)
TEST_PYPI_TOKEN = ""

setup(
    name="ds-toolkit-examen-project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
)
