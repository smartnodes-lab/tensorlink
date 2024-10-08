from setuptools import setup, find_packages
from src.config.custom_install import CustomInstallCommand


VERSION = "0.0.1"
DESCRIPTION = """Tensorlink is a generalized framework for distributed model scaling in PyTorch. 
It provides tools for parsing and distributing models across a network of peers, and integrates directly into existing 
PyTorch workflows. 
"""


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return requirements


setup(
    name="src",
    version=VERSION,
    author="Smartnodes Lab",
    author_email="smartnodes-lab@proton.me",
    description=DESCRIPTION,
    packages=find_packages(where='src'),  # This will find all packages in the 'src' directory
    package_dir={'': 'src'},  # Indicate that packages are under the 'src' directory
    install_requires=parse_requirements('requirements.txt'),  # Read dependencies from requirements.txt
    # Other arguments for setup()
)
