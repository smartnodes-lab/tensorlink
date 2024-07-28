from setuptools import setup, find_packages
from src.config.custom_install import CustomInstallCommand


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return requirements


setup(
    name="tensorlink",
    version="0.1.0",
    packages=find_packages(where='src'),  # Specify the directory to search for packages
    package_dir={'': 'src'},  # Indicate that packages are under the 'src' directory
    install_requires=parse_requirements('requirements.txt'),  # Read from requirements.txt
    cmdclass={
        "install": CustomInstallCommand,
    },
    # Other arguments for setup()
)
