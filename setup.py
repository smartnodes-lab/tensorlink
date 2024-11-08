from setuptools import setup, find_packages

# Version of the package
VERSION = "0.1.0"

# Description of the package
DESCRIPTION = (
    "Tensorlink is a generalized, plug-and-play framework for distributed model scaling in PyTorch. "
    "It provides tools for parsing and distributing models across a network of peers, and integrates directly into "
    "existing PyTorch workflows."
)


# Parse requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return requirements


setup(
    name="tensorlink",
    version=VERSION,
    author="Smartnodes Lab",
    author_email="smartnodes-lab@proton.me",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find packages in the current directory
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),  # Read dependencies from requirements.txt
    python_requires=">=3.11.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Unix",
    ],
    url="https://smart-nodes.xyz/tensorlink",
    project_urls={
        "Documentation": "https://smart-nodes.xyz/docs/overview",
        "Source": "https://github.com/smartnodes-lab/tensorlink",
    },
    license="MIT",
)
