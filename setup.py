from setuptools import setup, find_packages
import os

# Version of the package
VERSION = "0.1.1"

# Description of the package
DESCRIPTION = (
    "Tensorlink is a library designed to simplify the scaling of PyTorch model training and inference, offering tools "
    "to easily distribute models across a network of peers and share computational resources both locally and globally."
)


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Adjust the content for PyPI: Remove the last 4 lines and replace with markdown link
    content_lines = content.splitlines()
    if len(content_lines) >= 4:
        content_lines = content_lines[:-4]  # Remove the last 4 lines
    content_lines.append(
        '[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)]'
        '(https://www.buymeacoffee.com/smartnodes)'
    )
    return "\n".join(content_lines)


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
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find packages in the current directory
    include_package_data=True,
    exclude_package_data={"": [".env"]},
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
    url="https://smartnodes.ca/tensorlink",
    project_urls={
        "Documentation": "https://smartnodes.ca/docs",
        "Source": "https://github.com/smartnodes-lab/tensorlink",
    },
    license="MIT",
)
