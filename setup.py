from setuptools import setup, find_packages
from config.custom_install import CustomInstallCommand

setup(
    name="tensorlink",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
    # Other arguments for setup()
)
