from setuptools import find_packages, setup
from setuptools_cythonize import get_cmdclass

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    name="market_simulator",
    cmdclass=get_cmdclass(),
    packages=find_packages(),
    install_requires=install_requires,
    version="0.0.1"
)
