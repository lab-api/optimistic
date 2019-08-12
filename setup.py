from distutils.core import setup
from setuptools import find_packages
import os
setup(
    name='optimistic',
    version='0.1',
    description='General purpose optimization framework for experiments or simulations',
    author='Robert Fasano',
    author_email='robert.j.fasano@colorado.edu',
    packages=find_packages(exclude=['docs']),
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'pandas']
)
