from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='taxifare_deep',
    version="1.0",
    author="Bruno Lajoie",
    author_email="bruno@lewagon.org",
    description="""Packaged neural network-based predictor for the Kaggle's NY
       Taxi Fare challenge""",
    install_requires=requirements,
    packages=find_packages(),
    test_suite='tests',
    include_package_data=True,
    scripts=['scripts/taxifare_deep-run'],
    zip_safe=False
)
