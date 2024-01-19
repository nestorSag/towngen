# minimal setup.py
from setuptools import setup, find_packages

setup(
    name='towngen',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'pycountry',
        'keras-nlp',
        'tensorflow==2.14',
    ],
    entry_points={
        'console_scripts': [
            'towngen = towngen.entrypoint:entrypoint'
        ]
    }
)