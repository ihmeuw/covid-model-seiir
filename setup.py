from setuptools import setup
from setuptools import find_packages

setup(
    name='seiir_model',
    version='0.0.0',
    description='SEIIR model',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'argparse',
        'numpy',
        'odeopt',
        'pandas',
        'slime',
    ],
    zip_safe=False,
)
