from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='comaze',
    version='1.3.0',
    description='Framework to carry out Zero-Shot Emergent Communication experiments around the game CoMaze.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dasys-lab/comaze-python',
    
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Reinforcement Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=['gym',
                      'tensorboardx',
                      'torch',
                      'torchvision'],

    python_requires=">=3.6",
)
