from setuptools import setup, find_packages
import os

VERSION = '0.0.9'
DESCRIPTION = 'Python package for the computation of Rashba-Dresselhaus splitting in semiconducting systems.'
LONG_DESCRIPTION = 'A software package for the high throughput, or fine-grained computation of Rashba-Dresselhaus splitting in semiconducting systems at the electronic structure level.'

# Setting up
setup(
    name="rashbapy",
    version=VERSION,
    author="Robert Stanton",
    author_email="<robertmstanton@proton.me>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['ase'],
    keywords=['python', 'semiconductor', 'rashba', 'dresselhaus', 'soc',
              'spinorbit coupling', 'electronic structure', 'DFT', 'xTB'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
