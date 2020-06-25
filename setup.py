
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

setup(
    name="CILC",
    version="1.0",
    author="Maude Charmetant",
    author_email="mcharmetant@astro.uni-bonn.de",
    packages=["CILC"],
    url="https://github.com/MaudeCharmetant/CILC",
    license="no License",
    description=("ILC and CILC code"),
    long_description=open("README.rst").read(),
    package_data={"CILC": ["LICENSE", "masks/*.fits"]},
    include_package_data=True,
    install_requires=["numpy", "healpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
