#!/usr/bin/env python
from setuptools import setup, find_packages

install_requires = []

with open('requirements.txt') as f:
    for line in f.readlines():
        req = line.strip()
        if not req or req.startswith('#') or '://' in req:
            continue
        install_requires.append(req)

setup(
    name='connectivity_smallworld',
    description="Full datasets and simulation code for the manuscript"
                "'Short- and long-range connections differentially modulate the small-world network’s dynamics and state'",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/simonarvin/connectivity_smallworld',
    platforms='any',
    python_requires='>=3.7',
    version='0.1',
    install_requires=install_requires,
    project_urls={
        "Documentation": "https://github.com/simonarvin/connectivity_smallworld",
        "Source": "https://github.com/simonarvin/connectivity_smallworld",
        "Tracker": "https://github.com/simonarvin/connectivity_smallworld/issues"
    }
)
