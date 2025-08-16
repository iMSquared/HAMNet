#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup
import zipfile
import urllib.request

if __name__ == '__main__':
    setup(name='ham',
          use_scm_version=dict(
              root='..',
              relative_to=__file__,
              version_scheme='no-guess-dev'
          ),
          ext_modules=[],
          scripts=[])
