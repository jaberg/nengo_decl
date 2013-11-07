#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


setup(
    name="nengo_decl",
    version="0.0.1",
    author="James Bergstra",
    author_email="",
    packages=['nengo_decl'],
    scripts=[],
    url="",
    license="BSD-3",
    description='Declarative syntax for nengo',
    requires=[
        "numpy (>=1.5.0)",
        "nengo",
    ],
    #test_suite='nengo.tests',
)
