from setuptools import setup
from setuptools.command.install import install
import subprocess
import os

PACKAGES=[
    "xgboost==1.3.3", 
    "dash",
    "dash-bootstrap-components",
    "numpy",
    "matplotlib",
    "pandas",
    "whoosh",
    "shap",
    "fire"
]

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='cpt_code_app',
      version='0.1',
      description='Dashboard to display and assign CPT codes.',
#       url='https://github.com/jlevy44/PathPretrain',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['cpt-code-app=cpt_code_app.app:main'
                               ]
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['cpt_code_app'],
      install_requires=required)#PACKAGES)
