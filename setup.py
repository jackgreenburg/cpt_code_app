from setuptools import setup
from setuptools.command.install import install
import subprocess
import os

PACKAGES=[
    "xgboost==1.3.3", 
    "dash",#==1.20.0",
    "dash-bootstrap-components", #==0.12.2",
    #"flask==1.1.2",
    #"jinja2==2.11.3",
    #"markupsafe==1.1.1",
    #"itsdangerous==2.0.1", 
    #"werkzeug==1.0.1",
    "numpy",
    "matplotlib",
    "pandas",
    "whoosh",
    "shap",
    #"plotly==4.14.3",
    "fire"
]

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

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
      install_requires=PACKAGES)
