import os.path
from setuptools import setup

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Get requirements
requirements = []
with open(os.path.join(LOCAL_DIR, 'requirements.txt'), 'r') as infile:
    for line in infile:
        line = line.strip()
        if line and not line[0] == '#':
            requirements.append(line)

setup(name='lsi-oversampling',
      version='0.1',
      description='Local synthetic instances (LSI)',
      url='http://github.com/colinjosephbrown/LSI',
      author='Colin Joseph Brown',
      author_email='colinjosephbrown@gmail.com',
      license='Pending',
      packages=['lsi-oversampling'],
      install_requires=requirements)
