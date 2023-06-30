from setuptools import setup, find_packages


with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='sarcasme',
      version='1.0',
      description='Sarcasme detection with Tensorflow',
      author='Afaf Jaber,Cédric Najdek,Aminata Traore',
      #author_email='',
      url='https://github.com/AfaJab/2d-',
      packages=find_packages(),
      install_requires=requirements
     )
