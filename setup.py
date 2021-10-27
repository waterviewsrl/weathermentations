from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='weathermentations',
      version = '0.1.1',
      description = 'Weather augmentation routines for Albumentations',
      long_description = long_description,
      keywords = 'blur',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Multimedia :: Graphics'],
      url='http://github.com/desmoteo/weathermentations',
      author='Matteo Ferrabone',
      author_email='matteo.ferrabone@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires = ['numpy'],
      zip_safe = False)
