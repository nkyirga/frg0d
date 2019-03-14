from setuptools import setup

setup(name='frg0d',
      version='0.1',
      description='FRG solver for impurity Hamiltonians',
      author='Nahom Yirga',
      author_email='nkyirga@bu.edu',
      packages=['frg0d'],
      install_requires=[
        'numpy',
        'cython',
      ],
      zip_safe=False)
