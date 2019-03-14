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
      entry_points={
        'console_scripts': ['frgSolve=frg0d.frg0d:main'],
      },
      zip_safe=False)
