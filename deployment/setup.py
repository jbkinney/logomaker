from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='logomaker',
      version='0.8.0',
      description='Package for making Sequence Logos',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='Sequence Logos',
      url='http://logomaker.readthedocs.io',
      author='Ammar Tareen and Justin B. Kinney',
      author_email='tareen@cshl.edu',
      license='MIT',
      packages=['logomaker'],
      include_package_data=True,
      install_requires=[
        'numpy',
		'matplotlib>=2.2.2',
		'pandas'
      ],
      zip_safe=False)