from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='logomaker',
      version='0.8.4',
      description='Package for making Sequence Logos',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
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
		    'matplotlib',
		    'pandas'
      ],
      zip_safe=False)