from setuptools import setup

setup(name='pylogsentiment',
      version='0.0.1',
      description='Sentiment analysis for event logs.',
      long_description='Sentiment analysis for event logs.',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
      ],
      keywords='sentiment analysis, event logs',
      url='http://github.com/studiawan/pylogsentiment/',
      author='Hudan Studiawan',
      author_email='studiawan@gmail.com',
      license='MIT',
      packages=['pylogsentiment'],
      entry_points={
          'console_scripts': [
              'pylogsentiment = pylogsentiment.pylogsentiment:main'
          ],
      },
      install_requires=[
          'nerlogparser==0.0.1',
          'scikit-learn==0.20.2',
          'keras==2.1.6',
          'keras-metrics==0.0.4',
          'imbalanced-learn==0.4.3',
          'pyparsing==2.4.6',
          'tensorflow==1.8.0',
          'h5py==2.9.0'
      ],
      include_package_data=True,
      zip_safe=False)
