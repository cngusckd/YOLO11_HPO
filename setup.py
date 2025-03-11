from setuptools import setup, find_packages

setup(
    name='custom_util',
    version='0.1',
    packages=find_packages(),
    py_modules=['Model'],
    description='A utility package containing CustomDetTrainer class.',
    author='Your Name',
    author_email='your.email@example.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
