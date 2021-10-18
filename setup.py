import os
from codecs import open

from setuptools import find_packages, setup


# pwd
here = os.path.abspath(os.path.dirname(__file__))

# readme
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# required packages
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# set up
setup(
    name='HRSEPP',
    version='1.0.0',
    description='HRSEPP is a Python library',
    license='MIT',
    long_description=long_description,
    url='https://github.com/leelew/HRSEPP',
    author='Lu Li',
    author_email='lilu83@mail.sysu.edu.cn',
    classifiers=[
        'Development Status :: 3 - Alpha',  # 4 - Beta; 5 - Production/Stable
        'Intended Audience :: Developers',  # registered users
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='machine learning models, deep learning models',
    package_dir={'': 'HRSEPP'},
    packages=find_packages(where='HRSEPP'),
    python_requires='>=3.6, <4',
    install_requires=install_requires,
    include_package_data=False,
)
