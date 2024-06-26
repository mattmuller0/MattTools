from setuptools import setup, find_packages

setup(
    name='MattTools',
    version='0.4.4',
    packages=find_packages(exclude=['tests*']),
    install_requires=[ 
            'numpy',
            'matplotlib',
            'pandas',
            'scipy',
            'seaborn',
            'scikit-learn',
            'statsmodels',
            'tqdm',
            # add any other dependencies here
            ],
    author='Matthew Muller',
    author_email='matt.alex.muller@gmail.com',
    description='Some personal tools for data analysis',
    url='https://github.com/matttmuller0/MattTools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)