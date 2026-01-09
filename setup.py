"""
python setup.py sdist && rm -r linfeat.egg-info
"""

from setuptools import setup, find_packages


def get_version() -> str:
    with open('linfeat/__init__.py') as fh:
        for line in fh:
            if line.startswith('__version__'):
                return line.split('=')[1].strip()[1:-1]


def main():
    setup(
        name='linfeat',
        version=get_version(),
        description='Linear model feature selection',
        url='https://github.com/linyc74/linfeat',
        author='Yu-Cheng Lin',
        author_email='ylin@nycu.edu.tw',
        license='MIT',
        packages=find_packages(),
        python_requires='>3.10',
        install_requires=[],
        zip_safe=False
    )


if __name__ == '__main__':
    main()
