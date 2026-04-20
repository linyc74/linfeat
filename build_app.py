import os
import shutil
import platform
import argparse
import subprocess
from linfeat import __version__
from os.path import dirname, basename


PROG = 'python build_app.py'
APP_NAME = 'LinFeat'
DESCRIPTION = f'Build MacOS app or Windows exe for {APP_NAME}-{__version__}'
REQUIRED = []
OPTIONAL = [
    {
        'keys': ['-h', '--help'],
        'properties': {
            'action': 'help',
            'help': 'show this help message',
        }
    },
]


class EntryPoint:

    parser: argparse.ArgumentParser

    def main(self):
        self.set_parser()
        self.add_required_arguments()
        self.add_optional_arguments()
        self.run()

    def set_parser(self):
        self.parser = argparse.ArgumentParser(
            prog=PROG,
            description=DESCRIPTION,
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter)

    def add_required_arguments(self):
        group = self.parser.add_argument_group('required arguments')
        for item in REQUIRED:
            group.add_argument(*item['keys'], **item['properties'])

    def add_optional_arguments(self):
        group = self.parser.add_argument_group('optional arguments')
        for item in OPTIONAL:
            group.add_argument(*item['keys'], **item['properties'])

    def run(self):
        args = self.parser.parse_args()
        BuildApp().main()


class BuildApp:

    os_name: str
    entrypoint_py: str

    def main(self):
        self.set_os_name()
        self.write_entrypoint_py()
        if self.os_name == 'Darwin':
            self.write_setup_py()
            self.build_macos_app()
        elif self.os_name == 'Windows':
            self.build_windows_exe()

    def set_os_name(self):
        self.os_name = platform.system()
        assert self.os_name in ['Darwin', 'Windows'], f'Unsupported OS: {self.os_name}'

    def write_entrypoint_py(self):
        o = 'win' if self.os_name == 'Windows' else 'mac'
        self.entrypoint_py = f'{APP_NAME}-{o}-{__version__}.py'

        with open(self.entrypoint_py, 'w') as f:
            f.write(f'''\
from linfeat import app


if __name__ == '__main__':
    app()
''')

    def write_setup_py(self):
        with open('setup.py', 'w') as f:
            f.write(f'''\
import glob
from setuptools import setup


setup(
    app=['./{self.entrypoint_py}'],
    data_files=[],
    options={{
        'py2app': {{
            'iconfile': './icon/logo.ico',
            'resources': ['./icon'],
            'packages': ['cffi', 'PyQt5']
        }}
    }},
    setup_requires=['py2app'],
)
''')

    def build_macos_app(self):
        subprocess.check_call('python setup.py py2app', shell=True)

        f = self.entrypoint_py[:-3]
        shutil.copy('./lib/libffi.8.dylib', f'./dist/{f}.app/Contents/Frameworks/')

        os.rename(f'./dist/{f}.app', f'./{f}.app')

        subprocess.check_call(f'zip -r {f}.zip {f}.app', shell=True)

        for dir_ in ['build', 'dist']:
            shutil.rmtree(dir_)
        for file in [self.entrypoint_py, 'setup.py']:
            os.remove(file)

    def build_windows_exe(self):
        cmd = f'pyinstaller --clean --onefile --icon="icon/logo.ico" --add-data="icon;icon" {self.entrypoint_py}'
        subprocess.check_call(cmd, shell=True)

        f = self.entrypoint_py[:-3]
        os.rename(f'./dist/{f}.exe', f'./{f}.exe')

        for dir_ in ['build', 'dist']:
            shutil.rmtree(dir_)
        for file in [self.entrypoint_py, f'{f}.spec']:
            os.remove(file)


if __name__ == '__main__':
    EntryPoint().main()


