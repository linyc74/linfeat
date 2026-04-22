import sys
from PyQt5.QtWidgets import QApplication
from .basic import Parameters
from .workflows import linfeat, feature_selection_workflow, statistics_workflow
from .view import View
from .model import Model
from .controller import Controller


__version__ = 'v2.0.0-beta'


def app():
    app_ = QApplication(sys.argv)

    msg = f'''\
LinFeat {__version__}
College of Dentistry, National Yang Ming Chiao Tung University (NYCU), Taiwan
Yu-Cheng Lin, DDS, MS, PhD (ylin@nycu.edu.tw)'''
    print(msg, flush=True)

    # config taskbar icon
    app_id = f'NYCU.Dentistry.LinFeat.{__version__}'
    try:
        from ctypes import windll  # only exists on Windows
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except ImportError as e:
        print(e, flush=True)

    model = Model()
    view = View(model=model)
    _controller = Controller(model=model, view=view)

    sys.exit(app_.exec_())
