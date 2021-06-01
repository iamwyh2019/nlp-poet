#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
from MainWidget import Ui_Form

from PyQt5.QtWidgets import QApplication, QWidget

import qt5_applications

dirname = os.path.dirname(qt5_applications.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

import uientry as poet


class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.setupUi(self)
        self.btn.clicked.connect(self.run)
        self.textInput.returnPressed.connect(self.run)
        self.modeSelect.addItems(map(lambda x : x.name, poet.getAllModes()))
        self.modeSelect.setCurrentText(poet.getCurMode().name)
        self.modeSelect.currentIndexChanged.connect(self.changeMode)

    def run(self):
        heads = self.textInput.text()
        self.textOutput.setText(poet.getCurMode().entry(heads))
    
    def changeMode(self, index):
        poet.setCurModeIndex(index)


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
