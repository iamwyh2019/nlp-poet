#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from MainWidget import Ui_Form

from PyQt5.QtWidgets import QApplication, QWidget

import uientry as poet


class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.setupUi(self)
        self.btn.clicked.connect(self.run)

    def run(self):
        heads = self.textInput.text()
        self.textOutput.setText(poet.entry(heads))


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
