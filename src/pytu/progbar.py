from __future__ import print_function
import re
import sys, os

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'
    # STATS = '%(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=None, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self._width = width
        self.total = total
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)
        # self.fmt2 = re.sub(r'(?P<name>%\(.+?\))d',
        #     r'\g<name>%dd' % len(str(total)), self.STATS)

        self.current = 0

    def __call__(self):
        _, _columns = os.popen('stty size', 'r').read().split()
        COLUMNS = int(_columns)
        if self._width is None:
            self.width = COLUMNS
        else:
            self.width = self._width
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        # args2 = {
        #     'total': self.total,
        #     'current': self.current,
        #     'percent': percent * 100,
        #     'remaining': remaining
        # }
        # N = len(self.fmt2 % args2)
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        # bar = '[' + self.symbol * size + ' ' * (self.width - size - N - 5) + ']'
        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }

        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
