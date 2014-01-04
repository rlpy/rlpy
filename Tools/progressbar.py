# -*- coding: utf-8 -*-
# Copyright: 2009 Nadia Alramli
# License: BSD
# Modified by Christoph Dann
"""Draws an animated terminal progress bar
Usage:
    p = ProgressBar("blue")
    p.render(percentage, message)
"""

from . import terminal
import sys
import os
import datetime


class Timer(object):
    def __init__(self, name=None, print_enter=True, active=True):
        self.name = name
        self.enabled = active
        self.print_enter = print_enter

    def __enter__(self):
        self.tstart = datetime.datetime.now()
        if self.print_enter and self.name and self.enabled:
            print 'Start [%s]' % self.name

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            return
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % str(datetime.datetime.now() - self.tstart)


class ProgressBar(object):
    """Terminal progress bar class"""
    TEMPLATE = ('%(percent)-2s%% %(color)s%(progress)s%(normal)s%(empty)s %(message)s\n')
    PADDING = 7

    def __init__(self, enabled=True, color=None, width=None, block='#', empty=' '):
        """
        color -- color name (BLUE GREEN CYAN RED MAGENTA YELLOW WHITE BLACK)
        width -- bar width (optinal)
        block -- progress display character (default '█')
        empty -- bar display character (default ' ')
        """
        self.enabled = enabled
        if not enabled:
            return
        try:
            if not os.isatty(sys.stdout.fileno()):
                return
        except AttributeError:
            return
        if not os.isatty(sys.stdout.fileno()):
            return
        if color:
            self.color = getattr(terminal, color.upper())
        else:
            self.color = ''
        if width and width < terminal.COLUMNS - self.PADDING:
            self.width = width
        else:
            # Adjust to the width of the terminal
            self.width = terminal.COLUMNS - self.PADDING
        self.block = block
        self.empty = empty
        self.progress = None
        self.lines = 0

    def __enter__(self):
        self.tstart = datetime.datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        if not self.enabled: return
        if type is None:
            msg = 'Total: %s' % str(datetime.datetime.now() - self.tstart)
            self.done(msg)
        if type is KeyboardInterrupt:
            print "Aborted by user"

    def update(self,cur, total, msg = '', time=True):
        if not self.enabled: return
        elapsed = datetime.datetime.now() - self.tstart
        perc = float(cur) / float(total)
        if perc > 0:
            est = datetime.timedelta(seconds=elapsed.total_seconds() / perc)
        else:
            est = "-"
        msg += ' | {} / est. {}'.format(elapsed, est)
        msg = msg.strip()
        self.render(int(perc * 100), msg)

    def done(self, message='done'):
        return self.render(100, message)

    def render(self, percent, message = ''):
        """Print the progress bar
        percent -- the progress percentage %
        message -- message string (optional)
        """
        if not self.enabled: return
        if not hasattr(sys.stdout, "fileno") or not os.isatty(sys.stdout.fileno()):
            if message:
                print percent, '%  ', message
            else:
                print percent, '%  '
            return
        inline_msg_len = 0
        if message:
            # The length of the first line in the message
            inline_msg_len = len(message.splitlines()[0])
        if inline_msg_len + self.width + self.PADDING > terminal.COLUMNS:
            # The message is too long to fit in one line.
            # Adjust the bar width to fit.
            bar_width = terminal.COLUMNS - inline_msg_len - self.PADDING
        else:
            bar_width = self.width

        # Check if render is called for the first time
        if self.progress is not None:
            self.clear()
        self.progress = (bar_width * percent) / 100
        data = self.TEMPLATE % {
            'percent': percent,
            'color': self.color,
            'progress': self.block * self.progress,
            'normal': terminal.NORMAL,
            'empty': self.empty * (bar_width - self.progress),
            'message': message
        }
        sys.stdout.write(data)
        sys.stdout.flush()
        # The number of lines printed
        self.lines = len(data.splitlines())

    def clear(self):
        """Clear all printed lines"""
        if not self.enabled: return
        try:
            if not os.isatty(sys.stdout.fileno()):
                return
        except AttributeError:
            return
        sys.stdout.write(
            self.lines * (terminal.UP + terminal.BOL + terminal.CLEAR_EOL)
        )
