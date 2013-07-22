# testParser.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import re
import sys

class TestParser(object):
    
    def __init__(self, path):
        # save the path to the test file
        self.path = path

    def removeComments(self, rawlines):
        # remove any portion of a line following a '#' symbol
        fixed_lines = []
        for l in rawlines:
            idx = l.find('#')
            if idx == -1:
                fixed_lines.append(l)
            else:
                fixed_lines.append(l[0:idx])
        return '\n'.join(fixed_lines)

    def parse(self):
        # read in the test case and remove comments
        test = {}
        with open(self.path) as handle:
            raw_lines = handle.read().split('\n')
        
        test_text = self.removeComments(raw_lines)
        test['__raw_lines__'] = raw_lines
        test['path'] = self.path
        test['__emit__'] = []
        lines = test_text.split('\n')
        i = 0
        # read a property in each loop cycle
        while(i < len(lines)):
            # skip blank lines
            if re.match('\A\s*\Z', lines[i]):
                test['__emit__'].append(("raw", raw_lines[i]))
                i += 1
                continue
            m = re.match('\A([^"]*?):\s*"([^"]*)"\s*\Z', lines[i])
            if m:
                test[m.group(1)] = m.group(2)
                test['__emit__'].append(("oneline", m.group(1)))
                i += 1
                continue
            m = re.match('\A([^"]*?):\s*"""\s*\Z', lines[i])
            if m:
                msg = []
                i += 1
                while(not re.match('\A\s*"""\s*\Z', lines[i])):
                    msg.append(raw_lines[i])
                    i += 1
                test[m.group(1)] = '\n'.join(msg)
                test['__emit__'].append(("multiline", m.group(1)))
                i += 1
                continue
            print 'error parsing test file: %s' % self.path
            sys.exit(1)
        return test

    
def emitTestDict(testDict, handle):
    for kind, data in testDict['__emit__']:
        if kind == "raw":
            handle.write(data + "\n")
        elif kind == "oneline":
            handle.write('%s: "%s"\n' % (data, testDict[data]))
        elif kind == "multiline":
            handle.write('%s: """\n%s\n"""\n' % (data, testDict[data]))
        else:
            raise Exception("Bad __emit__")