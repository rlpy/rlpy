"""Used for logging information from the agent, domain, experiment, etc."""

from GeneralTools import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"

class Logger(object):
    buffer = ''         # You can print into a logger without initializing its filename. Whenever the filename is set, the buffer is flushed to the output.
    filename = ''
    def setOutput(self,filename):
        if self.filename != '':
            print "Warning: logger has been initialized to another file: %s. The rest of output will be in %s" % (self.filename, filename)
        self.filename = filename
        checkNCreateDirectory(filename)
        f = open(self.filename,'w')
        f.close()
    def log(self,str, printToScreen = True):
        """
        Logs a string to the output filename associated with this logger object.
        Optionally also print this string to screen.

        :param str: String to write to log
        :param printToScreen: Boolean, if true, also print str to screen.
        """
        if printToScreen: print str
        self.buffer += str +'\n'
        if self.filename != '':
            f = open(self.filename,'a')
            f.write(self.buffer)
            f.close()
            self.buffer = ''

    def debug(self, str):
        return #self.log(str)

    def line(self):
        self.log(SEP_LINE)
