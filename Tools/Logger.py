from GeneralTools import *

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
    def log(self,str):
    # Print something both in output and in a file
        print str
        self.buffer += str +'\n'
        if self.filename != '':
            f = open(self.filename,'a')
            f.write(self.buffer)
            f.close()
            self.buffer = ''
    def line(self):
        self.log(SEP_LINE)
        