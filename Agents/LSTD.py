######################################################
# Developed by Alborz Geramiard Feb 22th 2013 at MIT #
######################################################
# This purpose of this agent is to perform policy evluation using a single run of LSTD on a stack of data 
from LSPI import *
class LSTD(LSPI):
    def __init__(self,representation,policy,domain,logger, sample_window = 100, output_filename = 'FixedPolicy-theta-star.txt'):
        self.output_filename = '%s-%s' %(className(domain),output_filename)
        super(LSTD,self).__init__(representation,policy,domain,logger, sample_window = sample_window)
    def learn(self,s,a,r,ns,na,terminal):
        self.storeData(s,a,r,ns,na)        
        if self.samples_count == self.sample_window:
            self.samples_count  = 0
            self.LSTD()
            savetxt(self.output_filename,self.representation.theta, delimiter='\t')
            self.logger.log('Stored Theta* in %s' % self.output_filename)

