#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Experiment import *

class OnlineExperiment (Experiment):
    # Statistics are saved as :
    DEBUG               = 0         # Show s,a,r,s'
    STEP                = 0         # Learning Steps
    RETURN              = 1         # Sum of Rewards
    CLOCK_TIME          = 2         # Time in seconds so far
    FEATURE_SIZE        = 3         # Number of features used for value function representation
    EPISODE_LENGTH      = 4
    TERMINAL            = 5         # 0 = No Terminal, 1 = Normal Terminal, 2 = Critical Terminal
    EPISODE_NUMBER      = 6
    DISCOUNTED_RETURN   = 7
    STATS_NUM           = 8         # Number of statistics to be saved

    max_steps           = 0     # Total number of interactions
    performanceChecks   = 0     # Number of Performance Checks uniformly scattered along the trajectory
    LOG_INTERVAL        = 0     # Number of seconds between log prints

   