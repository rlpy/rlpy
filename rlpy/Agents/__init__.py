from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from .TDControlAgent import Q_Learning, SARSA
# for compatibility of old scripts
Q_LEARNING = Q_Learning
from .Greedy_GQ import Greedy_GQ
from .LSPI import LSPI
from .LSPI_SARSA import LSPI_SARSA
from .NaturalActorCritic import NaturalActorCritic
