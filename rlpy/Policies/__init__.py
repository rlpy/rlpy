from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from .eGreedy import eGreedy
from .UniformRandom import UniformRandom
from .gibbs import GibbsPolicy
from .FixedPolicy import FixedPolicy
from .FixedPolicy import BasicPuddlePolicy
