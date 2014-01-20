"""Fixed policy. Encodes fixed policies for particular domains."""

import Policy
import numpy as np
from Representations import QRBF
from Tools import randSet, className
__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class BasicPuddlePolicy(Policy.Policy):
    __author__ = "Christoph Dann"

    def __init__(self, *args, **kwargs):
        pass

    def pi(self, s, terminal, p_actions):
        # 0 up, 1 down
        assert(len(s) == 2)
        if 0 not in p_actions:
            assert(1 in p_actions)
            return 1
        if 1 not in p_actions:
            assert(0 in p_actions)
            return 0
        d = np.ones(2) - s
        if np.random.rand() * d.sum() < d[0]:
            return 0
        else:
            return 1

    def __getstate__(self):
        return self.__dict__


class PoleBalanceBangBangPolicy(Policy.Policy):
    __author__ = "Christoph Dann"

    def __init__(self, *args, **kwargs):
        pass

    def pi(self, s, terminal, p_actions):
        if s[1] < 0:
            return 0
        elif s[1] > 0:
            return 2
        else:
            return 1

    def __getstate__(self):
        return self.__dict__

class OptimalBlocksWorldPolicy(Policy.Policy):

    def __init__(self, domain, logger=None, random_action_prob=0.0):
        self.random_action_prob = random_action_prob
        self.domain = domain

    def prob(self, s, terminal, p_actions):
        res = np.zeros(self.domain.actions_num)
        if terminal:
            res[p_actions[0]] = 1.
            return res
        res[p_actions] = self.random_action_prob / len(p_actions)
        ind = self._optimal_action(s, terminal, p_actions)
        assert ind in p_actions
        res[ind] += 1 - self.random_action_prob
        return res

    def pi(self, s, terminal, p_actions):
        # Fixed policy rotate the blocksworld = Optimal Policy (Always pick the next piece of the tower and move it to the tower
        # Policy: Identify the top of the tower.

        if terminal:
            return p_actions[0]
        if np.random.rand() < self.random_action_prob:
            return np.random.choice(p_actions)
        return self._optimal_action(s, terminal, p_actions)

    def _optimal_action(self, s, terminal, p_actions=None):
        if terminal:
            return p_actions[0]
        #next_block is the block that should be stacked on the top of the tower
        #wrong_block is the highest block stacked on the top of the next_block
        #Wrong_tower_block is the highest stacked on the top of the tower
        domain = self.domain
        blocks = domain.blocks

        correct_tower_size = 0 # Length of the tower assumed to be built correctly.
        for i in xrange(blocks):
            # Check the next block
            block = correct_tower_size
            if (block == 0 and domain.on_table(block,s)) or domain.on(block,block-1,s):
                #This block is on the right position, check the next block
                correct_tower_size += 1
            else:
                #print s
                #print "Incorrect block:", block
                # The block is on the wrong place.
                # 1. Check if the tower is empty => If not take one block from the tower and put it on the table
                # 2. check to see if this wrong block is empty => If not put one block from its stack and put on the table
                # 3. Otherwise move this block on the tower

                ###################
                #1
                ###################
                if block != 0: # If the first block is in the wrong place, then the tower top which is table is empty by definition
                    ideal_tower_top     = block - 1
                    tower_top = domain.towerTop(ideal_tower_top,s)
                    if tower_top != ideal_tower_top:
                        # There is a wrong block there hence we should put it on the table first
                        return domain.getActionPutAonTable(tower_top) #put the top of the tower on the table since it is not correct
                ###################
                #2
                ###################
                block_top = domain.towerTop(block,s)
                if block_top != block:
                    # The target block to be stacked is not empty
                    return domain.getActionPutAonTable(block_top)
                ###################
                #3
                ###################
                if block == 0:
                    return domain.getActionPutAonTable(block)
                else:
                    return domain.getActionPutAonB(block,block-1)

    def __getstate__(self):
        return self.__dict__


class GoodCartPoleSwingupPolicy(Policy.Policy):
    """for FiniteTrackCartPoleSwingUpFriction"""

    def __init__(self, domain, *args, **kwargs):
        self.representation = QRBF(domain, num_rbfs=500, logger=None,
                          resolution_max=25, resolution_min=25,
                          const_feature=False, normalize=True, seed=1)
        self.representation.theta = np.array([  1.46851279e+00,   1.95520182e+00,   9.37091477e+00,
         9.51307279e-01,  -5.51994612e-03,  -2.82097841e-02,
        -1.27578357e-05,   1.32177894e+00,   4.71667612e-01,
        -2.57592045e-04,   1.63379342e+00,   5.78430087e+00,
         6.86746016e+00,   2.49608241e+00,   4.52411830e+00,
         6.07063841e+00,   6.42809219e+00,   7.93741980e+00,
         1.36495800e-05,   4.40553299e+00,   2.80794721e-10,
         5.80850567e+00,  -4.06262748e-03,   8.15138262e+00,
         8.35916348e-01,   1.32041397e+00,   4.74697173e+00,
        -1.24058335e-05,   7.34193705e+00,  -4.14577307e-01,
         5.72059000e+00,   5.14461053e+00,   1.83184271e-08,
         6.22132126e+00,  -5.52604419e-08,   3.74271860e+00,
         4.19206426e+00,   9.33922951e+00,   5.73908734e+00,
         9.77322016e+00,   7.22433789e+00,   1.22252354e-06,
         6.46084136e-03,   3.73088293e+00,   5.19602148e-02,
        -1.84215062e+01,  -2.88656871e-04,   2.33766184e+00,
         1.84418709e-11,   1.73288813e-01,   4.75865550e+00,
         6.32282562e+00,  -7.79306263e-01,   5.92504122e+00,
        -5.36411938e-07,   2.88564852e-18,   7.08360873e-04,
        -1.11230324e+00,   7.66351531e+00,   6.50856745e+00,
         2.78191690e-02,   9.23237147e+00,   1.21246539e-01,
         5.80406400e-02,   5.23332436e+00,   7.58936674e+00,
        -1.78146159e-01,   7.50167736e+00,  -1.07122183e-03,
         4.98111152e+00,   6.09513157e+00,  -1.97032176e-06,
         6.00881276e-05,   7.29113364e-04,  -2.49776599e+00,
         2.96400986e-02,   8.08754054e+00,  -1.84136587e-10,
         1.75176062e-03,   1.11782103e+00,   1.10824929e-09,
         1.15856336e-03,   6.86399233e-05,   1.77393210e+00,
        -5.65785960e-04,   7.10262970e+00,   9.50436085e+00,
         1.22531770e-05,   5.76578669e-12,   2.44294643e+00,
         2.63186358e-03,  -3.81642698e-10,  -2.62704229e-05,
         9.92926798e-01,   7.27373702e+00,   6.88087733e-03,
         5.53517903e-06,   6.57311791e+00,   2.49951198e-01,
         5.53858277e+00,   7.69349145e-01,   5.64086475e-10,
         8.83415869e-04,   2.33483998e-01,   5.91091341e-01,
         1.34903157e+00,   4.51356962e-01,   3.89880887e-01,
        -1.17744895e+01,   1.05209599e-05,   7.32082240e+00,
         6.17571856e+00,   8.04516727e+00,   9.38781306e+00,
         1.53320243e-04,   6.79065559e+00,  -2.72933961e-07,
         1.19957990e-01,   4.23141320e-10,   3.52856400e-04,
         9.11299156e-05,   5.33687644e+00,   7.68004056e-07,
         9.05179734e-02,   1.81293987e-01,   3.69498209e+00,
        -2.09416607e-07,   5.35445719e+00,   2.88065526e-03,
         6.09973748e+00,   3.81806931e+00,   5.24605639e-03,
         3.04704019e-07,   7.35353704e-08,   6.21222217e+00,
        -7.78957166e-08,   1.71928347e+00,   4.66961398e+00,
         6.43399122e+00,  -5.71395064e-01,   5.78513880e-01,
         2.43450859e+00,   6.34360695e-05,   1.00345115e-05,
         5.02137776e+00,   9.90922724e-09,   6.77823573e-07,
         2.03053122e-02,  -6.69748282e-07,   1.61988503e-02,
         4.74521541e+00,  -3.40219303e-12,   6.11466936e+00,
        -1.50717900e-10,   5.37309244e+00,   4.75148281e+00,
         5.75839485e-04,   2.37449138e-08,   1.45357297e-03,
        -6.44081845e-04,   4.78208399e-04,   5.31054107e+00,
         1.08486714e-01,   8.93642431e-03,  -4.75266222e-12,
         8.30159097e+00,  -3.68342675e+00,   3.38348478e+00,
         6.51473329e+00,   7.98008869e-05,   2.63116933e-04,
        -5.66612455e+00,   1.77492029e-08,   4.91446202e+00,
         1.16476628e-14,   1.93473191e-03,  -2.93757012e-10,
        -7.05007494e+00,   4.80153886e+00,   7.46933428e+00,
         3.97836951e-03,   1.89881645e-09,   1.84448883e+00,
         9.79797496e+00,   2.04559496e+00,  -7.58517534e-01,
         6.25854061e+00,   3.48088467e+00,   4.01245537e-11,
        -2.80552337e-13,   2.29124955e+00,   2.94606683e-09,
        -8.12397541e-02,   5.01057062e+00,   8.40438953e+00,
         5.25793676e-02,   1.45390920e-08,   8.76235444e+00,
         1.76648444e-13,   6.96063205e-04,   1.29999497e-03,
         1.52292642e-02,  -1.54456920e-04,   2.34925330e-10,
         1.31057964e+00,   8.04119511e+00,   5.99612198e+00,
         5.35156554e-05,   1.58562333e-05,   5.85420657e+00,
        -1.76020875e-06,  -1.54665016e+00,   6.12771726e+00,
         1.84309036e-10,   5.64601187e+00,   2.10261529e-01,
         3.10225816e-10,   2.82898060e-12,   3.52186548e-01,
        -2.62859416e-02,   8.80780220e+00,   4.97962393e+00,
         8.47324555e-06,   1.77353143e-05,  -6.65011572e-05,
         3.50408456e-12,   7.05299734e-01,   6.38336508e+00,
         5.31219070e+00,   7.53698055e+00,   7.29201343e-06,
         5.71048637e+00,   4.09042780e+00,   5.15982188e+00,
         7.79811805e-05,   6.50918284e+00,   2.01316249e-02,
         3.34591569e-02,   3.82343152e-06,   1.58563839e+00,
         3.04443263e-10,   1.00624962e-06,   7.73423648e+00,
         9.41209113e+00,   6.74753549e+00,   2.52297859e-01,
        -2.41529656e-02,   2.45987683e+00,  -1.37808149e-05,
         4.41186888e+00,   7.26524575e+00,   3.27877212e-03,
         4.79662399e-02,   5.72241788e-08,   4.45642568e-02,
         4.88184194e-08,   4.51071471e+00,   2.37910126e-02,
         2.93565067e-01,   2.54857306e-02,   3.50602710e+00,
        -6.83596784e-02,   3.34904156e+00,   8.73235547e+00,
         9.81934969e-05,  -4.80810078e-05,   6.39083458e-01,
         6.27958505e+00,   4.13841923e+00,  -4.82906589e-04,
         5.16498037e-03,   6.60098882e+00,   2.50675445e-01,
         1.00774360e-06,   4.10668279e-13,  -9.75400351e-18,
         3.42867709e+00,  -5.59190762e-10,   5.29644417e+00,
         4.00368171e-06,   4.87618180e+00,   5.27536094e+00,
         5.67049842e+00,   7.54076493e-05,   6.00601423e+00,
         7.61430285e-05,   4.82887054e+00,   7.34215638e+00,
         8.64624967e+00,   3.54129212e-05,   4.09984355e-07,
         8.34175771e+00,   8.09052770e+00,   1.80377960e-04,
         5.85797295e-09,   5.76874895e+00,   1.92581842e-02,
         9.86168489e-10,   3.83612274e+00,  -9.19833842e+00,
         6.15119188e+00,   1.55997862e-05,   7.61676499e-07,
         6.84608749e+00,   1.17754049e-04,  -1.21453150e-09,
         3.91185031e-08,   7.74179398e+00,   2.70871968e-01,
         4.07814815e-02,   2.95723923e-03,   2.37804220e+00,
        -2.90077014e-07,   9.64140320e+00,   2.97001647e-02,
         1.35291735e-03,   6.40715884e+00,   7.22422018e+00,
         5.10172727e+00,   7.06406282e+00,   3.63846218e-10,
         6.64504938e+00,   4.35765857e+00,   1.69717733e-03,
         1.35662183e+00,   8.27132395e+00,   4.23621652e+00,
         9.37137829e+00,   1.07411256e-02,  -7.77774257e-03,
        -4.56863537e-01,   7.48641608e+00,   6.56201950e+00,
        -7.55312643e-14,   4.83692838e+00,   4.39439515e-09,
         7.48574692e+00,   7.62181424e+00,   6.82215370e+00,
         7.26413869e+00,   8.35547915e-08,   4.81661969e-02,
         1.36767363e-07,   9.39230561e-01,  -1.22958267e-08,
         2.60079926e+00,   1.03380580e-03,   8.44630913e+00,
         3.08120424e-01,   1.05260093e+00,   5.96536031e+00,
         5.71860388e+00,   2.28900289e-06,   1.34137961e-05,
         6.61647049e+00,   6.56916747e-16,   7.62745538e+00,
         9.32010822e+00,   3.84879096e-04,   4.36273301e-05,
         7.08324083e-06,  -3.15190636e-10,   8.33473859e+00,
        -6.08880304e-08,   1.96149057e+00,   5.50602454e+00,
         6.64547176e-08,   1.44157784e+00,   5.94404749e+00,
         4.10056096e-01,   1.18039346e+00,  -1.15823147e-12,
         8.81289707e+00,   5.99145373e+00,  -1.86402318e-07,
         1.01415968e-08,  -7.44468241e-05,   8.56709669e-09,
         5.65552419e-11,   2.91667667e-18,   2.46858280e+00,
         4.07660112e-03,   8.22471659e+00,   5.32249551e+00,
        -1.15840938e-13,   8.58049652e-01,   5.33703039e+00,
         4.50140295e+00,   6.03865015e+00,   3.71981667e+00,
         7.12884493e+00,   1.72940651e-13,   5.23449071e+00,
         4.03767346e-03,  -1.46222282e-05,   5.47622077e-09,
         1.05510335e+00,   6.39456947e+00,   7.25188223e-04,
         4.82936027e+00,   3.62059429e+00,   9.12493591e-01,
         2.79146895e+00,   6.52550174e+00,   6.61008857e+00,
         5.80742849e+00,   1.12832375e+00,   3.95645671e+00,
         5.32806676e+00,   4.46729533e+00,   6.82113097e+00,
         3.43635150e-08,   3.13854743e-06,   2.29224437e-01,
         1.49309639e-07,   6.91485038e-11,   6.74752149e-10,
         9.33028921e-05,   3.69068087e+00,   1.39756343e+00,
         9.63374125e+00,  -8.30769753e-03,   6.21596705e+00,
         6.46611957e+00,   3.15659660e-07,  -1.69880205e-07,
         9.42768234e+00,   5.10236810e+00,   1.32388124e+00,
         3.03510964e-02,  -5.42229231e-06,   9.72548517e-05,
         8.83087669e-03,   8.37670832e-14,   2.30957424e+00,
         5.03871486e+00,   3.17262565e+00,   1.01816877e-03,
         3.77125472e-01,   3.96863733e+00,   2.41335895e+00,
         4.56938875e+00,   7.67132739e+00,   2.80145188e-05,
         4.48584419e-01,   3.25625912e-07,  -2.98137532e-12,
        -2.04824293e-04,   3.92966534e-09,   6.85393447e+00,
         1.01641438e-13,   2.28597035e+00,   4.48576986e-03,
        -4.37952804e+00,   1.60731273e-05,   8.42692597e+00,
         6.21536006e-04,   1.46818583e+00,  -3.09267856e-02,
         6.80771660e-03,   2.10653576e+00,  -2.90379281e-03,
         2.31788059e+00,   5.87177780e+00,  -3.63722816e-02,
         1.44182988e-08,  -3.77358644e+00,   5.67549101e+00,
         7.20366315e+00,   1.92123711e+00,   6.11616363e+00,
         6.37698822e-07,   2.93999421e+00,   6.98639448e-01,
         1.01718677e+01,   5.07020568e+00,   8.17498842e+00,
         1.80267963e-01,   6.80622696e+00,   6.65335931e+00,
         3.58235527e-02,  -2.25179555e-10,   3.38606182e-01,
         3.18061558e+00,  -2.77011182e-08,   1.72799510e-01,
        -4.68746629e-12,   1.87811286e-16,   6.29859530e+00,
        -1.98535255e-05,  -4.80037922e-07,   7.01833322e-10,
         3.76419345e+00,   5.67540340e+00,   1.22136037e-10,
        -3.11059950e+00,   3.87260163e-16,   1.20586025e+00,
         6.80763873e+00,   7.78126732e-02,   6.51472625e+00,
         2.89425676e-01,   9.12905640e+00,   5.81009576e+00,
         6.69035966e+00,  -4.94186147e-04,   2.80195104e+00,
         1.12545757e-02,   1.16218630e-02,   6.01598126e-05,
         6.16154938e+00,   9.40986320e+00,   5.63275621e+00,
         6.43170545e+00,   1.69953075e+00,   4.53379466e+00,
         5.37257181e+00,   5.75462594e+00,   2.25645503e-06,
         8.39088970e-01,   6.22978210e-02,   1.03219733e-01,
         7.01721665e-04,   3.47405476e+00,   1.32682361e-01,
         7.68832519e+00,  -9.08458702e-01,  -1.21981084e-02,
         4.52296772e+00,   8.71520200e+00,   4.97551642e+00,
         7.00513861e+00,   3.97102143e-14,   6.70700883e+00,
         7.42354680e-08,   5.43097416e+00,   8.73514392e-02,
         6.41704252e+00,   5.60213577e-01,   1.23288714e+00,
         6.12311222e+00,   1.08595345e-03,  -6.18894022e-02,
         8.35734167e-04,  -1.16193020e-06,  -1.18725513e+01,
         1.19188562e+00,   2.69821320e-01,   7.08935330e-12,
         6.25279291e+00,   5.86525530e+00,   4.26085766e+00,
         8.87553857e+00,   8.94752838e+00,  -1.87032210e-05,
        -3.60392544e-19,  -1.92194803e-04,   1.06074842e+00,
         6.36303600e+00,   2.06512410e+00,   2.16670271e-02,
         2.16505707e+00,   3.97485019e+00,   6.92457873e-01,
         5.12344408e+00,   6.43392675e+00,  -1.38167150e-01,
         7.00555292e+00,  -2.18379258e-04,   6.11707489e+00,
         4.24005896e-01,   2.09410593e-15,   3.97930192e-09,
         3.61187684e-03,   5.04986984e+00,   4.05567910e+00,
         6.85750367e+00,   2.49372318e-11,   6.24104532e-03,
        -1.75305281e-03,   1.18866162e-09,   1.60343012e-04,
         9.49891992e-04,   4.10546408e-01,   2.26509364e-06,
         4.46070456e+00,   7.91909905e+00,   2.00551446e-06,
         5.82042286e-10,   4.83293802e-01,   6.68500057e-08,
        -3.50941662e-13,  -2.96961482e-07,   2.16370381e-02,
         5.35961099e+00,   2.50417558e-04,  -7.42651320e-05,
         1.05571370e+00,   1.43181575e+00,   8.98838311e-02,
         5.58602772e+00,  -1.42210629e-09,   6.64539259e-05,
         1.72725774e-02,   1.34608279e-02,   5.27717028e+00,
        -3.73915744e-03,   7.62273124e-03,   7.65233286e+00,
         5.88558384e-07,   6.23278797e-01,   2.35901404e+00,
         6.82186458e+00,   9.40896839e+00,  -1.67846556e-03,
         6.48778069e+00,   1.17538943e-01,   1.53038636e-06,
         5.92244146e-05,  -1.52437884e-03,  -3.29434390e-03,
         5.65026463e+00,   3.31364228e-06,   2.61589093e-05,
         6.55778191e-02,   6.12690759e+00,  -3.55373022e-07,
         5.07792108e+00,   2.13351162e-03,   3.51302360e+00,
        -8.64397559e+00,  -1.31877021e-04,   2.50344170e-04,
         1.14927678e-03,   5.91137290e+00,   4.34865815e-05,
         1.27739989e-03,   6.35050071e+00,   2.89765757e+00,
         6.25162302e+00,   5.46794006e+00,   5.90479297e-01,
         1.59546478e+00,   9.13959536e-07,   6.35946842e+00,
         2.32148755e-05,   7.51898274e-06,   2.81847906e-02,
        -3.19085729e-06,   8.95716621e-04,   6.30383198e+00,
         9.46017196e-10,   6.61391914e+00,   2.63390041e-07,
         2.07102364e-01,   5.85231188e-04,   1.17556047e-04,
         3.32139837e-09,   2.08231651e-07,   5.93519728e-06,
         5.14224399e-03,   5.97043692e+00,   5.36894541e-03,
         1.46513599e-01,  -1.58249698e-10,   9.23423703e+00,
         5.92093972e+00,   1.14636934e-04,   8.27473347e+00,
         5.18741551e-06,   6.38097020e-10,   4.91554674e+00,
         3.54730709e-09,   1.31230706e+00,  -8.37926821e-15,
         3.74009362e-04,   2.78326451e-10,  -7.85348250e+00,
         6.22253097e+00,   5.81611237e+00,   1.18009416e-01,
         1.59536062e-08,   1.45789145e+00,   9.31667450e+00,
         2.77854452e-01,  -1.35750250e+01,   5.07084377e-01,
         1.37910564e-01,   7.30809246e-12,  -2.59374871e-14,
         5.77329032e+00,   6.54925982e-01,  -8.50805573e-02,
         4.43471411e-01,   6.92019939e+00,   4.51906885e-02,
         7.46014952e-03,   9.00380417e+00,  -8.84990019e-13,
        -6.20325472e-08,   1.28437139e-11,   4.13560310e-03,
         2.88388722e-03,   2.03977623e-10,   1.62159311e-01,
         1.30352512e-01,   7.21443091e+00,   6.79405404e-05,
         7.16599030e-05,   3.77720472e+00,   3.65235635e-06,
        -1.56566629e-02,   3.54484799e+00,   8.55480252e-14,
         6.52728048e+00,   1.26928873e+00,  -1.95079950e-12,
        -2.47237316e-19,   1.29877777e-01,   1.22805784e-01,
         8.87407316e+00,   5.88046186e+00,  -5.39546773e-12,
        -1.89431224e-06,   5.84191637e-07,   9.66647506e-02,
         2.39567751e+00,   9.85608157e-01,   8.66887188e+00,
         4.00617113e-01,   4.58204312e-06,   8.82137529e+00,
         5.63693439e+00,   2.38360573e+00,   3.02840016e-11,
         5.04839636e+00,   1.42215638e-01,   5.84824526e-05,
        -2.16779662e-05,   4.50234438e-02,   1.27600134e-09,
         2.10561482e-07,   1.76359978e+00,   9.15586719e+00,
         3.76469670e+00,   8.52591598e-02,  -4.00113563e-01,
         8.68905816e-05,   1.68196947e-07,   9.59659021e+00,
         6.75494742e+00,   2.92027724e-03,  -3.38428851e-05,
         1.58900306e-05,   5.63282910e+00,   4.66784084e-09,
         6.45859592e+00,   9.34724212e-05,   3.00756764e+00,
         1.61934233e+00,   8.22274090e+00,   1.60925156e-01,
         5.52092386e-01,   9.65184468e+00,   1.06207016e-03,
        -7.35857698e-10,   3.15895109e-02,   6.24808624e+00,
         7.99774356e+00,   2.76158013e-03,   3.60365841e-06,
         3.27675820e-02,   9.38740675e-01,   8.80300636e-06,
         7.90023406e-14,  -2.31533408e-19,   5.05141159e-01,
        -6.40064332e-10,   2.50739314e+00,   1.14447085e-05,
         5.91728889e+00,   6.35016424e+00,  -6.88791394e-01,
         4.59324663e-05,   4.82208661e+00,  -5.86804956e-08,
        -3.46298427e-03,   1.09004776e+01,   1.18584389e+00,
         1.18949288e-05,   1.25503690e-06,   8.84019461e+00,
         5.76907308e+00,   8.85482631e-04,   2.80391749e-05,
         4.39762758e+00,   1.92879468e-01,  -3.16058073e-02,
         9.18346933e-03,  -6.58587455e+00,   6.34983991e+00,
         5.78202037e-01,   4.87827318e-04,   4.94176675e+00,
         2.14517906e-04,   3.91481377e-10,   7.43647120e-09,
         1.06089829e+00,   3.80448853e-01,   5.53963155e-02,
         1.30928323e-02,   1.61221657e-04,  -1.67032158e-07,
         9.49697215e+00,   2.31739130e+00,   5.63847704e-04,
         9.43478781e+00,   7.86964864e-01,   7.39723188e+00,
         4.72697011e-01,   1.38330961e-01,   7.52814119e+00,
         6.30024373e+00,   1.94798472e-01,  -3.76021172e-01,
         1.83897731e+00,  -1.15085094e-01,   9.38556729e+00,
         8.83908946e-04,   3.54204118e-04,   5.28965434e+00,
         8.98942574e+00,   7.05103816e+00,  -7.63946173e-08,
         1.08232921e+01,   8.41976676e-03,   3.35089119e+00,
         6.33553224e+00,   1.59000985e+00,   1.23811642e+00,
         8.53512986e-01,   3.05978564e+00,   1.88263451e-07,
         3.00885685e-04,   4.27415573e-09,   7.75122790e-01,
         1.78028891e-04,   5.27914664e+00,   4.63729327e+00,
        -9.88839955e-11,   5.93130130e+00,   6.34646955e+00,
         2.54700412e-06,   2.12998112e-04,   8.64017717e+00,
         6.33476459e-01,   1.41860125e-01,   8.68811216e+00,
         1.74973901e-03,   1.57852437e-01,   3.88781056e-06,
         1.28713046e-08,   4.00533949e+00,   5.24031526e-08,
         9.65288545e-02,   8.07442616e+00,   4.10033467e-08,
         5.01547844e+00,   5.18355562e+00,   7.87931152e-02,
         8.06004565e-04,  -4.83328109e-13,   5.07456904e+00,
        -6.25323520e+00,   6.63098848e-08,  -7.55524216e-13,
         6.79395350e-11,  -7.69535260e-02,  -6.03297625e-10,
        -1.13383631e-19,   6.22108528e+00,   1.44806555e+00,
         5.98704468e+00,   6.28793417e+00,  -7.95610119e-14,
         1.12420990e-04,   8.14608861e-01,  -5.16798924e-01,
         6.28768346e+00,  -5.14702040e+00,   5.62885029e+00,
         1.17180918e-06,   2.86253885e-01,   4.08958039e+00,
        -1.60789430e-05,   1.48359804e-09,   5.11668966e+00,
         4.08687274e-03,  -2.25628879e-03,   6.09507424e+00,
         4.12331375e-01,  -7.99042529e-04,  -4.13950123e-03,
         4.93722385e+00,   4.12486699e+00,   3.28156985e+00,
        -2.57717524e-04,  -1.08526839e-01,   6.65352289e+00,
         5.66014268e+00,   5.15580782e+00,   1.34136176e-09,
         4.71437818e-09,   8.39681824e-05,   2.83940791e-05,
        -1.49519282e-09,   1.27802460e-12,   1.78949142e-04,
        -5.85965712e-01,   3.17811411e-03,   8.24421353e+00,
         4.76275120e-02,   2.97528635e+00,   6.39044734e+00,
        -9.96548132e-03,   1.66027613e-08,   9.38014033e+00,
         6.87977711e+00,   1.08746883e-01,   1.10715310e-04,
        -1.45209809e-07,   6.47151644e-04,   3.18194015e-02,
         8.86334153e-21,   8.71450808e-02,   5.43343126e+00,
         9.20061708e-07,   2.56126928e-09,   1.13312387e+00,
        -2.99470272e-03,   5.36080425e+00,   1.63949611e+00,
         8.55101062e+00,   1.57205099e-03,   4.35539046e-05,
        -1.68275695e-11,   5.77519214e-12,   1.21664908e-02,
        -4.77066656e-11,   1.90489094e+00,   4.34965767e-14,
         5.89579979e+00,   8.62977097e-03,   4.91956892e+00,
         7.22953858e-06,   7.15540031e+00,   3.05309394e+00,
         5.33660380e+00,   6.19710900e-01,   1.39493280e-02,
         7.11128691e+00,  -3.89523498e-02,   2.80448196e-06,
         6.48132183e+00,   3.31316195e+00,  -5.03517793e-11,
        -7.67914540e+00,   2.43277265e-01,   6.54482257e+00,
         5.83421256e+00,   7.40750879e+00,   9.53936521e-04,
         1.52489771e-03,   5.03646355e-03,   8.03418294e+00,
         1.47789354e-03,  -1.08194373e+01,   9.02392902e-01,
         5.96558355e+00,   6.41364453e-02,   4.20061451e-04,
         6.14326697e-08,   1.21069375e-06,   6.71762154e+00,
        -2.88985488e-08,   3.29572866e-03,  -3.36273066e-11,
        -1.23471242e-14,   4.57779714e+00,   2.79542902e-06,
         1.38791746e-06,   5.05063774e-10,   5.35783751e+00,
         4.13947629e+00,   2.39982010e-08,  -3.20568520e+00,
         2.54283388e-03,   8.56357276e-02,   6.04149785e+00,
         5.19114046e-03])

    def __getstate__(self):
        return dict(theta=self.representation.theta, centers=self.representation.rbfs_mu,
                    sigma=self.representation.rbfs_sigma)

    def pi(self, s, terminal, p_actions):
        b_actions = self.representation.bestActions(s, terminal, p_actions)
        return np.random.choice(b_actions)

class FixedPolicy(Policy.Policy):

    policyName  = '' # The name of the desired policy, where applicable. Otherwise ignored.
    tableOfValues = None

    gridWorldPolicyNames = ['cw_circle', 'ccw_circle']

    def __init__(self, representation, logger, policyName = 'MISSINGNO', tableOfValues=None):
        self.policyName = policyName
        self.tableOfValues = tableOfValues
        super(FixedPolicy, self).__init__(representation,logger)

    supportedDomains = ['InfCartPoleBalance','BlocksWorld','IntruderMonitoring',\
                        'SystemAdministrator','MountainCar','PST','GridWorld',\
                        ]
    def pi(self,s, terminal, p_actions):
        if self.tableOfValues:
            return self.tableOfValues[(s)]
        return self.pi2(s)

    def pi2(self,s, terminal, p_actions):
        domain = self.representation.domain
        if not className(domain) in self.supportedDomains:
            print "ERROR: There is no fixed policy defined for %s" % className(domain)
            return None

        if className(domain) == 'GridWorld':
            # Actions are Up, Down, Left, Right
            if not self.policyName in self.gridWorldPolicyNames:
                print "Error: There is no GridWorld policy with name %s" % self.policyName
                return None

            if self.policyName == 'cw_circle':
                # Cycle through actions, starting with 0, causing agent to go in loop
                if not hasattr(self, "curAction"):
                    self.curAction = 0  # it doesn't exist yet, so initialize it [immediately incremented]
                while (not(self.curAction in domain.possibleActions(s))):
                    # We can't do something simple because of the order in which actions are defined
                    # must do switch statement
                    if self.curAction == 0: #up
                        self.curAction = 3
                    elif self.curAction == 3: #right
                        self.curAction = 1
                    elif self.curAction == 1: #down
                        self.curAction = 2
                    elif self.curAction == 2: # left
                        self.curAction = 0
                    else: print 'Something terrible happened...got an invalid action on GridWorld Fixed Policy'
    #                 self.curAction = self.curAction % domain.actions_num
            elif self.policyName == 'ccw_circle':
                # Cycle through actions, starting with 0, causing agent to go in loop
                if not hasattr(self, "curAction"):
                    self.curAction = 1  # it doesn't exist yet, so initialize it
                while (not(self.curAction in domain.possibleActions(s))):
                    # We can't do something simple because of the order in which actions are defined
                    # must do switch statement
                    if self.curAction == 3: #right
                        self.curAction = 0
                    elif self.curAction == 0: #up
                        self.curAction = 2
                    elif self.curAction == 2: #left
                        self.curAction = 1
                    elif self.curAction == 1: # down
                        self.curAction = 3
                    else: print 'Something terrible happened...got an invalid action on GridWorld Fixed Policy'
    #                 self.curAction = self.curAction % domain.actions_num

            else:
                print "Error: No policy defined with name %s, but listed in gridWorldPolicyNames" % self.policyName
                print "You need to create a switch statement for the policy name above, or remove it from gridWorldPolicyNames"
                return None
            return self.curAction

#             # Cycle through actions, starting with 0, causing agent to go in other direction
#             if not hasattr(pi, "curAction"):
#                 pi.curAction = domain.actions_num-1  # it doesn't exist yet, so initialize it
#             if not(pi.curAction in domain.possibleActions(s)):
#                 pi.curAction -= 1
#                 if pi.curAction < 0: pi.curAction = domain.actions_num-1




        if className(domain) == 'InfCartPoleBalance':
            # Fixed policy rotate the pendulum in the opposite direction of the thetadot
            theta, thetadot = s
            if thetadot > 0:
                return 2
            else:
                return 0
        if className(domain) == 'BlocksWorld':
            # Fixed policy rotate the blocksworld = Optimal Policy (Always pick the next piece of the tower and move it to the tower
            # Policy: Identify the top of the tower.
            # move the next piece on the tower with 95% chance 5% take a random action

            #Random Action with some probability
            #TODO fix isTerminal use here
            if np.random.rand() < .3 or domain.isTerminal():
                return randSet(domain.possibleActions(s))

            #non-Random Policy
            #next_block is the block that should be stacked on the top of the tower
            #wrong_block is the highest block stacked on the top of the next_block
            #Wrong_tower_block is the highest stacked on the top of the tower
            blocks = domain.blocks
            correct_tower_size = 0 # Length of the tower assumed to be built correctly.
            while True:
                # Check the next block
                block = correct_tower_size
                if (block == 0 and domain.on_table(block,s)) or domain.on(block,block-1,s):
                    #This block is on the right position, check the next block
                    correct_tower_size += 1
                else:
                    #print s
                    #print "Incorrect block:", block
                    # The block is on the wrong place.
                    # 1. Check if the tower is empty => If not take one block from the tower and put it on the table
                    # 2. check to see if this wrong block is empty => If not put one block from its stack and put on the table
                    # 3. Otherwise move this block on the tower

                    ###################
                    #1
                    ###################
                    if block != 0: # If the first block is in the wrong place, then the tower top which is table is empty by definition
                        ideal_tower_top     = block - 1
                        tower_top = domain.towerTop(ideal_tower_top,s)
                        if tower_top != ideal_tower_top:
                            # There is a wrong block there hence we should put it on the table first
                            return domain.getActionPutAonTable(tower_top) #put the top of the tower on the table since it is not correct
                    ###################
                    #2
                    ###################
                    block_top = domain.towerTop(block,s)
                    if block_top != block:
                        # The target block to be stacked is not empty
                        return domain.getActionPutAonTable(block_top)
                    ###################
                    #3
                    ###################
                    if block == 0:
                        return domain.getActionPutAonTable(block)
                    else:
                        return domain.getActionPutAonB(block,block-1)
        if className(domain) == 'IntruderMonitoring':
            # Each UAV assign themselves to a target
            # Each UAV finds the closest danger zone to its target and go towards there.
            # If UAVs_num > Target, the rest will hold position
            #Move all agents based on the taken action
            agents  = array(s[:domain.NUMBER_OF_AGENTS*2].reshape(-1,2))
            targets = array(s[domain.NUMBER_OF_AGENTS*2:].reshape(-1,2))
            zones   = domain.danger_zone_locations
            actions = ones(len(agents),dtype=integer)*4 # Default action is hold
            planned_agents_num = min(len(agents),len(targets))
            for i in arange(planned_agents_num):
                #Find cloasest zone (manhattan) to the corresponding target
                target          = targets[i,:]
                distances       = sum(abs(tile(target,(len(zones),1)) - zones),axis=1)
                z_row,z_col     = zones[argmin(distances),:]
                # find the valid action
                a_row,a_col     = agents[i,:]
                a = 4 # hold as a default action
                if a_row > z_row:
                    a = 0 # up
                if a_row < z_row:
                    a = 1 # down
                if a_col > z_col:
                    a = 2 # left
                if a_col < z_col:
                    a = 3 # right
                actions[i] = a
#                print "Agent=", agents[i,:]
#                print "Target", target
#                print "Zone", zones[argmin(distances),:]
#                print "Action", a
#                print '============'
            return vec2id(actions,ones(len(agents),dtype=integer)*5)
        if className(domain) == 'SystemAdministrator':
            # Select a broken computer and reset it
            brokenComputers = where(s==0)[0]
            if len(brokenComputers):
                return randSet(brokenComputers)
            else:
                return domain.computers_num
        if className(domain) == 'MountainCar':
            # Accelerate in the direction of the valley
            # WORK IN PROGRESS
            x,xdot = s
            if xdot > 0:
                return 2
            else:
                return 0
        if className(domain) == 'PST':
            # One stays at comm, n-1 stay at target area. Whenever fuel is lower than reaching the base the move back
            print s
            s       = domain.state2Struct(s)
            uavs    = domain.NUM_UAV
            print s
            return vec2id(zeros(uavs),ones(uavs)*3)
