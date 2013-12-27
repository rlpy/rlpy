from Representations import Representation
import numpy as np

class Tree(Representation.Representation):

    def __init__(self, domain, logger, p_structure=.05, m=100, lam=2000, kappa=0.1,
                 learn_rate_coef=0.1, learn_rate_exp=-0.05, grow_coef=25, grow_exp=1.1):
        self.p_structure = p_structure
        self.m = m
        self.lam = lam
        self.kappa = kappa
        self.learn_rate_coef = learn_rate_coef
        self.learn_rate_exp = learn_rate_exp
        self.depth = 1
        self.num_dim = domain.state_space_dims
        self.root = Node(self, 1)
        self.num_nodes = 1
        self.grow_coef = grow_coef
        self.grow_exp = grow_exp
        self.gamma = domain.gamma
        self.nodelist = [self.root]
        self.t = 0

    def update_depth(self, depth):
        self.depth = max(depth, self.depth)

    def next_id(self):
        self.num_nodes += 1
        return self.num_nodes

    def add_node(self, node):
        self.nodelist.append(node)
        assert(len(self.nodelist) == self.num_nodes)
        self.update_depth(node.depth)

    @property
    def features_num(self):
        return self.num_nodes

    def learn(self, s, r, ns, terminal):
        structure_point = np.random.rand() < self.p_structure
        if not structure_point:
            self.t += 1
        return self.root.descent(s).learn(s, r, ns, terminal=terminal, structure_point=structure_point)

    def predict(self, s, terminal=False):
        if terminal:
            return 0.
        return self.root.descent(s).value

    def predict_id(self, s, terminal=False):
        return self.root.descent(s).id

    @property
    def theta(self):
        res = np.array([n.value for n in self.nodelist])
        return res

    def output(self):
        print "Nodes:", self.features_num
        print "Depth:", self.depth

class Node(object):

    depth = 0
    right = None
    left = None
    value = 0.
    tree = None


    def __init__(self, tree, id=-1, value=0., depth=1):
        self.value = value
        self.depth = depth
        if id == -1 and tree is not None:
            id = tree.next_id()
            tree.add_node(self)
        self.id = id
        self.tree = tree
        m = self.tree.m
        d = min(1 + np.random.poisson(self.tree.lam), self.tree.num_dim)
        self.structure_count = 0
        self.struct_st = 0.
        self.struct_stsq = 0.
        self.cand_split_dim = np.random.permutation(np.arange(self.tree.num_dim))[:d]
        self.cand_split_st = np.zeros((d, m, 2))
        self.cand_split_val = np.zeros((d, m))
        self.cand_split_stsq = np.zeros((d, m, 2))
        self.cand_split_count = np.zeros((d, m, 2), dtype="int")

    @property
    def leaf(self):
        return self.right is None and self.left is None

    def output(self, recursive=True):
        print "Node ID", self.id
        print "\tLeaf", self.leaf
        print "\tValue", self.value
        if not self.leaf:
            print "\tSplit s[", self.split_d, "] <= ", self.split_val
            print "\tLeft: Node", self.left.id, ",\tRight: Node", self.right.id
        print "\tDepth", self.depth
        print "\tStructure points seen", self.structure_count
        print "\tCand split dim", self.cand_split_dim
        if recursive and not self.leaf:
            self.left.output(recursive)
            self.right.output(recursive)

    def descent(self, s):
        if self.leaf:
            return self
        if s[self.split_d] <= self.split_val:
            return self.left.descent(s)
        else:
            return self.right.descent(s)

    def learn(self, s, r, ns, terminal, structure_point=False):
        if structure_point:
            # structure stream
            return self._learn_structure(s, r, ns, terminal)
        else:
            return self._learn_estimation(s, r, ns, terminal)

    def alpha(self):
        return self.tree.grow_coef * (self.tree.grow_exp)**self.depth

    def beta(self):
        return 4 * self.alpha()

    def mu(self):
        """learning rate for updating the value estimate"""
        return self.tree.learn_rate_coef * self.tree.t**self.tree.learn_rate_exp

    def _learn_estimation(self, s, r, ns, terminal):
        """improve estimation value based on this observation in this node
        only called in leafs"""
        delta = r + self.tree.gamma * self.tree.predict(ns, terminal=terminal) - self.value

        self.value += delta * self.mu()
        assert np.isfinite(self.value)

    def split_node(self, dimension_id, split_id):
        self.split_d = self.cand_split_dim[dimension_id]
        self.split_val = self.cand_split_val[dimension_id, split_id]
        # create children
        self.left = Node(self.tree, -1, self.value, self.depth + 1)
        self.right = Node(self.tree, -1, self.value, self.depth + 1)

    def _learn_structure(self, s, r, ns, terminal):
        i = self.structure_count
        # add candidate splits if not enough available
        if i < self.tree.m:
            for j in xrange(self.cand_split_val.shape[0]):
                self.cand_split_val[j, i] = s[self.cand_split_dim[j]]

        if i == self.tree.m:
            pass
            # all splits are generated
            # update stats for splits based on previous data

        lowest_cost = np.inf
        ind = (-1, -1)
        # update split status
        val = r + self.tree.gamma * self.tree.predict(ns)
        assert(not np.isnan(val))
        self.struct_st += val
        self.struct_stsq += val ** 2
        self.structure_count += 1

        decision = (s[self.cand_split_dim][:,None] > self.cand_split_val).astype("int")
        self.cand_split_st[:,:,0] += (1 - decision) * val
        self.cand_split_st[:,:,1] += decision * val
        self.cand_split_stsq[:,:,0] += (1 - decision) * val ** 2
        self.cand_split_stsq[:,:,1] += decision * val ** 2
        self.cand_split_count[:,:,0] += (1 - decision)
        self.cand_split_count[:,:,1] += decision

        # compute reduction in variance for each split
        exp = self.cand_split_st / self.cand_split_count
        var = self.cand_split_stsq / self.cand_split_count - exp**2
        wvar = self.cand_split_count * var
        ss = self.cand_split_count.sum(axis=2)
        wvar = wvar.sum(axis=2)
        wvar[ss > 0] /= ss[ss > 0]
        objective = self.struct_stsq / self.structure_count - (self.struct_st / self.structure_count) ** 2 - wvar
        objective[np.any(self.cand_split_count < self.alpha(), axis = 2)] = -np.inf
        assert(not np.any(np.isnan(objective)))
        # find biggest reduction
        a, b = np.unravel_index(np.nanargmax(objective), objective.shape)

        #for j in xrange(self.cand_split_val.shape[0]):
        #    for k in xrange(self.root.m):
        #        print self.cand_split_val[j, k], objective[j, k]

        if objective[a, b] > self.tree.kappa or (i > self.beta() and np.isfinite(objective[a, b])):
            # I've seen enough! Split, bitch!
            #import ipdb;ipdb.set_trace()
            self.split_node(a, b)

    def phi_nonTerminal(self, s):
        if self != self.root:
            return self.root.phi_nonTerminal(s)
        res = np.zeros(self.features_num)
        res[self.prediction_id(s)] = 1.
        return res

    def featureType(self):
        return bool

