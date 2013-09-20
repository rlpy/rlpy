#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <assert.h>
#include <string>
#include <stdexcept>

double gaussian_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths);

double linf_triangle_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths);

class cmp_dim {
    public:
    bool operator()(const std::pair<unsigned int, std::vector<unsigned int>>& p1, const std::pair<unsigned int, std::vector<unsigned int>>& p2) const {
        if (p1.second.size() == p2.second.size()) {
            for (unsigned int i=0; i < p1.second.size(); i++) {
                if (p1.second[i] == p2.second[i])
                    continue;
                return p1.second[i] < p2.second[i];
            }
        } else {
            return p1.second.size() < p2.second.size();
        }
        return p1.first < p2.first;
    }
};

class cmp_spec {
    public:
    bool operator()(const std::pair<unsigned int, std::vector<unsigned int>>& p1, const std::pair<unsigned int, std::vector<unsigned int>>& p2) const {
        if (p1.second.size() == p2.second.size()) {
            return p1.first > p2.first;
        } else {
            return p1.second.size() > p2.second.size();
        }
    }
};

template <typename T1, typename T2> std::vector<typename std::map<T1, T2>::iterator> is_subset(std::set<T1>& sb, std::map<T1, T2>& ss);

struct hash_X {
    // assumes that both std::vectors are sorte
    // duplicate entries are only considered once
    size_t operator()(const std::pair<std::set<unsigned int>, std::set<unsigned int>>& x) const{
        size_t res = 0;
        auto i = x.first.cbegin();
        auto j = x.second.cbegin();
        while (i != x.first.cend() || j != x.second.cend()) {
            if (i == x.first.cend()) {
                res ^= std::hash<unsigned int>()(*j);
                j++;
            } else if (j == x.second.cend()) {
                res ^= std::hash<unsigned int>()(*i);
                i++;
            }
            else if (*i < *j) {
                res ^= std::hash<unsigned int>()(*i);
                i++;
            } else if (*i == *j) {
                res ^= std::hash<unsigned int>()(*i);
                i++; j++;
            } else {
                res ^= std::hash<unsigned int>()(*j);
                j++;
            } 

        }
        return res;
  }
};

class Candidate {
    public:
        double relevance = 0;
        double total_activation = 0;
        double total_error = 0;
};

typedef std::unordered_map<std::pair<std::set<unsigned int>, std::set<unsigned int>>, Candidate, hash_X> CandidateMap;
typedef std::unordered_map<std::pair<std::set<unsigned int>, std::set<unsigned int>>, unsigned int, hash_X> IdMap;

class FastKiFDD {
    private:

        bool combination_compatible(unsigned int id1, unsigned int id2);
        void increment_id(void);

        std::map<unsigned int, double> get_active_base_ids(const std::vector<double>& s);
        
        std::map<unsigned int, std::vector<unsigned int>> get_active_base_ids_per_dim(const std::vector<double>& s);
        
        double update_candidate(Candidate& c, double td_error, double phi1, double phi2);

        void add_base(const std::vector<double>& center, unsigned int dim);
        void add_refined(unsigned int idx1, unsigned int idx2, CandidateMap::iterator cit);
    public:
        bool verbose;
        CandidateMap candidates;
        IdMap base_ids_to_id;
        int features_num;
        bool normalization = true;
        double activation_threshold;
        double discovery_threshold;
        double max_neighbor_similarity;
        unsigned int max_active_neighbors;
        std::vector<std::vector<unsigned int> > dims;
        std::vector<std::vector<double> > centers;
        std::vector<std::set<unsigned int> > base_ids;
        std::function< double(std::vector<double>, std::vector<double>, std::vector<unsigned int>)> kernel;
        std::set<std::pair<unsigned int, std::vector<unsigned int>>, cmp_dim> sorted_dim_ids;
        std::set<std::pair<unsigned int, std::vector<unsigned int>>, cmp_spec> sorted_spec_ids;
        int sparsification;



        FastKiFDD(double activation_threshold, double discovery_threshold,
                std::string kernel_spec, std::vector<double> kernel_widths,
                int sparsification, double max_neighbor_similarity, 
                unsigned int max_active_neighbors) {
            if (kernel_spec == "gaussian_kernel") {
                kernel = [kernel_widths] (std::vector<double> s1, std::vector<double> s2, std::vector<unsigned int> dim) { return gaussian_kernel(&s1[0], &s2[0], dim, &kernel_widths[0]); };
            } else if (kernel_spec == "linf_triangle_kernel") {
                kernel = [kernel_widths] (std::vector<double> s1, std::vector<double> s2, std::vector<unsigned int> dim) { return linf_triangle_kernel(&s1[0], &s2[0], dim, &kernel_widths[0]); };
            } else {
                throw std::invalid_argument("No C++ implementation found for this kernel");
            }
            features_num = 0;
            this->sparsification = sparsification;
            this->discovery_threshold = discovery_threshold;
            this->activation_threshold = activation_threshold;
            this->max_neighbor_similarity = max_neighbor_similarity;
            this->max_active_neighbors = max_active_neighbors;
            verbose = false;
        }



        std::vector<double> phi(const std::vector<double>& s);

        unsigned int discover(const std::vector<double>& s, unsigned int a, double td_error, 
                      std::vector<double> previous_phi);

};
