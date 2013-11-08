#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include "FastKiFDD.h"




template <typename T1, typename T2> std::vector<typename std::map<T1, T2>::iterator> is_subset(std::set<T1>& sb, std::map<T1, T2>& ss) {
    std::vector<typename std::map<T1, T2>::iterator> result;
    typename std::map<T1, T2>::iterator i = ss.begin();
    typename std::set<T1>::iterator j=sb.begin();
    while( i != ss.end() && j != sb.end()) {
        if (i->first == *j) {
            result.push_back(i);
            i++;
            j++;
        }
        else if( i->first < *j) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}


void FastKiFDD::increment_id(void) {
    sorted_spec_ids.insert(make_pair(features_num, dims[features_num]));
    sorted_dim_ids.insert(make_pair(features_num, dims[features_num]));
    features_num++;

}

std::map<unsigned int, double> FastKiFDD::get_active_base_ids(const std::vector<double>& s) {
    std::map<unsigned int, double> active_ids;
    for (auto i : sorted_dim_ids) {
        if (i.second.size() > 1)
            break;
        if (kernel(centers[i.first], s, dims[i.first]) > activation_threshold)
            active_ids[i.first] = 1.;
    }
    return active_ids;
}
std::map<unsigned int, std::vector<unsigned int>> FastKiFDD::get_active_base_ids_per_dim(const std::vector<double>& s) {
    std::map<unsigned int, std::vector<unsigned int>> active_ids;
    for (auto i : sorted_dim_ids) {
        //std:: << "ID " << i.first << " in " << i.second.size() << " dimension "<< i.second[0]<< " with a value of  " << kernel(centers[i.first], s, dims[i.first]) << std::endl;
        if (i.second.size() > 1) {
            //std:: << "too many dimensions" << std::endl;
            break;
        }
        if (kernel(centers[i.first], s, dims[i.first]) > activation_threshold)
            active_ids[i.second[0]].push_back(i.first);
    }
    return active_ids;
}

double FastKiFDD::update_candidate(Candidate& c, double td_error, double phi1, double phi2) {
    c.total_activation += pow(phi1 * phi2, 2);
    c.total_error += phi1 * phi2 * td_error;
    c.relevance = abs(c.total_error) / sqrt(c.total_activation);
    return c.relevance;
}


void FastKiFDD::add_base(const std::vector<double>& center, unsigned int dim) {
    // adds a new 1-dimensional base feature
    dims.push_back({dim });
    centers.push_back(center);
    std::set <unsigned int> s = {(unsigned int)features_num };
    base_ids.push_back(s);
    base_ids_to_id[make_pair(s, s)] = features_num;
    increment_id();

    //TODO add candidates
    for (int i=0; i < features_num - 1; i++) {
        if (find(dims[i].begin(), dims[i].end(), dim) == dims[i].end()) {
            // different dimensions --> combination reasonable, add candidate
            candidates.insert(make_pair(make_pair(base_ids[i], s), Candidate()));
        }
    }
    if (verbose)
        std::cout << "New base feature " << features_num - 1 << " in dimension " << dim << "; " << candidates.size() << " Candidates in total" << std::endl;
}

std::vector<unsigned int> FastKiFDD::filter_ids(std::map<unsigned int, double> active_ids, const std::vector<double>& s) {
    std::vector<unsigned int> cur_feat_ids;
    for (auto in : sorted_spec_ids) {
        unsigned int i = in.first;
        auto pos = is_subset<unsigned int, double>(base_ids[i], active_ids);
        if (pos.size() != base_ids[i].size()) {
            continue;
        }
        bool good = true;
        for (unsigned int j : cur_feat_ids) {
            if (base_ids[j].size() == base_ids[i].size())
                break;
            if (std::includes(base_ids[j].begin(), base_ids[j].end(),
                  base_ids[i].begin(), base_ids[i].end())) {
                good = false;
                break;
            }
            
        }
        if (good && (sparsification > 10 || kernel(centers[i], s, dims[i]) > activation_threshold))
            cur_feat_ids.push_back(i);
    }
    return cur_feat_ids;
}
std::vector<double> FastKiFDD::phi(const std::vector<double>& s) {
    std::vector<double> output = std::vector<double>(features_num, 0);
    double out_sum = 0;
    if (sparsification > 0) {
        std::map<unsigned int, double> active_ids = get_active_base_ids(s);
        if (sparsification == 10) {
            auto cur_ids = filter_ids(active_ids, s);
            for (unsigned int i : cur_ids) {
                output[i] = kernel(centers[i], s, dims[i]);
                out_sum += abs(output[i]);
            }
        } else {

        // iterate from most specific to most general feature
        // O(n*k*k)
        for (auto i : sorted_spec_ids) {
            unsigned int id = i.first;
            // O(n*k)
            auto pos = is_subset<unsigned int, double>(base_ids[id], active_ids);
            // O(k)
            if (pos.size() == base_ids[id].size()) {
                // we really have a subset
                double value = kernel(centers[id], s, dims[id]);
                if (sparsification > 2 || value > activation_threshold) {
                    if (sparsification > 1) {
                        // proper spasification, with non-smooth
                        // effects
                        output[id] = value;
                        out_sum += abs(value);
                        for (auto p : pos)
                            active_ids.erase(p);
                    } else {
                        // hacks somewhat smooth sparsification
                        double u = 0.;
                        for (auto p : pos)
                            u = std::max(u, p->second);
                        output[id] = u * value;
                        out_sum += abs(output[id]);
                        for (auto p : pos) {
                            p->second = std::min(p->second, output[id]);
                            if (p->second <= 0.)
                                active_ids.erase(p);
                        }
                    }
                }
            }
        }}

    } else {
        // O(n)
        for (int i=0; i < features_num; i++) {
            output[i] = kernel(centers[i], s, dims[i]);
            out_sum += abs(output[i]);
        }
    }
    // O(n)
    if (normalization and out_sum != 0.) {
        for (int i=0; i < features_num; i++)
            output[i] = output[i] / out_sum;
    }
    return output;

}

bool FastKiFDD::combination_compatible(unsigned int id1, unsigned int id2) {

    auto i = dims[id1].cbegin();
    auto j = dims[id2].cbegin();
    while (i != dims[id1].cend() && j != dims[id2].cend()) {
        if (*i == *j) {
            if (centers[id1][*i] != centers[id2][*j])
                return false;
            i++; j++;
        } else if (*i < *j) {
            i++;
        } else {
            j++;
        }
    }
    return true;

}

void FastKiFDD::add_refined(unsigned int idx1, unsigned int idx2, CandidateMap::iterator cit) {
    // adds a combination of existing features
    std::vector<unsigned int> new_dim = std::vector<unsigned int>(dims[idx1].size() + dims[idx2].size());
    merge(dims[idx1].cbegin(), dims[idx1].cend(), 
          dims[idx2].cbegin(), dims[idx2].cend(), new_dim.begin());
    dims.push_back(new_dim);
    
    std::set<unsigned int> new_ids;
    new_ids.insert(base_ids[idx1].cbegin(), base_ids[idx1].cend());
    new_ids.insert(base_ids[idx2].cbegin(), base_ids[idx2].cend());
    base_ids.push_back(new_ids);
    
    base_ids_to_id[make_pair(base_ids[idx1], base_ids[idx2])] = features_num;
    
    std::vector<double> center = std::vector<double>(centers[idx1]);
    for (unsigned int i : dims[idx2]) {
        center[i] = centers[idx2][i];
    }
    centers.push_back(center);
    increment_id();
    // delete candidate
    candidates.erase(cit);
    if (verbose) {
        std::cout << "New refined feature " << features_num - 1 << " as combo of ";
        for (auto id : new_ids)
            std::cout << id << ", "; 
    }


    // add new candidates
    for (unsigned int i=0; i < (unsigned int)features_num - 1; i++) {
        CandidateMap::iterator cit = candidates.find(make_pair(base_ids[i], new_ids));
        if (cit != candidates.end()) // already a candidate
            continue;
        IdMap::iterator iit = base_ids_to_id.find(make_pair(base_ids[i], new_ids));
        if (iit != base_ids_to_id.end()) // already a feature
            continue;

        // make sure the feature combination shares the same base ids in
        // overlapping dimensions
        if (combination_compatible(i, features_num - 1)) {
            candidates[make_pair(base_ids[i], new_ids)] = Candidate();
        }

    }
    if (verbose)
        std::cout << "; " << candidates.size() << " Candidates in total" << std::endl;
    

}

unsigned int FastKiFDD::discover(const std::vector<double>& s, unsigned int a, double td_error, 
                std::vector<double> previous_phi) {
    assert(features_num == sorted_dim_ids.size());
    unsigned int discovered = 0;
    std::map<unsigned int, std::vector<unsigned int>> active_ids = get_active_base_ids_per_dim(s);
    // add new base features
    for (unsigned int d=0; d < s.size(); d++) {
        //std::cout << "Dimension" << d << "; " << active_ids[d].size() << " neighbors of " << max_active_neighbors << std::endl;
        if (active_ids[d].size() >= max_active_neighbors) {
            continue;
        }
        bool too_close = false;
        for (auto id : active_ids[d]) {
            if (kernel(centers[id], s, dims[id]) > max_neighbor_similarity) {
                too_close = true;
                break;
            }
            
        }
        if (too_close)
            continue;
        add_base(s, d);
        discovered++;
    }
    if (discovered > 0)
        previous_phi = phi(s);
    
    // update relevances and maybe expand
    for (auto i=active_ids.cbegin(); i != active_ids.cend(); i++) {
        for(unsigned int id1 : i->second) {
            for (auto j=i; j !=active_ids.cend(); j++) {
                if (i == j) continue;
                for(unsigned int id2 : j->second) {
                    // all possible combinations of active features in
                    // id1 and id2
                    CandidateMap::iterator cit = candidates.find(make_pair(base_ids[id1], base_ids[id2]));
                    if (cit == candidates.end())
                        continue;
                    double relevance = update_candidate(cit->second, td_error, previous_phi[id1], previous_phi[id2]);
                    if (relevance > discovery_threshold) {
                        // expand feature
                        add_refined(id1, id2, cit);
                        discovered++;
                    }
                }
            }
        }
    }

    return discovered;

}

