#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "FastKiFDD.h"

using namespace std;

double gaussian_kernel(const double* s1, const double* s2, 
                    const vector<unsigned int>& dim, 
                    const double* widths) {
    double exponent = 0;
    for (unsigned int d : dim) {
        exponent += - pow((s1[d] - s2[d]) / widths[d], 2);
    }
    return exp(exponent);

}

double linf_triangle_kernel(const double* s1, const double* s2, 
                    const vector<unsigned int>& dim, 
                    const double* widths) {
    double res = 1, r;
    for (unsigned int d : dim) {

        r = 1 - abs(s1[d] - s2[d]) / widths[d];
        if (r <= 0) 
            return 0;
        else 
            res = fmin(r, res);
    }
    return res;

}

template <typename T1, typename T2> vector<typename map<T1, T2>::iterator> is_subset(set<T1>& sb, map<T1, T2>& ss) {
    vector<typename map<T1, T2>::iterator> result;
    typename map<T1, T2>::iterator i = ss.begin();
    typename set<T1>::iterator j=sb.begin();
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

map<unsigned int, double> FastKiFDD::get_active_base_ids(const vector<double>& s) {
    map<unsigned int, double> active_ids;
    for (auto i : sorted_dim_ids) {
        if (i.second.size() > 1)
            break;
        if (kernel(centers[i.first], s, dims[i.first]) > activation_threshold)
            active_ids[i.first] = 1.;
    }
    return active_ids;
}
map<unsigned int, vector<unsigned int>> FastKiFDD::get_active_base_ids_per_dim(const vector<double>& s) {
    map<unsigned int, vector<unsigned int>> active_ids;
    for (auto i : sorted_dim_ids) {
        //cout << "ID " << i.first << " in " << i.second.size() << " dimension "<< i.second[0]<< " with a value of  " << kernel(centers[i.first], s, dims[i.first]) << endl;
        if (i.second.size() > 1) {
            //cout << "too many dimensions" << endl;
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


void FastKiFDD::add_base(const vector<double>& center, unsigned int dim) {
    // adds a new 1-dimensional base feature
    dims.push_back({dim });
    centers.push_back(center);
    set <unsigned int> s = {(unsigned int)features_num };
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
        cout << "New base feature " << features_num - 1 << " in dimension " << dim << "; " << candidates.size() << " Candidates in total" << endl;
}

vector<double> FastKiFDD::phi(const vector<double>& s) {
    vector<double> output = vector<double>(features_num, 0);
    double out_sum = 0;
    if (sparsification > 0) {
        map<unsigned int, double> active_ids = get_active_base_ids(s);
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
                            u = max(u, p->second);
                        output[id] = u * value;
                        out_sum += abs(output[id]);
                        for (auto p : pos) {
                            p->second = min(p->second, output[id]);
                            if (p->second <= 0.)
                                active_ids.erase(p);
                        }
                    }
                }
            }
        }

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
    vector<unsigned int> new_dim = vector<unsigned int>(dims[idx1].size() + dims[idx2].size());
    merge(dims[idx1].cbegin(), dims[idx1].cend(), 
          dims[idx2].cbegin(), dims[idx2].cend(), new_dim.begin());
    dims.push_back(new_dim);
    
    set<unsigned int> new_ids;
    new_ids.insert(base_ids[idx1].cbegin(), base_ids[idx1].cend());
    new_ids.insert(base_ids[idx2].cbegin(), base_ids[idx2].cend());
    base_ids.push_back(new_ids);
    
    base_ids_to_id[make_pair(base_ids[idx1], base_ids[idx2])] = features_num;
    
    vector<double> center = vector<double>(centers[idx1]);
    for (unsigned int i : dims[idx2]) {
        center[i] = centers[idx2][i];
    }
    centers.push_back(center);
    increment_id();
    // delete candidate
    candidates.erase(cit);
    if (verbose) {
        cout << "New refined feature " << features_num - 1 << " as combo of ";
        for (auto id : new_ids)
            cout << id << ", "; 
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
        cout << "; " << candidates.size() << " Candidates in total" << endl;
    

}

unsigned int FastKiFDD::discover(const vector<double>& s, unsigned int a, double td_error, 
                vector<double> previous_phi) {
    assert(features_num == sorted_dim_ids.size());
    unsigned int discovered = 0;
    map<unsigned int, vector<unsigned int>> active_ids = get_active_base_ids_per_dim(s);
    // add new base features
    for (unsigned int d=0; d < s.size(); d++) {
        //cout << "Dimension" << d << "; " << active_ids[d].size() << " neighbors of " << max_active_neighbors << endl;
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

