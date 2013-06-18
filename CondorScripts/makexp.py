#!/usr/bin/python
# Alborz Geramifard March 19th 2013 MIT
# This script prepares directories and main files for various parameters in one shot
# Inputs:
# dictionary of parameters
# agent: agent name = {SARSA, Q-LEARNING, LSPI}
# a0: Boyan alpha decay parameters = {.1,1}
# N0: Boyan alpha decay parameters = {100,1000}
# iFDD-T: iFDD threshold
import os, sys, time, re, string
#find RL_ROOT and add it to path
from Script_Tools import *
RL_PYTHON_ROOT = findRLRoot()
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import perms,decimals
#Mapping from each token name to their corresponding parameter in the text file in the main.py. Some tokens may have more than one corresponding parameter
TOKEN_DICT={'T':['iFDDOnlineThreshold','BatchDiscoveryThreshold'], 'a':['initial_alpha'], 'N':['boyan_N0'], 'agent':['agent'], 'rep':['representation'], 'domain':['domain'], 'solver':['MDPSolver'], 'rbfs':['RBFS']}
TOKEN_OBJECTS = ['domain','rep','agent','solver']
def iscomment(str):
    stripped = str.strip()
    return len(stripped) == 0 or stripped[0] == '#'
def adjustFile(src,des,tokens,values):
    # Adjust the mainFile to set the param to value
    done_tokens = []
    for line in src:
        eq_index = line.find('=')
        if eq_index == -1:
            des.write(line)
        else:
            addedLine = False
            for token_index,token in enumerate(tokens):
                value = values[token_index]
                for param in TOKEN_DICT[token]:
                    param_index = line.find(param,0,eq_index)
                    if param_index != -1:
                        if token in TOKEN_OBJECTS:
                            # uncomment the correct one
                            after_eq_token = line[eq_index:line.find('(',eq_index)].strip(' =')
                            if after_eq_token == value:
                                #This is the correct line. Uncomment it
                                des.write('##################################################\n')
                                des.write('## Edited by makexp.py script\n')
                                first_hash = line.find('#')
                                if first_hash != -1:
                                    des.write(line[0:first_hash]+line[first_hash+1:])
                                else:
                                    des.write(line)
                                des.write('##################################################\n')
                                addedLine = True
                                done_tokens.append(token)
                            else:
                                # Check to see this line is commented.
                                if not iscomment(line):
                                    des.write('#'+line)
                                    addedLine = True
                        else:
                            done_tokens.append(token)
                            initial_tab,_,_ = line.partition(param)
                            newline = initial_tab + param + ' =\t' + str(value) +' # Edited by makexp.py script\n'
                            des.write('##################################################\n')
                            des.write('## Edited by makexp.py script\n')
                            des.write(newline)
                            des.write('##################################################\n')
                            addedLine = True
            if not addedLine:
                des.write(line)
    remainingTokens = set(tokens) - set(done_tokens)
    if len(remainingTokens):
            print 'WARNING: Not all tokens were used during directory generation. Check spellings for:', list(remainingTokens)
    des.close()
if __name__ == '__main__':
    
    os.system('clear');

    print('*********************************************************************');    
    print('***************** Creating directories and jobs *********************');    
    print('*********************************************************************');     
    
    if len(sys.argv) == 1:
        # simply copy the main.py here
        os.system('rm -rf main.py; cp %s/main.py .' % RL_PYTHON_ROOT)
        print '>>>> Copied main file here.'
    else:
        dict = {}   # Dictionary mapping each parameter name to its list of values
        parameters = [] #List of parameters in string
        counts = [] # Number of options for each parameter
        for i,e in enumerate(sys.argv):
            if i == 0: continue
            p,_,v = e.rpartition('=')
            p = p.strip() #Remove spaces
            v = v.strip() #Removes spaces
            if v[0] != '[': v = '['+v+']' # If not passed as a list at brackets around it
            
            # If inputs are strings add correct quote around them
            if v[1].isalpha():
                v = v.replace('[','[\"')
                v = v.replace(']','\"]')
                v = v.replace(',','\",\"')
            dict[p] = eval(v)
            parameters.append(p)
            counts.append(len(eval(v)))
        permutations = perms(counts).astype(int)
        for permutation in permutations:
            dir_name = ''
            #1. Make Directory Name
            for param_index, value_index in enumerate(permutation):
                if param_index != 0: dir_name += '-'
                param = parameters[param_index]
                value = dict[param][value_index]
                if param in TOKEN_OBJECTS:
                    dir_name += value 
                elif value == int(value):
                    dir_name += '%s%d' % (param,int(value))
                else:
                    dir_name += '%s%%0.%df' % (param,decimals(value))
                    dir_name = dir_name % value
            if not os.path.exists(dir_name): os.mkdir(dir_name)
            print 'Created: %s' % dir_name
            #os.system('cp %s/main.py %s' % (RL_PYTHON_ROOT, dir_name))
            main_file_orig = open('%s/main.py' % (RL_PYTHON_ROOT), 'r')
            output_f =  open('%s/main.py' % dir_name, 'w')
            values = [dict[parameters[i]][value_index] for i,value_index in enumerate(permutation)]
            adjustFile(main_file_orig,output_f,parameters,values)
            main_file_orig.close()           
            
            
            
            
        