from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Import LogoMaker
import sys
sys.path.append('../')
import logomaker
import inspect
import make_logo


param_dict = logomaker.documentation_parser.parse_documentation_file('make_logo_arguments.txt')


default_values = inspect.getargspec(logomaker.make_logo)
doc_dict = dict(zip(default_values[0], list(default_values[3])))

# form dictionary with default values

sectionSet = set()
doc_dict_2 = {}
param_pairs = [(val.param_num, val.section, val.name,val.description) for val in param_dict.values()]
for num, section, name, description in sorted(param_pairs):
    #print '%d: %s, %s, %s '%(num, section, name,description)
    #print '%d: %s, %s ' % (num, section, name)
    doc_dict_2[name] = (doc_dict[name],description,section)
    # for unique sections
    sectionSet.add(section)

# change to list to access section as elements
sectionList = sorted(list(sectionSet))

sectionIndex = 0    # index to iterate unique sections

sectionDict = {}

# sort by section 1, value, 2 section.
for key, value in sorted(doc_dict_2.items(), key=lambda x: x[1][2]):
    # section matches unique seciton in set, return all parameters associated with it
    if(value[2]==sectionList[sectionIndex]) and sectionIndex<len(sectionList):
        #sectionList[sectionIndex] is going to be button name and id
        # key are going to be table elements
        #print(sectionList[sectionIndex], key, value[0], value[1])

        # new dict with section name as key and values being all the associated parameter names.
        # along with default values and descriptions
        if sectionList[sectionIndex] in sectionDict:
            sectionDict[sectionList[sectionIndex]].append([key, value[0], value[1]])
        else:
            sectionDict[sectionList[sectionIndex]] = [[key, value[0], value[1]]]
    else:
        sectionIndex+=1


for key,value in sectionDict.iteritems():
    # key is section name
    # value[i][0]: (i,0) is parameter name, (i,1) is default value, (i,2) is description
    # value[0] prints first list
    for val in enumerate(value):
        print val[1]
    #print key, value





