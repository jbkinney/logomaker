import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import logomaker

# Create CRP logo from IUPAC motif
iupac_seq = 'WWNTGTGANNNNNNTCACANWW'

# valid dataframe
iupac_mat = logomaker.iupac_to_matrix(iupac_seq, to_type='probability')

#logo = logomaker.Logo(iupac_mat,negate_values=False, colors='RdBu')
#logo = logomaker.Logo(iupac_mat,negate_values=False,colors=(0.1,0.3,0.5))
logo = logomaker.Logo(iupac_mat,negate=False,colors='classic',figsize=[5,2])

random_df  = pd.DataFrame(np.random.randint(-100,100,size=(10, 4)), columns=list('ABCD'))
#logo = logomaker.Logo(random_df,negate_values=-1)


plt.show()