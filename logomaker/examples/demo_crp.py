# import matplotlib and logomaker
import matplotlib.pyplot as plt
import logomaker

# load example dataset provided with logomaker
df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

# render logo
logomaker.Logo(df)

# make sure to show logo
plt.show()
