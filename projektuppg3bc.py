import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("SLS22_cleaned.csv")

trickkolumner = ["trick 1", "trick 2", "trick 3", "trick 4"]

grupperat_for_skateboardare = df.groupby("id")    

all_data = []

for skateboardare, grupp in grupperat_for_skateboardare:
    poanglistade = grupp[trickkolumner].values.flatten()

    p = 0
    ickenollpoang = []

    for trick in poanglistade:
        if trick > 0:
            p += 1
            ickenollpoang.append(trick)

    p = p / len(poanglistade)

    ickenollaray = np.array(ickenollpoang)
    transformerat = np.log(ickenollaray / (1 - ickenollaray))

    täta = np.mean(transformerat)
    omega = np.var(transformerat)
    print(skateboardare, " täta: ", täta, ", omega: ", omega)   #Vi tar enkelt fram skattade värden för täta och omega.

    bernoulli = stats.bernoulli.rvs(p, size=400)
    normal_values = stats.norm.rvs(täta, omega, size=400)

    trick400 = np.where(bernoulli == 1, normal_values, 0)
    trick400[bernoulli == 1] = np.exp(trick400[bernoulli == 1]) / (np.exp(trick400[bernoulli == 1]) + 1)    #Generering av data enligt den hierarkiska modellen.

    tsim = []
    for i in range(100):
        z4 = np.max(trick400[4 * i:4 * i + 4])
        z3 = np.sort(trick400[4 * i:4 * i + 4])[-2]
        tsim.append(z3 + z4)

    print(skateboardare, " stickprovsmedelsvärde: ", np.mean(tsim))

    all_data.append(tsim)

plt.boxplot(all_data, widths=0.8)
plt.xticks(range(1, len(all_data) + 1), grupperat_for_skateboardare.groups.keys(), rotation=90)
plt.title('Boxplots för alla skateboardare')
plt.tight_layout()
plt.show()
