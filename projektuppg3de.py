import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv("SLS22_cleaned.csv")

trickkolumner = ["trick 1", "trick 2", "trick 3", "trick 4"] #Vi har typ samma kod här, men vi klistrar in kod från uppg2 som utökning (för att få run-betyg)

grupperat_for_skateboardare = df.groupby("id")      #Inget nytt här egentligen. Vi bara lägger ihop kod från 3 och 2.

betygB_list = []
skateboardare_list = []

for skateboardare, grupp in grupperat_for_skateboardare:
    poanglistade = grupp[trickkolumner].values.flatten()
    
    p = 0
    ickenollpoang = []

    for trick in poanglistade:
        if trick > 0:
            p += 1
            ickenollpoang.append(trick)
    
    p = p/len(poanglistade)

    ickenollaray = np.array(ickenollpoang)

    transformerat = np.log(ickenollaray/(1-ickenollaray))

    täta = np.mean(transformerat)
    omega = np.var(transformerat)

    bernoulli = stats.bernoulli.rvs(p, size=400)
    normal_values = stats.norm.rvs(täta, omega, size=400)

    trick400 = np.where(bernoulli == 1, normal_values, 0)
    trick400[bernoulli == 1] = np.exp(trick400[bernoulli == 1]) / (np.exp(trick400[bernoulli == 1]) + 1)

    tsim = []
    for i in range(100):
        z4 = np.max(trick400[4*i:4*i+4])
        z3 = np.sort(trick400[4*i:4*i+4])[-2]
        tsim.append(z3+z4)

    run1s = grupp["run 1"].values.flatten()
    run2s = grupp["run 2"].values.flatten()

    run1s = np.log(run1s/(1-run1s))
    run2s = np.log(run2s/(1-run2s))

    täta1 = np.mean(run1s)
    sigma = np.cov(run1s, run2s)
    omega1 = sigma[0][0]
    varlambda = sigma[0][1] / omega1
    omega2 = sigma[1][1] - varlambda**2 * omega1
    täta2 = np.mean(run2s) - varlambda * täta1

    r1genererat = stats.norm.rvs(täta1, omega1, size=100)
    r2genererat = varlambda * r1genererat + stats.norm.rvs(täta2, omega2, size=100)

    kombinerat = np.maximum(r1genererat, r2genererat)

    mellannollochett = np.exp(kombinerat) / (np.exp(kombinerat) + 1)

    betygB = mellannollochett + tsim

    betygB_list.append(betygB)
    skateboardare_list.append(skateboardare)

    print(skateboardare, " stickprovsmedelvärde: ", np.mean(betygB))


plt.figure(figsize=(12, 8))
plt.boxplot(betygB_list, widths=0.8, labels=skateboardare_list)
plt.title("Jämförelse av betyg B för alla skateboardare")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
