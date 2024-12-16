import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("SLS22_cleaned.csv")

run1poang = df["run 1"].values.flatten()
run2poang = df["run 2"].values.flatten()

grun1poang = np.log(run1poang / (1 - run1poang))     #Vi tar g
grun2poang = np.log(run2poang / (1 - run2poang))

covariansmatris = np.cov(grun1poang, grun2poang)     #Stickprovskovariansmatris
print(covariansmatris)

korrelation = covariansmatris[0][1]/np.sqrt(covariansmatris[0][0])/np.sqrt(covariansmatris[1][1])
print(korrelation)

ttest = np.sqrt(len(grun1poang)-2)*korrelation/np.sqrt(1-korrelation**2)
print(ttest)

alfa = 0.05
                    #Vi jämför T värdet med c-värdet genom att skriva ut båda.

c = stats.t.ppf(1 - alfa/2, 48)
print(c)

grupperat_for_skateboardare = df.groupby("id")

all_data = []

for skatebordare, grupp in grupperat_for_skateboardare:
    run1s = grupp["run 1"].values.flatten()
    run2s = grupp["run 2"].values.flatten()

    run1s = np.log(run1s / (1 - run1s))
    run2s = np.log(run2s / (1 - run2s))

    täta1 = np.mean(run1s)
    sigma = np.cov(run1s, run2s)
    omega1 = sigma[0][0]
    varlambda = sigma[0][1] / omega1
    omega2 = sigma[1][1] - varlambda ** 2 * omega1      #Samma algebra som är härledd i rapporten
    täta2 = np.mean(run2s) - varlambda * täta1

    print(skatebordare, " täta 1: ", täta1, " täta2: ", täta2, " lambda: ", varlambda, " omega1: ", omega1, " omega2: ", omega2)

    r1genererat = stats.norm.rvs(täta1, omega1, size=100)
    r2genererat = varlambda * r1genererat + stats.norm.rvs(täta2, omega2, size=100)

    kombinerat = np.maximum(r1genererat, r2genererat)
    mellannollochett = np.exp(kombinerat) / (np.exp(kombinerat) + 1)     #Vi tar g invers

    all_data.append(mellannollochett)   #Vi använder all_data för att kunna boxplotta allt i samma bild, för synlighet.

    print(skatebordare, " stickprovsmedelvärde: ", np.mean(mellannollochett))

plt.boxplot(all_data, widths=0.8)
plt.xticks(range(1, len(all_data) + 1), grupperat_for_skateboardare.groups.keys(), rotation=90)
plt.title('Boxplots för alla skateboardare')
plt.tight_layout()
plt.show()
