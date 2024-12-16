import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("statdat/SLS22_cleaned.csv")    #Vi gör mycket av samma här. Men inga plottar och bara en kedja, med vilken vi tar fram aposterioriprediktiv data.

trickkolumner = ["trick 1", "trick 2", "trick 3", "trick 4"]
grupperat_for_skateboardare = df.groupby("id")

def aposteriorifunktion(täta, data, omega):  # betingat datan
    apriori = stats.t.logpdf(täta, df=4, loc=2, scale=0.3)
    dataförd = np.sum(stats.norm.logpdf(data, loc=täta, scale=omega), axis=0)
    return apriori + dataförd



alla_betygB = []
alla_labels = []

for skateboardare, grupp in grupperat_for_skateboardare:
    
    poanglistade = grupp[trickkolumner].values.flatten()
    ickenollpoang = [trick for trick in poanglistade if trick > 0]
    
    transformerat = np.log(np.array(ickenollpoang) / (1 - np.array(ickenollpoang)))
    tätastart = np.mean(transformerat)
    omega = np.var(transformerat)

    tätalista = [np.random.normal(tätastart, omega)]
    
    for i in range(9999):
        if skateboardare == "O’neill":
            y = tätalista[-1] + stats.uniform.rvs(loc=-0.01, scale=0.02)
        else:
            y = tätalista[-1] + stats.uniform.rvs(loc=-0.1, scale=0.2)
        r = np.exp(aposteriorifunktion(y, transformerat, omega)) / np.exp(aposteriorifunktion(tätalista[-1], transformerat, omega))
        u = np.random.uniform(0, 1)
        if u < min(1, r):
            tätalista.append(y)
        else:
            tätalista.append(tätalista[-1])

    genereradetäta = tätalista[-400:]
    p = len(ickenollpoang) / len(poanglistade)
    bernoulli = stats.bernoulli.rvs(p, size=400)

    genereratstickprov = [np.random.normal(täta, omega) for täta in genereradetäta]

    trick400 = np.where(bernoulli == 1, genereratstickprov, 0)
    trick400[bernoulli == 1] = np.exp(trick400[bernoulli == 1]) / (np.exp(trick400[bernoulli == 1]) + 1)

    tsim = []
    for i in range(100):
        z4 = np.max(trick400[4 * i:4 * i + 4])
        z3 = np.sort(trick400[4 * i:4 * i + 4])[-2]
        tsim.append(z3 + z4)

    run1s = grupp["run 1"].values.flatten()
    run2s = grupp["run 2"].values.flatten()

    run1s = np.log(run1s / (1 - run1s))
    run2s = np.log(run2s / (1 - run2s))

    täta1 = np.mean(run1s)
    sigma = np.cov(run1s, run2s)
    omega1 = sigma[0][0]
    varlambda = sigma[0][1] / omega1
    omega2 = sigma[1][1] - varlambda ** 2 * omega1
    täta2 = np.mean(run2s) - varlambda * täta1

    r1genererat = stats.norm.rvs(täta1, omega1, size=100)
    r2genererat = varlambda * r1genererat + stats.norm.rvs(täta2, omega2, size=100)

    kombinerat = np.maximum(r1genererat, r2genererat)

    mellannollochett = np.exp(kombinerat) / (np.exp(kombinerat) + 1)

    betygB = mellannollochett + tsim

    print(f"Stickprovsmedelvärde för {skateboardare}: {np.mean(betygB)}")

    
    alla_betygB.append(betygB)
    alla_labels.append(skateboardare)


plt.figure(figsize=(10, 6))
plt.boxplot(alla_betygB, widths=0.8)


plt.xticks(range(1, len(alla_labels) + 1), alla_labels, rotation=45)

plt.title("Boxplots för alla skateboardare")
plt.tight_layout()
plt.show()
