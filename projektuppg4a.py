import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("statdat/SLS22_cleaned.csv")

trickkolumner = ["trick 1", "trick 2", "trick 3", "trick 4"]
grupperat_for_skateboardare = df.groupby("id")

def aposteriorifunktion(täta, data, omega):  # betingat datan
    apriori = stats.t.logpdf(täta, df=4, loc=2, scale=0.3)
    dataförd = np.sum(stats.norm.logpdf(data, loc=täta, scale=omega), axis=0)  #Vi logaritmerar fram och tillbaka för att undvika numeriska fel.
    return apriori + dataförd

färger = ['blue', 'orange', 'green', 'red']  

for skateboardare, grupp in grupperat_for_skateboardare:
    
    poanglistade = grupp[trickkolumner].values.flatten()
    ickenollpoang = [trick for trick in poanglistade if trick > 0]
    
    transformerat = np.log(np.array(ickenollpoang) / (1 - np.array(ickenollpoang)))
    tätastart = np.mean(transformerat)
    omega = np.var(transformerat)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))    #En plott för kedjan, en för löpande medelvärde och en för histogram
    for run in range(4): 
        tätalista = [np.random.normal(tätastart, omega)]
        
        for i in range(9999):
            if skateboardare == "O’neill":   #O'niell hade svårt att uppdatera värden, så vi minskar skalan.
                
                y = tätalista[-1] + stats.uniform.rvs(loc=-0.01, scale=0.02)
            else:
                y = tätalista[-1] + stats.uniform.rvs(loc=-0.1, scale=0.2)
            r = np.exp(aposteriorifunktion(y, transformerat, omega)) / np.exp(aposteriorifunktion(tätalista[-1], transformerat, omega))
            u = np.random.uniform(0, 1)
            if u < min(1, r):
                tätalista.append(y)
            else:
                tätalista.append(tätalista[-1])

        hist = tätalista[-500:]

        
        axs[0].plot(np.linspace(start=1, stop=10000, num=len(tätalista)), tätalista)
        löpande_medelvärde = np.cumsum(tätalista) / np.arange(1, len(tätalista) + 1)
        axs[1].plot(np.linspace(start=1, stop=10000, num=len(tätalista)), löpande_medelvärde)
        
        
        axs[2].hist(hist, bins=50, density=True, alpha=0.5, color=färger[run])

        x = np.linspace(-5, 10, 1000)
        onormaliserat = [np.exp(aposteriorifunktion(xs, transformerat, omega)) for xs in x]
        normeringskonst = np.trapz(onormaliserat, x)
        
        x = np.linspace(min(hist), max(hist), 100)
        y = [np.exp(aposteriorifunktion(xs, transformerat, omega)) / normeringskonst for xs in x]
        axs[2].plot(x, y, color='black', linestyle='--')

    axs[0].set_title(f'X-värden for {skateboardare}')

    axs[1].set_title(f'Löpande Medelvärde for {skateboardare}')    

    axs[2].set_title(f'Histogram {skateboardare}')
    
  

    plt.tight_layout()
    plt.show()
