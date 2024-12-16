import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SLS22_cleaned.csv")


trickkolumner = ["trick 1", "trick 2", "trick 3", "trick 4"]

trickpoang = df[trickkolumner].values.flatten()      #Vi isolerar kolumner som beskriver trickpoäng

plt.hist(trickpoang, bins=10, edgecolor='black')
plt.title("Frekvens av poängen")                #Plot visar att poängen är antingen noll eller runt 0.8.


plt.show()


grupperat_for_skateboardare = df.groupby("id")



vantevardenbeta = {}
for skateboardare, grupp in grupperat_for_skateboardare:                #Slingare för att hantera alla skateboardare separat
    poanglistade = grupp[trickkolumner].values.flatten()
    p = 0
    ickenollpoang = []
    
    for trick in poanglistade:
        if trick > 0:
            p += 1
            ickenollpoang.append(trick)

    m1 = np.mean(ickenollpoang)
    m2 = sum(trickp**2 for trickp in ickenollpoang)/len(ickenollpoang)
    
    alfa = (m1**2 *(1-m1))/(m2-m1**2)-m1     #Skattning av alfa och beta med momentmetoden, algebra finns i pdf.
    beta = alfa*(1-m1)/m2
    print(skateboardare, " fick skatting av p: ", p/len(poanglistade))    #Enkel skattning av p
    print(skateboardare, " fick skattning av alfa: ", alfa)
    print(skateboardare, " fick skattning av beta: ", beta)
    
    vantevardenbeta[skateboardare] = alfa/(alfa+beta)


sorteradevantevarden = sorted(vantevardenbeta.items(), key=lambda x: x[1], reverse=True)   #Vi beräknar väntevärdet med de skattade värdena för alfa och beta. 
                                                                                           #(ML skattningens invariansegenskap)
for namn, varde in sorteradevantevarden:
    print(namn, varde)