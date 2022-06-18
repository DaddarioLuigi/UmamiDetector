# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import data.dataprocess
from data import *
import models as model


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scelta = -1
    while (scelta != 0):
        print("-------------------------")
        print("0. Uscita")
        print()
        print("1. Predici il valore del glutammato")
        print("2. Classifica un cibo")
        print("3. Effettua una classificazione con HistGradientClassifier")
        print("4. Regressione con HistGradientRegressor")
        print("5. Download reports and plots")
        print("...")
        print("...")
        print()
        scelta = int(input("Scegli: "))
        print("------------------------")
        if (scelta == 0):
            pass
        elif (scelta == 1):
            print("Per predire il valore del glutammato dovrai inserire 23 features. E' possibile utilizzare"
                  "il file README per poter capire come fare.")
        elif (scelta == 2):
            print("Per classificare un cibo dovrai inserire 23 features. E' possibile"
                  "utilizzare il file README per poter capire come fare.")
        elif (scelta == 3):
            print("Puoi utilizzare questa funzione per poter confrontare gli algoritmi di classificazione")
        elif (scelta == 4):
            print("Puoi utilizzare questa funzione per poter confrontare gli algoritmi di regressione")
        elif (scelta == 5):
            print("Reports downloading...")
        else:
            print("Non capisco...")


print("Arrivederci! Torna su UmamiDetector")
        # See PyCharm help at https://www.jetbrains.com/help/pycharm/
