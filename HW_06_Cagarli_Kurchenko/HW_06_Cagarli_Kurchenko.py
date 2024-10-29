# CSCI 420
# HW 6
#
# Lindsay Cagarli
# Anna Kurchenko
#

import math
import matplotlib.pyplot as plt
import numpy as np

# NOTE: There is an extra, across the board, 
# 25% penalty in this assignment for code that 
# cannot be easily read

FILENAME = "HW_06_Cagarli_Kurchenko/HW_CLUSTERING_SHOPPING_CART_v2241a.csv"

def get_data():
    """
    Returns the data in the csv file.
    """
    return np.genfromtxt(FILENAME, delimiter=',', skip_header=1)


def agglomeration():
    print("agglomerate")


def create_dendrogram():
    print("dendrogram")
    

def main():
    print("hw6")

if __name__ == "__main__":
    main()
