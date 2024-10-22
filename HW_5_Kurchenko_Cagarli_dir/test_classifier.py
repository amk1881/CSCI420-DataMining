#
# CSCI 420 HW05
# Classifier
#
# Lindsay Cagarli 
# Anna Kurchenko
# 

import os
import sys
import numpy as np

def classify(data):
    for row in range(len(data)):
        if data[row][0] < 60.0:
            if data[row][1] < 2.0:
                if data[row][0] < 59.0:
                    if data[row][1] < 1.0:
                        if data[row][2] < 9.0:
                            return 0.0
                        else:
                            return 0.0
                    else:
                        if data[row][2] < 9.0:
                            return 0.0
                        else:
                            return 0.0
                else:
                    if data[row][2] < 8.0:
                        if data[row][1] < 1.0:
                            return 0.0
                        else:
                            return 0.0
                    else:
                        if data[row][2] < 9.0:
                            return 0.0
                        else:
                            return 0.0
            else:
                if data[row][2] < 9.0:
                    if data[row][0] < 59.0:
                        if data[row][2] < 2.0:
                            return 2.0
                        else:
                            return 0.0
                    else:
                        if data[row][2] < 3.0:
                            return 2.0
                        else:
                            return 1.0
                else:
                    if data[row][1] < 3.0:
                        if data[row][0] < 59.0:
                            return 0.0
                        else:
                            return 2.0
                    else:
                        if data[row][0] < 57.0:
                            return 0.0
                        else:
                            return 2.0
        else:
            if data[row][2] < 9.0:
                if data[row][2] < 7.0:
                    if data[row][1] < 1.0:
                        if data[row][0] < 65.0:
                            return 1.0
                        else:
                            return 2.0
                    else:
                        if data[row][0] < 63.0:
                            return 1.0
                        else:
                            return 2.0
                else:
                    if data[row][0] < 61.0:
                        if data[row][2] < 8.0:
                            return 1.0
                        else:
                            return 1.0
                    else:
                        if data[row][0] < 66.0:
                            return 1.0
                        else:
                            return 1.0
            else:
                if data[row][0] < 61.0:
                    if data[row][1] < 1.0:
                        if data[row][2] < 10.0:
                            return 0.0
                        else:
                            return 0.0
                    else:
                        if data[row][1] < 3.0:
                            return 2.0
                        else:
                            return 2.0
                else:
                    if data[row][1] < 1.0:
                        if data[row][2] < 10.0:
                            return 2.0
                        else:
                            return 2.0
                    else:
                        if data[row][1] < 2.0:
                            return 2.0
                        else:
                            return 2.0

# Program driver
def main():
    if len(sys.argv) > 1:
        # Load validation data
        validation_data = np.loadtxt(sys.argv[1], delimiter=',')
        for row in range(len(validation_data)):
            intent = classify(validation_data)
            print(intent)
    else:
        print("ERR: Missing Test_Suite Directory")

if __name__ == '__main__':
    main()
