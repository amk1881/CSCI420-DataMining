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


# This resulting decision tree classifier program looks something like this.
#    .... header and prologue ...
#    Given one parameter --
#    the string containing the filename to read in.

#    read in the input file
#    for each line in the input_test_data :
#        if ( attribute_a >= threshold_a ) :
#            if ( attribute_b > threshold_b ) :
#                intent = 2; # aggressive.
#            else :
#                intent = 1; # non-agressive.
#        else :
#            if ( attribute_c <= threshold_c ) :
#                intent = 2; # aggressive.
#            else :
#                intent = 1; # non-agressive.
#    ...
        
#    print( intent ). # print out the class value for each line of the provided validation file



#    
# Program Driver
# 
def main():

    if len(sys.argv) > 1:
    
        # getting data
        print("")

    else:
        print("ERR: Missing Test_Suite Directory")

if __name__ == '__main__':
    main()
