
# CSCI 420 HW 05
# Lindsay Cagarli
# Anna Kurchenko

import pandas as pd
import numpy as np

def classify_record(record):
    if record['RoofRack'] <= 0.0:
        if record['HasGlasses'] <= 0.0:
            if record['SideDents'] <= 0.0:
                if record['Speed'] <= 67.0:
                    if record['Brightness'] <= 0.0:
                        return 'PULL_OVER'
                    else:
                        if record['BumperDamage'] <= 0.0:
                            if record['Wears_Hat'] <= 0.0:
                                if record['Speed'] <= 66.0:
                                    return 'letpass'
                                else:
                                    return 'PULL_OVER'
                            else:
                                if record['Brightness'] <= 8.0:
                                    return 'PULL_OVER'
                                else:
                                    return 'PULL_OVER'
                        else:
                            if record['Speed'] <= 57.0:
                                if record['Brightness'] <= 1.0:
                                    return 'PULL_OVER'
                                else:
                                    return 'letpass'
                            else:
                                if record['NLaneChanges'] <= 2.0:
                                    return 'PULL_OVER'
                                else:
                                    return 'PULL_OVER'
                else:
                    return 'PULL_OVER'
            else:
                if record['Speed'] <= 56.0:
                    if record['Brightness'] <= 8.0:
                        if record['Brightness'] <= 5.0:
                            if record['HasSpoiler'] <= 0.0:
                                if record['Brightness'] <= 1.0:
                                    return 'PULL_OVER'
                                else:
                                    return 'PULL_OVER'
                            else:
                                return 'letpass'
                        else:
                            return 'letpass'
                    else:
                        return 'PULL_OVER'
                else:
                    return 'PULL_OVER'
        else:
            return 'PULL_OVER'
    else:
        return 'letpass'

def main(filename):
    # read in data
    data = pd.read_csv(filename)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(np.floor)
    
    # Classify each record
    predictions = []
    for _, record in data.iterrows():
        prediction = classify_record(record)
        predictions.append(prediction)
    
    # Save predictions and output 
    predictions_string = ','.join(predictions) 
    with open('HW_05_Kurchenko_Cagarli_MyClassifications.csv', 'w') as f:
        f.write(predictions_string) 

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python HW_05_Kurchenko_Cagarli_Classifier.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
