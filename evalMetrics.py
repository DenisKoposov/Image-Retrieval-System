import numpy as np

def ErrorRateAt95Recall(labels, distances):
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    #print(np.cumsum(labels))
    #print(recall_point * np.sum(labels))
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))
    #print("thrshld", threshold_index)
    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    #print("FP", FP)
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    #print("TN", TN)
    return float(FP) / float(FP + TN) * 100.0