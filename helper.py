import csv
import os

def saveResult(fileName, labels):
    with open(fileName, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["ImageID", "Digit"])
        for i in range(len(labels)):
            writer.writerow([i+1, labels[i]])
