import os

from spam import *


def main():
    trainDir = os.path.join(os.path.curdir, "train-mails")
    testDir = os.path.join(os.path.curdir, "test-mails")

    naiveBayesClassi(trainDir, testDir)

    return 0

if __name__ == "__main__":
    main()