import time
import subprocess

rootLoc = "Misc/Benchmarks"
benchmarks = ["QLearningAgentTF1.py", "QLearningAgentTF2.py",
              "QLearningAgentPyTorch.py"]


def benchmark():
    global benchmarkResults
    benchmarkResults = {}
    for test in benchmarks:
        fileLoc = "/".join([rootLoc, test])
        start = time.time()
        subprocess.call(["python", fileLoc])
        end = time.time()
        benchmarkResults[test] = {}
        benchmarkResults[test]["StartTime"] = start
        benchmarkResults[test]["EndTime"] = end
        benchmarkResults[test]["Duration"] = end-start


def printResults():
    for result in benchmarkResults:
        printStr = "{} took {:.3} seconds".format(result, benchmarkResults[result]["Duration"])
        print(printStr)


if __name__ == "__main__":
    benchmark()
    printResults()
