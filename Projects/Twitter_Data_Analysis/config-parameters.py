from __future__ import division
import sys


"""
It calculates the number of executors, cores and executor memory to be requested for the Assignment C program.
It also calculates the number of files to read in each iteration based on the above parameters.
The number of files to read and the other parameters has to be passed as a command line argument to run the main program.

This program prints out the cluster parameters in one line which can be used as is to run the main program.
For the number of files parameter, it provides a minimum, maximum and recommended. We urge you to use the recommended
number of files as an argument to main program.

Below is the command line arguments to run this code

python config-parameters.py <number of nodes in cluster> <number of cores in each node> <per node memory in GB>

For example:

python config-parameters.py 6 16 64

"""

if len(sys.argv) != 4:
    print("Incorrect number of arguments provided")
    sys.exit()
else:
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    M = int(sys.argv[3])

    cores = min(5, C-2)

    executors = int(N*(C-1)/cores) -1

    X = M/(int(executors/N) + 1)

    memory = min(int(X - (X*0.07)), 64)

    print("\nSpark Conf inputs to run the Assignment C main program\n")

    print("--num-executors {} --executor-cores {} --executor-memory {}G".format(executors, cores, memory))

    print("\nNumber of files to read in each iteration. This has to be given as the last command line argument.\n")

    print("Min {} files and Max {} files\n".format(memory*100, executors*(memory-1)*100))

    print("Recommended number of files to read in each iteration: {}\n".format((executors-1)*(memory-1)*100))
