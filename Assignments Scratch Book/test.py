# file1 = "/Users/monicadabas/Desktop/MSCourseWork/Spring2017/Big_Data-STA9497/2017-STAT-9794-Monica-Dabas/AssignmentA_DataScrubberAndAnalyzer/noise.txt"
# file2 = "/Users/monicadabas/Desktop/MSCourseWork/Spring2017/Big_Data-STA9497/2017-STAT-9794-Monica-Dabas/Assignments Scratch Book/noise_scratch.txt"
# with open(file1,"r") as file1:
#     lines_scrub = file1.readlines()
# with open(file2,"r") as file2:
#     lines_scratch = file2.readlines()
#
# print("Noise in Scrub: {}".format(len(lines_scrub)))
# print("Noise in Scratch: {}".format(len(lines_scratch)))
#
# d = {}
# for line in lines_scratch:
#     if line in d:
#         d[line] += 1
#     else:
#         d[line] = 1
#
# dup = 0
# for key in d:
#     if d[key] > 1:
#         print(key, ",",d[key])
#         dup += 1
#
# print("Duplicate indexes: {}".format(dup))
#
# count = [0]*12
#
# for line in lines_scratch:
#     line.strip("\n")
#     if int(line) < 10000:
#         count[0] += 1
#     elif 9999 < int(line) < 20000:
#         count[1] += 1
#     elif 19999 < int(line) < 20002:
#         count[2] += 1
#     elif 20001 < int(line) < 30002:
#         count[3] += 1
#     elif 30001 < int(line) < 40001:
#         count[4] += 1
#     elif 40000 < int(line) < 50001:
#         count[5] += 1
#     elif 50000 < int(line) < 60000:
#         count[6] += 1
#     elif 59999 < int(line) < 70000:
#         count[7] += 1
#     elif 69999 < int(line) < 80000:
#         count[8] += 1
#     elif 79999 < int(line) < 80003:
#         count[9] += 1
#     elif 80002 < int(line) < 90003:
#         count[10] += 1
#     else:
#         count[11] += 1
#
# print("Scratch noise per category: {}".format(count))
#
# count1 = [0]*12
# for line in lines_scrub:
#     line.strip("\n")
#     if int(line) < 10000:
#         count1[0] += 1
#     elif 9999 < int(line) < 20000:
#         count1[1] += 1
#     elif 19999 < int(line) < 20002:
#         count1[2] += 1
#     elif 20001 < int(line) < 30002:
#         count1[3] += 1
#     elif 30001 < int(line) < 40001:
#         count1[4] += 1
#     elif 40000 < int(line) < 50001:
#         count1[5] += 1
#     elif 50000 < int(line) < 60000:
#         count1[6] += 1
#     elif 59999 < int(line) < 70000:
#         count1[7] += 1
#     elif 69999 < int(line) < 80000:
#         count1[8] += 1
#     elif 79999 < int(line) < 80003:
#         count1[9] += 1
#     elif 80002 < int(line) < 90003:
#         count1[10] += 1
#     else:
#         count1[11] += 1
#
# print("Scrub noise per category: {}".format(count1))
#
# Scratch_not_in_scrub = []
# for i in lines_scratch:
#     if i not in lines_scrub:
#         Scratch_not_in_scrub.append(i)
#
# scrub_not_in_scratch = []
# for i in lines_scrub:
#     if i not in lines_scratch:
#         scrub_not_in_scratch.append(i)
#
# print("Noise not captured by Scrub: {}".format(Scratch_not_in_scrub))
# print(len(Scratch_not_in_scrub))
# print("Extra noise captured by Scrub: {}".format(scrub_not_in_scratch))
# print(len(scrub_not_in_scratch))

# data-big.txt: ticks: 35358866; noise: 3631450
# data-small.txt: ticks: 100000; noise: 307

# 10 nodes
# INFO:root:Time to get blocks: 8.263459
# INFO:root:Time to scrub: 2.252471
# INFO:root:Time taken to write noise.txt: 0.100675
# INFO:root:Total time taken is: 10.616612

# 5 nodes
# INFO:root:Time to get blocks: 0.650874
# INFO:root:Time to scrub: 2.599808
# INFO:root:Time taken to write noise.txt: 0.068569
# INFO:root:Total time taken is: 3.319258

import logging
import sys
from memory_profiler import profile


print(sys.argv)
loglevel = sys.argv[1][6:]
from memory_profiler import profile
print(loglevel)

num_level = getattr(logging, loglevel.upper())
if not isinstance(num_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)

logging.basicConfig(filename='example.log',level=num_level,filemode='w')

logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

with open("memory_logger.log", 'w') as mf:
    @profile(stream=mf)
    def a():
        print("Entered a")
        A = 6
        c = A**2

    @profile(stream=mf)
    def b():
        print("Entered b")


    a()
    b()
