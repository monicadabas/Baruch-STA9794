from mpi4py import MPI
import numpy as np
import pandas as pd
from itertools import islice
import sys
import logging
from collections import Counter


# declaration of log file

logging.basicConfig(filename ="Scrub_log.log", level= logging.DEBUG, filemode='w')


# definition of block class so store the start and end character index of a block
# block is the part of file to be processed by a node

class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# returns number of ticks or rows in each block (each node processes one block)

def get_line_count(fh,block):
    buffer_size = block.end + 1 - block.start
    buffer = np.empty(buffer_size, dtype=str)

    count = 0
    fh.Seek(block.start)
    fh.Read([buffer, MPI.CHAR])

    for i in range(buffer_size):
        if buffer[i] == "\n":
            count += 1

    return count


# returns adjusted block start and end and the number of ticks in adjusted block

def adjust_blocks(fh,block,rank, nprocs):
    buffer_size = 100
    buffer = np.empty(buffer_size, dtype=str)
    error = fh.Set_errhandler(MPI.ERRORS_RETURN)

    # adjust the block start
    if not(rank == 0):
        fh.Seek(block.start)
        if error is not None:
            print("Could not seek file")
            MPI.Finalize()

        fh.Read([buffer, MPI.CHAR])
        if error is not None:
            print("Could not read file")
            MPI.Finalize()

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.start += i + 1
                break

    # adjust the block end
    if not(rank == nprocs-1):
        fh.Seek(block.end)
        if error is not None:
            print("Call to seek file failed")
            MPI.Finalize()

        fh.Read([buffer, MPI.CHAR])
        if error is not None:
            print("Call to read file failed")
            MPI.Finalize()

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.end += i
                break

    print("Process: {}, Adjusted block: [{}, {}]".format(rank,block.start,block.end))
    count = get_line_count(fh,block)
    print("Process: {}, Line Count: {}".format(rank,count))

    return block, count


# function to return timestamp string to datetime format or returns false if not in right format

def date_parse(s):
    try:
        return pd.datetime.strptime(s, '%Y%m%d:%H:%M:%S.%f')
    except ValueError:
        return "Error"


# Ticks class with 4 features (3 from the data given and 4th as index)

class Ticks:
    def __init__(self, timestamp, price, units, index):
        self. timestamp = timestamp
        self.price = price
        self.units = units
        self.index = index


# checks if the row tick has all values and in right format. If not returns False and raw tick,
# else returns True and raw tick converted to a list of formatted attributes

def check_format(tick):
    line = tick.split(",")
    if len(line) != 3:
        return False, tick, "Length less than 3"
    try:
        line[1] = float(line[1])
        line[2] = int(line[2])
    except ValueError:
        return False, tick, "price or Unit not float/ int"

    if line[1] <= 0 or line[2] <= 0:
        return False, tick, "price or unit <= 0"

    line[0] = date_parse(line[0])
    if line[0] == "Error":
        return False, tick, "timestamp not in format"

    return True, line, "All in format"


class IsValidResult:
    def __init__(self, bool, timestamp, reason):
        self.bool = bool # boolean False is noise else True
        self.timestamp = timestamp # latest timestamp to compare future ticks
        self.reason = reason # reason if tick is a noise else it is All good


# The function checks if the tick with all data in right format is a noise or not
# Checks if timestamp is within 3 seconds of the timestamp of last identified signal

def isvalid(tick,t):

    if abs(tick.timestamp-t).total_seconds() <= 3:
        return IsValidResult(True, tick.timestamp,"All good")

    else:
        #logging.info("Timestamps is more than 3 seconds away from earlier timestamp")
        return IsValidResult(False,t,"More than 3 seconds difference")


""" identify noise and write its index in a list, returns the list to main function
Arguments it takes are:
file is data.txt
block is object of Block class with first and last character index of the part of file to be processed by the node
first_index is the index of the first row in the block which is its index in data.txt
count is the number of rows in this block"""


def identify_noise(file, block, first_index, count):

    with open(file, 'r') as fh:
        noise_list = []
        fh.seek(block.start)
        total = 0  # number of characters of the block read
        current_line_index = first_index
        t = date_parse("00010101:00:00:00.000000")  # default initial timestamp for 1st comparison
        while count > 0:
            print("initial timestamp is {} and index is {}".format(t,current_line_index))
            buffer_size = min(10000, count)  # number of rows to be processed at a time
            count -= buffer_size
            lines = islice(fh, buffer_size)
            data = []
            for tick in lines:
                total += len(tick)
                is_valid_bool, tick_list,reason = check_format(tick)
                if not is_valid_bool:
                    logging.info("Index: {}, Reason: {}".format(current_line_index,reason))
                    noise_list.append(current_line_index)
                else:
                    data.append(Ticks(tick_list[0],tick_list[1],tick_list[2],current_line_index))
                current_line_index += 1

            counts = Counter()
            for i in data:
                if i.timestamp in counts:
                    counts[i.timestamp] += 1
                else:
                    counts[i.timestamp] = 1

            for i in data:
                if counts[i.timestamp] > 1:
                    logging.info("Index: {}, Reason: Duplicate timestamp".format(i.index))
                    noise_list.append(i.index)
                    del i

            for i in range(len(data)):
                if t == date_parse("00010101:00:00:00.000000"):
                    for j in range(i,len(data)-1):
                        if t == date_parse("00010101:00:00:00.000000"):
                            if abs(data[j].timestamp-data[j+1].timestamp).total_seconds() <= 3:
                                t = data[j].timestamp
                                #logging.info("Initial t was t0 and now it is: {} taken from index: {}".format(t,data[j].index))
                                break

                is_valid_result = isvalid(data[i],t)

                if not is_valid_result.bool:
                    logging.info("Index: {}, Reason: {}, Latest Timestamp: {}".
                                 format(data[i].index,is_valid_result.reason,t))
                    noise_list.append(data[i].index)
                else:
                    #logging.info("Till now t was: {}".format(t))
                    t = is_valid_result.timestamp
                    #logging.info("t is now: {} at index {}".format(t,data[i].index))

    return noise_list


# First function to be called when program starts

def main(argv):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    fh = MPI.File.Open(comm, argv)
    file_size = fh.Get_size()
    block_size = int(file_size/nprocs)

    if file_size < nprocs:
        print("Insufficient data in file")
        MPI.Finalize()

    # divide file into nominal blocks for each process (same length)
    block_start = rank * block_size

    if not(rank == nprocs-1):
        block_end = block_start + block_size - 1
    else:
        block_end = file_size

    block = Block(block_start,block_end)
    print("Process {}: Block: [{}, {}]".format(rank,block.start,block.end))

    # get adjusted blocks so that a block starts at a new tick and ends at the end of a tick
    block, line_count = adjust_blocks(fh,block,rank, nprocs)

    # each node sends their line_count to all the nodes with rank greater than itself
    for i in range(rank+1, nprocs):
        comm.send(line_count, dest=i)

    # process each block

    """if rank is zero the index of first tick is zero else
    index of first tick is the sum of line_count of processes with rank less than the node's rank.
    Once each node has the index of the first tick in its block, further processing starts
    Each node sends the list of noise to root node (0). Root node receives all noises and writes them into
    noise.txt"""

    first_index = 0  # index of the first tick in block

    if rank == 0:
        # identify_noise(argv,block,first_index,line_count)
        noise_list = identify_noise(argv,block,first_index,line_count)
    else:
        for i in range(rank):
            first_index += comm.recv(source=i)

        # identify_noise(argv,block,first_index,line_count)
        noise_list = identify_noise(argv,block,first_index,line_count)
        comm.send(noise_list,dest=0)

    if rank == 0:
        with open("noise.txt","w") as noise:
            for index in noise_list:
                noise.write(str(index)+"\n")
            for i in range(1,nprocs):
                noise_list = comm.recv(source=i)
                for index in noise_list:
                    noise.write(str(index)+"\n")


# Point of entry

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1])