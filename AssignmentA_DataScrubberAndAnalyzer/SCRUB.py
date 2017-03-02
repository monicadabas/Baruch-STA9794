from mpi4py import MPI
import numpy as np
import pandas as pd
from operator import attrgetter
import sys
import os


class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# return number of ticks or rows in each block (each node processes one block)
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


# function to return timestamp string to datetime format or returns false if not in right format
def date_parse(s):
    try:
        t = s.split(".")
        t = str(":".join(t))
        return pd.datetime.strptime(t, '%Y%m%d:%H:%M:%S:%f')
    except ValueError:
        return "Error"


# Ticks class with 4 features (3 from the data given and 4th as index), also checks the format of ticks
class Ticks:
    def __init__(self, timestamp, price, units, index):
        self. timestamp = timestamp
        self.price = price
        self.units = units
        self.index = index

    # checks if the object has all values in right format, if not returns False, else returns True
    def check_format(self):
        self.timestamp = date_parse(self.timestamp)
        if self.timestamp == "Error":
            return False, self

        try:
            self.price = float(self.price)
            self.units = int(self.units)
        except ValueError:
            return False, self

        if self.price <= 0 or self.units <= 0:
            return False, self

        return True, self


class IsValidResult:
    def __init__(self, bool, timestamp):
        self.bool = bool
        self.timestamp = timestamp

"""segregate the chunk of data with data in right format into noise and signal
The function received two arguments 1) a list of tick objects as data 2) the timestamp with which to compare the data
It returns a list of index of noise ticks in data as a list of index, the data list it received after removing all the noise
and the latest timestamp with which the next chunk of ticks should be compared"""


def isvalid(tick,t):
    t0 = date_parse("00010101:00:00:00.000000")

    if abs(tick.timestamp-t).total_seconds() <= 3 or t == t0:
        return IsValidResult(True, tick.timestamp)

    elif t == tick.timestamp:
        #logging.info("Duplicate timestamp")
        return IsValidResult(False,t)

    else:
        #logging.info("Timestamps is more than 3 seconds than earlier timestamp")
        return IsValidResult(False,t)


# data parser to convert list of characters read by MPI file into list of Ticks class objects
# def parser(chunk,current_index):

""" identify noise and signal and write them in separate files
# file is MPI file
# block is object of Block class with first and last character index of the part of file to be processed by a node
# first_index is the index of the first row in the block which is its index in data.txt
# count is the number of rows in this block"""


def identify_noise(fh, block, first_index, count):
    print("File format is : {}".format(type(fh)))
    fh.Seek(block.start)
    current_chunk_end = block.start
    current_index = first_index
    t = date_parse("00010101:00:00:00.000000") # a default initial timestamp

    while current_chunk_end < block.end:
        fh.Seek(current_chunk_end)
        block_size = block.end - current_chunk_end + 1
        buffer_size = min(10000, block_size)
        chunk = np.empty(buffer_size)
        fh.Read([chunk, buffer_size])

        data, row_count, chunk_end = parser(chunk,current_index)
        current_chunk_end += chunk_end
        current_index += row_count

        with open("noise.txt", 'w') as noise, open("signal.txt", 'w') as signal:
            for objt in data:
                is_valid, obj = objt.check_format()
                if not is_valid:
                    noise.write(obj.index)
                    data.remove(obj)

            # do the rest of the checks relative to other rows
            # sort data so that duplicates can be caught

            data = sorted(data, key=attrgetter('timestamp'))
            for tick in data:
                result = isvalid(tick,t)

                if not result.bool:
                    noise.write(obj.index)
                    data.remove(obj)
                else:
                    t = result.timestamp
            #noise_list, signal_data, latest_timestamp = segregate(data,t0)

            for tick in data:
                signal.write("{},{}".format(str(tick.timestamp), str(tick.price)))


"""returns adjusted block start and end and the number of ticks in adjusted block"""


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
        #print("Process: {}, Start buffer: {}".format(rank,buffer))

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
        #print("Process: {}, End buffer: {}".format(rank,buffer))

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.end += i
                break

    print("Process: {}, Adjusted block: [{}, {}]".format(rank,block.start,block.end))
    count = get_line_count(fh,block)
    print("Process: {}, Line Count: {}".format(rank,count))

    return block, count


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
    index of first tick is the sum of line_count of processes with rank less than the node's rank"""

    first_index = 0 # index of the first tick in block
    if rank == 0:
        # print("Process: {}, Index starts at: {}".format(rank,first_index))
        identify_noise(fh,block,first_index,line_count)
    else:
        """each node received line_count of the nodes with rank less than itself and
        add them to get the index of the first tick in its block"""
        for i in range(rank):
            first_index += comm.recv(source=i)
        # print("Process: {}, Index starts at: {}".format(rank,first_index))
        identify_noise(fh,block,first_index,line_count)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1])