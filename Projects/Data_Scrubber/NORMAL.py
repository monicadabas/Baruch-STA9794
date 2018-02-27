from __future__ import division
from mpi4py import MPI
from datetime import datetime
import logging
import sys
from itertools import islice
from operator import itemgetter
from math import log, e, sqrt
import numpy as np
from guppy import hpy
# from memory_profiler import profile


h = hpy()

# declaration of log file for memory profiling
# mf = open("Normal_functions_memory_log.log", 'w')

# declaration of log file

frmt = '%(levelname)s:%(asctime)s:%(message)s'
fn = "Normal_log.log"

if len(sys.argv) == 4:
    loglevel = sys.argv[3][6:]
    try:
        num_level = getattr(logging, loglevel.upper())
    except AttributeError:
        print("Incorrect logging level provided")
        sys.exit()
    if not isinstance(num_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
else:
    num_level = 'ERROR'

logging.basicConfig(filename=fn, format=frmt, datefmt='%m/%d/%Y %I:%M:%S %p', level=num_level, filemode='w')

start_time = datetime.now()

# definition of block class so store the start and end character index of a block
# block is the part of file to be processed by a node


class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# returns number of ticks or rows in each block (each node processes one block)
# @profile(stream=mf)
def get_line_count(fh,block):
    block_size = block.end + 1 - block.start
    count_of_characters_read = 0
    line_count = 0
    fh.Seek(block.start)

    while count_of_characters_read < block_size:
        buffer_size = min(1000,block_size-count_of_characters_read)
        buff = np.empty(buffer_size, dtype=str)
        fh.Read([buff, MPI.CHAR])

        for i in range(buffer_size):
            if buff[i] == "\n":
                line_count += 1

        count_of_characters_read += buffer_size
    return line_count


# returns adjusted block start and end and the number of ticks in adjusted block
# @profile(stream=mf)
def adjust_blocks(fh,block,rank, nprocs):
    buffer_size = 100
    buffer = np.empty(buffer_size, dtype=str)
    error = fh.Set_errhandler(MPI.ERRORS_RETURN)

    # adjust the block start
    if not(rank == 0):
        fh.Seek(block.start)
        if error is not None:
            print("Could not seek file")
            sys.exit()

        fh.Read([buffer, MPI.CHAR])
        if error is not None:
            print("Could not read file")
            sys.exit()

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.start += i + 1
                break

    # adjust the block end
    if not(rank == nprocs-1):
        fh.Seek(block.end)
        if error is not None:
            print("Call to seek file failed")
            sys.exit()

        fh.Read([buffer, MPI.CHAR])
        if error is not None:
            print("Call to read file failed")
            sys.exit()

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.end += i
                break

    line_count = get_line_count(fh,block)

    return block, line_count


# class definition for statistics to define a normal distribution
class StatsOfNormalDist:
    def __init__(self, mean_, std_dev, variance_, skewness_, kurtosis_, elements_):
        self.mean_ = mean_
        self.std_dev = std_dev
        self.variance_ = variance_
        self.skewness_ = skewness_
        self.kurtosis_ = kurtosis_
        self.elements_ = elements_


# class definition for parameters required to get normal distribution statistics
class BlockStats:
    def __init__(self, n, m1, m2, m3, m4):
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4


"""
This function is processed by each node. It takes in the data block, noise.txt for the concerned block
along with other parameters like data_block (object of class Block which has block start and end),
first_index which is the index of first tick in block as its index in data.txt, line_count is the number of ticks
in the block.
Function processes a block in chunks, identify the signal ticks, computes the parameters required to calculate
statistics for normal distribution and returns the object of class BlockStats.
"""


# @profile(stream=mf)
def get_stats(data_file,noise_indices,data_block,first_index,line_count):
    t1 = datetime.now()
    with open(data_file, 'r') as fh:
        fh.seek(data_block.start)
        lines_read = 0  # number of ticks read
        len_noise = len(noise_indices)  # number of noise ticks in this block
        current_index_data_file = first_index  # index of the current tick in block
        current_index_noise_indices = 0  # index of current noise tick
        n, m1, m2, m3, m4 = 0, 0, 0, 0, 0  # parameters to get stats for normal distribution

        while lines_read < line_count:  # this loop goes till all the ticks in block are read
            buffer_size = min(1000,line_count-lines_read)
            buff = islice(fh,buffer_size)
            time_price = []  # list to store time and price of signal ticks in one chunk
            for tick in buff:
                if current_index_data_file == int(noise_indices[current_index_noise_indices]):
                    if len_noise - 1 > current_index_noise_indices:
                        current_index_noise_indices += 1
                    current_index_data_file += 1
                    continue
                else:
                    current_index_data_file += 1
                    signal = tick.split(",")
                    time_price.append(signal)

            time_price = sorted(time_price, key=itemgetter(0))
            last_tick = time_price[-1]
            if lines_read == 0:
                start_index = 1
            else:
                start_index = 0
            for i in range(start_index, len(time_price)):
                price_i = float(time_price[i][1])

                if start_index == 1:
                    price_i_plus_1 = float(time_price[i-1][1])

                else:
                    price_i_plus_1 = float(last_tick[1])
                try:
                    log_return = log(price_i_plus_1/price_i)
                    n1 = n
                    n += 1
                    delta = log_return - m1
                    delta_n = delta/n
                    delta_n2 = delta_n**2
                    term1 = delta * delta_n * n1
                    m1 += delta_n
                    m4 += term1*delta_n2*(n**2 - 3*n + 3) + 6*delta_n2*m2 - 4*delta_n*m3
                    m3 += term1*delta_n*(n-2) - 3*delta_n*m2
                    m2 += term1
                except Exception:
                    pass

            lines_read += buffer_size
    t2 = datetime.now()
    time_taken = (t2-t1).total_seconds()
    return BlockStats(n,m1,m2,m3,m4),time_taken


# this function computes the parameters required to calculate stats for normal distribution taking the
# parameters of two blocks at a time
# @profile(stream=mf)
def combined_stats(a,b):
    n = a.n + b.n
    delta = b.m1 - a.m1
    delta2 = delta**2
    delta3 = delta*delta2
    delta4 = delta2**2

    m1 = (a.n*a.m1 + b.n*b.m1) / n

    m2 = a.m2 + b.m2 + delta2 * a.n * b.n / n

    m3 = a.m3 + b.m3 + delta3 * a.n * b.n * (a.n - b.n)/(n**2)
    m3 += 3.0*delta * (a.n*b.m2 - b.n*a.m2)/n

    m4 = a.m4 + b.m4 + delta4*a.n*b.n * (a.n**2 - a.n*b.n + b.n**2) /n**3
    m4 += 6.0*delta2 * (a.n**2*b.m2 + b.n**2*a.m2)/(n**2) + 4.0*delta*(a.n*b.m3 - b.n*a.m3) / n

    return BlockStats(n,m1,m2,m3,m4)


# First function to be called when program starts
# Takes two arguments- 1st is the raw data.txt file and second is the noise.txt file
# noise.txt has the index of ticks that were identified as noise by SCRUB.py in increasing order
# @profile(stream=mf)
def main(data, noise):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    fh = MPI.File.Open(comm, data)
    file_size = fh.Get_size()
    block_size = int(file_size/nprocs)

    if file_size < nprocs:
        print("Insufficient data in file")

    with open(noise, 'r') as noise_file:
        noise_ticks = noise_file.readlines()

    # divide file into nominal blocks for each process (same length)
    block_start = rank * block_size

    if not(rank == nprocs-1):
        block_end = block_start + block_size - 1
    else:
        block_end = file_size

    block = Block(block_start,block_end)

    # get adjusted blocks so that a block starts at a new tick and ends at the end of a tick
    data_block, line_count = adjust_blocks(fh,block,rank, nprocs)

    # each node sends their line_count to all the nodes with rank greater than itself
    for i in range(rank+1, nprocs):
        comm.send(line_count, dest=i)

    got_all_blocks_index = datetime.now()

    # process each block

    """if rank is zero the index of first tick is zero else
    index of first tick is the sum of line_count of processes with rank less than the node's rank.
    Noise file is also divided for each block to contain the indices of noises only in that block.
    Once each node has the index of the first tick in its block, and its noise indices list, further processing starts
    Each node sends the stats parameters to root node (0). Root node receives all stats and combine them to get
    normal distribution stats for the complete data"""

    first_index = 0  # index of the first tick in block

    if rank == 0:
        noise_indices = list(filter(lambda x: first_index<=int(x)< first_index+line_count,noise_ticks))
        block_stats, time_taken = get_stats(data,noise_indices,data_block,first_index,line_count)

    else:
        for i in range(rank):
            first_index += comm.recv(source=i)

        noise_indices = list(filter(lambda x: first_index<=int(x)< first_index+line_count,noise_ticks))

        block_stats, time_taken = get_stats(data,noise_indices,data_block,first_index,line_count)
        comm.send([block_stats,time_taken],dest=0)

    # node with rank zero receives the stats from each block and combine them to get stats for complete data
    if rank == 0:
        t1 = datetime.now()
        # block_stats_list stores the stats of each block, initialized with stats of block processed by rank zero
        tt = [time_taken]
        block_stats_list = [block_stats]
        for i in range(1, nprocs):
            block_stats2, time_taken = comm.recv(source=i)
            tt.append(time_taken)
            block_stats_list.append(block_stats2)

        # combines the stats of each block using defined function
        data_stats = reduce(combined_stats, block_stats_list)

        # calculates the stats for normal distribution for the complete data
        data_mean = data_stats.m1
        data_variance = data_stats.m2/(data_stats.n-1)
        data_std_dev = sqrt(data_variance)
        data_skewness = sqrt(data_stats.n)*data_stats.m3/data_stats.m2**1.5
        data_kurtosis = data_stats.n*data_stats.m4/data_stats.m2**2

        # prints the normal distribution stats for data
        print("Mean: {}".format(data_mean))
        print("Standard Deviation: {}".format(data_std_dev))
        print("Variance: {}".format(data_variance))
        print("Kurtosis: {}".format(data_kurtosis))
        print("Skewness: {}".format(data_skewness))

        if data_kurtosis < 3 and 0 <= abs(data_skewness) <= 0.5:
            print("Returns are log normal")

        else:
            print("Returns are NOT log normal")

        all_done = datetime.now()

        logging.info("Time Profile")
        logging.info("Time to get blocks: {}".format((got_all_blocks_index-start_time).total_seconds()))
        logging.info("Time to identify signals and compute parameters: {}".format((max(tt))))
        logging.info("Time taken by root node to receive data from other nodes and compute normal "
                     "distribution statistics: {}".format((all_done-t1).total_seconds()))
        logging.info("Total time taken is: {}".format((all_done-start_time).total_seconds()))

        logging.info("Memory profile")
        logging.info(h.heap())

    MPI.Finalize()


# Point of entry

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1],sys.argv[2])

# mf.close()
