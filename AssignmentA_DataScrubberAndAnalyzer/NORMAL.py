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
from scipy.stats.mstats import normaltest


h = hpy()

start_time = datetime.now()
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


class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# returns number of ticks or rows in each block (each node processes one block)
#@ profile(stream=mf)
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
#@ profile(stream=mf)
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

    #print("Process: {}, Adjusted block: [{}, {}]".format(rank,block.start,block.end))
    line_count = get_line_count(fh,block)
    #print("Process: {}, Line Count: {}".format(rank,line_count))

    return block, line_count


class StatsOfNormalDist:
    def __init__(self, mean_, std_dev, variance_, skewness_, kurtosis_, elements_):
        self.mean_ = mean_
        self.std_dev = std_dev
        self.variance_ = variance_
        self.skewness_ = skewness_
        self.kurtosis_ = kurtosis_
        self.elements_ = elements_


# def get_stats(data_file,noise_indices,data_block,first_index,line_count):
#     with open(data_file, 'r') as fh:
#         fh.seek(data_block.start)
#         lines_read = 0
#         current_index_data_file = first_index
#         current_index_noise_indices = 0
#         block_mean = 0
#         block_variance = 0
#         #signal_count = 0
#
#         while lines_read < line_count:
#             buffer_size = min(1000,line_count-lines_read)
#             buff = islice(fh,buffer_size)
#             time_price_linecount = []
#             ticks = []
#             for tick in buff:
#                 lines_read += 1
#                 # print("current_index_data_file:{}".format(current_index_data_file))
#                 # print("current_index_noise_indices: {}".format(current_index_noise_indices))
#                 # print("noise_indices[current_index_noise_indices]: {}".format(noise_indices[current_index_noise_indices]))
#                 if current_index_data_file == noise_indices[current_index_noise_indices]:
#                     current_index_noise_indices += 1
#                 else:
#                     ticks.append(tick)
#                 #signal_count += 1
#                 current_index_data_file += 1
#                 signal = ticks[-1].split(",")
#                 signal.append(lines_read-current_index_noise_indices-1)
#                 time_price_linecount.append(signal)
#
#             time_price_linecount = sorted(time_price_linecount, key=itemgetter(0))
#             last_tick = time_price_linecount[-1]
#             if lines_read == 0:
#                 start_index = 1
#             else:
#                 start_index = 0
#
#             for i in range(start_index, len(time_price_linecount)):
#                 price_i = float(time_price_linecount[i][1])
#
#                 if start_index == 1:
#                     price_i_plus_1 = float(time_price_linecount[i-1][1])
#
#                 else:
#                     price_i_plus_1 = float(last_tick[1])
#
#                 try:
#                     log_return = log(price_i_plus_1/price_i,e)
#                     last_mean = block_mean
#                     block_mean = last_mean + (log_return-last_mean)/(time_price_linecount[i][3])
#                     block_variance += (log_return-last_mean)*(log_return-block_mean)
#                     block_variance /= (line_count - len(noise_indices)-1)
#                 except Exception:
#                     pass
#
#         block_std_dev = sqrt(block_variance)
#
#         return StatsOfNormalDist(block_mean,block_std_dev,block_variance)


def get_stats(data_file,noise_indices,data_block,first_index,line_count):
    n, m1, m2, m3, m4 = 0, 0, 0, 0, 0
    len_noise = len(noise_indices)
    with open(data_file, 'r') as fh:
        fh.seek(data_block.start)
        lines_read = 0
        current_index_data_file = first_index
        current_index_noise_indices = 0

        while lines_read < line_count:
            buffer_size = min(1000,line_count-lines_read)
            buff = islice(fh,buffer_size)
            signal_ticks = []
            last_tick = None
            for i in range(buffer_size):
                tick_index = lines_read + first_index + i
                if tick_index != noise_indices[current_index_noise_indices]:
                    signal = buff[i].split(",")
                    signal_ticks.append([signal[0], signal[1]])
                else:
                    if current_index_noise_indices + 1 < len_noise:
                        current_index_noise_indices += 1
                    continue
            signal_ticks = sorted(signal_ticks, key=itemgetter(0))
            len_signal_ticks = len(signal_ticks)
            if last_tick is None:
                start = 1
            else:
                start = 0
            returns = [] # percentage log returns
            for i in range(start, len_signal_ticks):
                log_return = log(signal_ticks[i][1]/signal_ticks[i-1][1], e)
                returns.append(log_return)

            for i in returns:
                n1 = n
                n += 1
                delta = i - m1
                delta_n = delta/n
                delta_n2 = delta_n**2
                term1 = delta * delta_n * n1
                m1 += delta_n
                m4 += term1*delta_n2*(n**2 - 3*n + 3) + 6*delta_n2*m2 - 4*delta_n*m3
                m3 += term1*delta_n*(n-2) - 3*delta_n*m2
                m2 += term1

    no_of_returns = n
    block_mean = m1
    block_variance = m2/(n-1)
    block_std_dev = sqrt(block_variance)
    block_skewness = sqrt(n)*m3/m2*sqrt(m2)
    block_kurtosis = n*m4/m2**2 - 3

    return StatsOfNormalDist(block_mean,block_std_dev,block_variance,block_skewness,block_kurtosis,no_of_returns)


# First function to be called when program starts
# Takes two arguments- 1st is the raw data.txt file and second is the noise.txt file
# noise.txt has the index of ticks that were identified as noise by SCRUB.py in increasing order
#@ profile
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
    #print("Process {}: Block: [{}, {}]".format(rank,block.start,block.end))

    # get adjusted blocks so that a block starts at a new tick and ends at the end of a tick
    data_block, line_count = adjust_blocks(fh,block,rank, nprocs)

    # each node sends their line_count to all the nodes with rank greater than itself
    for i in range(rank+1, nprocs):
        comm.send(line_count, dest=i)

    got_all_blocks_index = datetime.now()

     # process each block

    """if rank is zero the index of first tick is zero else
    index of first tick is the sum of line_count of processes with rank less than the node's rank.
    Once each node has the index of the first tick in its block, further processing starts
    Each node sends the list of noise to root node (0). Root node receives all noises and writes them into
    noise.txt"""

    first_index = 0  # index of the first tick in block

    if rank == 0:
        noise_indices = list(filter(lambda x: first_index<=int(x)< first_index+line_count,noise_ticks))
        normal_stats = get_stats(data,noise_indices,data_block,first_index,line_count)

    else:
        for i in range(rank):
            first_index += comm.recv(source=i)

        noise_indices = list(filter(lambda x: first_index<=int(x)< first_index+line_count,noise_ticks))

        normal_stats = get_stats(data,noise_indices,data_block,first_index,line_count)
        comm.send([normal_stats,line_count-len(noise_indices)],dest=0)


    finish_scrubbing = datetime.now()

    if rank == 0:
        means = [round(normal_stats.mean_,3)]
        std_deviations = [round(normal_stats.std_dev*100,3)]
        variances = [round(normal_stats.variance_*10000,3)]
        kurtosis_s = [round(normal_stats.kurtosis_*10000,3)]
        no_of_returns = [round(normal_stats.elements_*10000,3)]
        no_of_signals = [line_count-len(noise_indices)]
        for i in range(1, nprocs):
            stats, count = comm.recv(source=i)
            means.append(round(stats.mean_,3))
            std_deviations.append(round(stats.std_dev*100,3))
            variances.append(round(stats.variance_*10000,3))
            kurtosis_s.append(round(stats.kurtosis_*10000,3))
            no_of_returns.append(round(stats.elements_*10000,3))
            no_of_signals.append(count)

        print("Means: {}".format(means))
        print("Standard Deviations: {}".format(std_deviations))
        print("Variances: {}".format(variances))
        print("Kurtosis: {}".format(kurtosis_s))
        print("No of returns: {}".format(no_of_returns))
        print("Signal Count: {}".format(no_of_signals))

        #print(normaltest(means))

        all_done = datetime.now()

        logging.info("Time Profile")
        logging.info("Time to get blocks: {}".format((got_all_blocks_index-start_time).total_seconds()))
        logging.info("Time to scrub: {}".format((finish_scrubbing-got_all_blocks_index).total_seconds()))
        #logging.info("Time taken to write noise.txt: {}".format((finish_writing_noise-start_writing_noise).total_seconds()))
        logging.info("Total time taken is: {}".format((all_done-start_time).total_seconds()))

        logging.info("Memory profile")
        logging.info(h.heap())


# Point of entry

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1],sys.argv[2])