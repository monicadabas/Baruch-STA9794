from SCRUB import get_line_count, Block, adjust_blocks
from mpi4py import MPI
from datetime import datetime
import logging
import sys
from itertools import islice


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


class Stats_of_normal_dist:
    def __init__(self, mean_, std_dev, variance_):
        self.mean_ = mean_
        self.std_dev = std_dev
        self.variance_ = variance_



def get_stats(data,noise_indices,data_block,first_index,line_count):


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
    noise_indices = []
    if rank == 0:
        for line in noise_ticks:
            line = int(line.strip("\n"))
            if line < line_count:
                noise_indices.append(line)
            else:
                break

        normal_stats = get_stats(data,noise_indices,data_block,first_index,line_count)

    else:
        for i in range(rank):
            first_index += comm.recv(source=i)

        for line in noise_ticks:
            line = int(line.strip("\n"))
            if line < first_index + line_count - 1:
                noise_indices.append(line)
            else:
                break

        normal_stats = get_stats(data,noise_indices,data_block,first_index,line_count)
        comm.send(normal_stats,dest=0)

    finish_scrubbing = datetime.now()

    if rank == 0:
        means = [normal_stats.mean_]
        std_deviations = [normal_stats.std_dev]
        variances = [normal_stats.variance_]

        for i in range(1, nprocs):
            stats = comm.recv(source=i)
            means.append(stats.mean_)
            std_deviations.append(stats.std_dev)
            variances.append(stats.variance_)

        print(means, std_deviations,variances)


# Point of entry

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1],sys.argv[2])