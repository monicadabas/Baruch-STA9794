from mpi4py import MPI
import numpy as np
import sys
import os


class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end


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


def adjust_blocks(fh,block,rank, nprocs):
    buffer_size = 100
    buffer = np.empty(buffer_size, dtype=str)

    # adjust the block start
    if not(rank == 0):
        fh.Seek(block.start)
        fh.Read([buffer, MPI.CHAR])
        #print("Process: {}, Start buffer: {}".format(rank,buffer))

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.start += i + 1
                break

    # adjust the block end
    if not(rank == nprocs-1):
        fh.Seek(block.end)
        fh.Read([buffer, MPI.CHAR])
        #print("Process: {}, End buffer: {}".format(rank,buffer))

        for i in range(buffer_size):
            if buffer[i] == "\n":
                block.end += i
                break

    print("Process: {}, Adjusted block: [{}, {}]".format(rank,block.start,block.end))

    count = get_line_count(fh,block)
    print("Process: {}, Line Count: {}".format(rank,count))


def main(argv):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    fh = MPI.File.Open(comm, argv)
    file_size = fh.Get_size()
    block_size = int(file_size/nprocs)

    # divide file into blocks for each process (same length)
    block_start = rank * block_size

    if not(rank == nprocs-1):
        block_end = block_start + block_size - 1
    else:
        block_end = file_size

    block = Block(block_start,block_end)

    print("Process {}: Block: [{}, {}]".format(rank,block.start,block.end))

    adjust_blocks(fh,block,rank, nprocs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Insufficient arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1])
        
