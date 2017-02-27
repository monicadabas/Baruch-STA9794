from mpi4py import MPI
import numpy as np
import pandas as pd
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


#identify noise and signal and write them in separate files
def identify_noise(file, block, count):
    print("File format is : {}".format(type(file)))
    #file.Seek(block.start)
    #with open(file, 'r') as file:
    file.Seek(block.start)
    while count > 0:
        buffer_size = min(10, count)
        chunker = file.Readlines(chunksize=buffer_size)
        count -= buffer_size
        print("Length of chunker: {}, Count: {}".format(len(chunker), count))

        #lines = [line for line in file][:count]






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
    #return Block(block.start, block.end)
    count = get_line_count(fh,block)
    print("Process: {}, Line Count: {}".format(rank,count))

    return block, count


def main(argv):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    fh = MPI.File.Open(comm, argv)
    error = fh.Set_errhandler(MPI.ERRORS_RETURN)
    file_size = fh.Get_size()
    block_size = int(file_size/nprocs)
    #print("Error: {}".format(error))


    # divide file into blocks for each process (same length)
    block_start = rank * block_size

    if not(rank == nprocs-1):
        block_end = block_start + block_size - 1
    else:
        block_end = file_size

    block = Block(block_start,block_end)

    print("Process {}: Block: [{}, {}]".format(rank,block.start,block.end))

    block, count = adjust_blocks(fh,block,rank, nprocs)

    identify_noise(fh,block,count)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Insufficient arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1])