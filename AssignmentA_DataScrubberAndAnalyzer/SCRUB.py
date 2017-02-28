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
        if date_parse(self.timestamp) == "Error":
            return False

        try:
            self.price = float(self.price)
            self.units = int(self.units)
        except ValueError:
            return False

        if self.price <= 0 or self.units <= 0:
            return False

        return True

# data parser to convert list of characters read by MPI file into list of Ticks class objects
def parser(chunk,current_index):



# identify noise and signal and write them in separate files
# file is MPI file
# block is object of Block class with first and last character index of the part of file to be processed by a node
# first_index is the index of the first row in the block which is its index in data.txt
# count is the number of rows in this block
def identify_noise(file, block, first_index, count):
    print("File format is : {}".format(type(file)))
    file.Seek(block.start)
    current_chunk_end = block.start
    current_index = first_index
    while current_index < count:
        file.Seek(current_chunk_end)
        block_size = block.end - block.start + 1
        buffer_size = min(10000, block_size)
        buf = np.empty(buffer_size)
        chunk = file.Read([buf, buffer_size])
        data, row_count, chunk_end = parser(chunk,current_index)
        current_chunk_end += chunk_end
        current_index += row_count

        with open("noise.txt", 'w') as noise, open("signal.txt", 'w') as signal:
            for obj in data:
                if not obj.check_format():
                    noise.write(obj.index)
                    data.remove(obj)
                    #write this obj into noise else do the rest of the checks relative to other rows

                else:

                    signal.write(obj.index)






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
    block, count = adjust_blocks(fh,block,rank, nprocs)
    if rank == 0:
        identify_noise(fh,block,0,count)
    else:
        comm.Reduce()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of arguments provided")
        sys.exit(0)
    else:
        main(sys.argv[1])