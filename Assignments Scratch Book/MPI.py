from mpi4py import MPI
import sys
import os

class Block:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def main(argv):

        file_size = os.path.getsize(argv)
        #MPI.Status = status
        #MPI.File = fh
        #MPI.Offset = filesize, blocksize, block_start, block_end

        buffer = 100
        blocks = []
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        block_size = int(file_size/nprocs)
        fh = MPI.File.Open(comm, argv)

        for i in range(nprocs):
            if not(i == 0):
                start = i * block_size
            else:
                start = 0
            if not(i == nprocs-1):
                end = start + block_size - 1
            else:
                end = file_size

            block = Block(start,end)
            blocks.append(block)

        for i in range(nprocs-1):
            value_at_end_index = fh.seek[blocks[i].end]
            if value_at_end_index != "/n":
                data = fh.read(buffer)
                for index in range(buffer):
                    if data[index] == "\n":
                        blocks[i].end += index
                        blocks[i+1].start = blocks[i].end + 1

        for i in range(len(blocks)):
            print("Block: {}".format(i))
            print("Block start: {}".format(blocks[i].start))
            print("Block end: {}".format(blocks[i].end))

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Insufficient arguments provided")
        sys.exit(0)
    else:
        main(argv[1])