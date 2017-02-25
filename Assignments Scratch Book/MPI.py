from mpi4py import MPI
import sys


def main(argv):


    #MPI_Status = status
    #MPI_File = fh
    #MPI_Offset = filesize, blocksize, block_start, block_end

    #buffer = 100
    block = None

    rank = MPI.COMM_WORLD.Get_rank()
    print(rank)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Insufficient arguments provided")
        sys.exit(0)
    main(sys.argv[1])