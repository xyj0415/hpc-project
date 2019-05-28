#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <iostream>

#define conv(i, j, n) ((i) * ((n) + 2) + (j))
using namespace std;

enum DirectionTag {Left = 8888, Right, Top, Bottom, TopLeft, TopRight, BottomLeft, BottomRight};

// inline int conv(int i, int j, int n)
// {
//     return i * (n + 2) + j;
// }

inline int conv2(int rank, int xdir, int ydir, int n)
{
    int x = rank / n, y = rank % n;
    x = (x + xdir + n) % n;
    y = (y + ydir + n) % n;
    return x * n + y;
}

int main(int argc, char* argv[])
{
    int n, max_iters, p, mpirank; 
    if(argc !=3){
        fprintf(stderr, "usage: ./cpu_parallelized [dimension] [iterations]\n");
        exit(EXIT_FAILURE);
    }
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &max_iters);


    int sqrt_p = (int) (sqrt(p) + 0.001);
    if (n % sqrt_p != 0 && mpirank == 0)
    {
        printf("n must be multiplier of sqrt(p).\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    int ln = n / sqrt_p;
    int board_size = (ln + 2) * (ln + 2);

    srand(time(0));
    uint8_t* board = (uint8_t*) malloc(sizeof(uint8_t) * board_size);
    uint8_t* board_new = (uint8_t*) malloc(sizeof(uint8_t) * board_size);

    for (int i = 1; i <= ln; i++)
    {
        for (int j = 1; j <= ln; j++)
        {
            board[conv(i, j, ln)] = rand() & 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();


    
    for (int iter = 0; iter < max_iters; iter++)
    {
        for (int i = 1; i <= ln; i++)
        {
            //left border
            MPI_Send(&(board[conv(i, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, -1, sqrt_p), Left, MPI_COMM_WORLD);
            //right border
            MPI_Send(&(board[conv(i, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, 1, sqrt_p), Right, MPI_COMM_WORLD);
        }
        // top border
        MPI_Send(&(board[conv(ln, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 0, sqrt_p), Top, MPI_COMM_WORLD);
        // bottom border
        MPI_Send(&(board[conv(1, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 0, sqrt_p), Bottom, MPI_COMM_WORLD);
        // topleft corner
        MPI_Send(&(board[conv(ln, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, -1, sqrt_p), TopLeft, MPI_COMM_WORLD);
        // topright corner
        MPI_Send(&(board[conv(ln, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 1, sqrt_p), TopRight, MPI_COMM_WORLD);
        // bottomright corner
        MPI_Send(&(board[conv(1, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 1, sqrt_p), BottomRight, MPI_COMM_WORLD);
        // bottomleft corner
        MPI_Send(&(board[conv(1, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, -1, sqrt_p), BottomLeft, MPI_COMM_WORLD);


        for (int i = 1; i <= ln; i++)
        {
            MPI_Recv(&(board[conv(i, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, -1, sqrt_p), Right, MPI_COMM_WORLD, &status);
            MPI_Recv(&(board[conv(i, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, 1, sqrt_p), Left, MPI_COMM_WORLD, &status);
        }

        MPI_Recv(&(board[conv(ln + 1, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 0, sqrt_p), Bottom, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 0, sqrt_p), Top, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(ln + 1, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, -1, sqrt_p), BottomRight, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 1, sqrt_p), TopLeft, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(ln + 1, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 1, sqrt_p), BottomLeft, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, -1, sqrt_p), TopRight, MPI_COMM_WORLD, &status);


        for (int i = 1; i <= ln; i++)
        {
            for (int j = 1; j <= ln; j++)
            {
                uint8_t alive_num = 0;
                for (int xdir = -1; xdir <= 1; xdir++)
                {
                    for (int ydir = -1; ydir <= 1; ydir++)
                    {
                        if (xdir != 0 || ydir != 0)
                            alive_num += board[conv(i + xdir, j + ydir, ln)];
                    }
                }
                if (board[conv(i, j, ln)] == 0)
                {
                    board_new[conv(i, j, ln)] = alive_num == 3 ? 1 : 0;
                }
                else
                {
                    board_new[conv(i, j, ln)] = (alive_num == 2 || alive_num == 3) ? 1 : 0;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        swap(board, board_new);
    }
    if (mpirank == 0)
    {
        cout << "Time:" << MPI_Wtime() - tt << "s" << endl;
        // for (int i = 1; i <= ln; i++)
        // {
        //     for (int j = 1; j <= ln; j++)
        //     {
        //         cout << (char)(board_new[conv(i, j, ln)] + '0');
        //     }
        //     cout << endl;
        // }
    }
    MPI_Finalize();
    free(board);
    free(board_new);
    return 0;
}