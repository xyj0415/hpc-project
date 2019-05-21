#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <iostream>

using namespace std;

#define GetBit(x, n) (((x) & (1 << (7 - n))) >> (7 - n))
#define SetBit(x, n) (x = x | (1 << (7 - n)))
#define ClearBit(x, n) (x = x & (0xff - (1 << (7 - n)))) 
#define conv(i, j, n) ((i) * ((n) + 2) + (j))

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
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &max_iters);

    int sqrt_p = (int) (sqrt(p) + 0.001);
    if (n % (sqrt_p * 8) != 0 && mpirank == 0)
    {
        printf("n must be multiplier of (sqrt(p) * 8).\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    int ln = n / sqrt_p / 8;
    int board_size = (ln * 8 + 2) * (ln + 2);

    srand(time(0));
    uint8_t* board = (uint8_t*) malloc(sizeof(uint8_t) * board_size);
    uint8_t* board_new = (uint8_t*) malloc(sizeof(uint8_t) * board_size);

    for (int i = 1; i <= ln * 8; i++)
    {
        for (int j = 1; j <= ln; j++)
        {
            board[conv(i, j, ln)] = rand() & 255;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();


    
    for (int iter = 0; iter < max_iters; iter++)
    {
        for (int i = 1; i <= ln * 8; i++)
        {
            //left border
            MPI_Send(&(board[conv(i, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, -1, sqrt_p), Left, MPI_COMM_WORLD);
            //right border
            MPI_Send(&(board[conv(i, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, 1, sqrt_p), Right, MPI_COMM_WORLD);
        }
        // top border
        MPI_Send(&(board[conv(ln * 8, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 0, sqrt_p), Top, MPI_COMM_WORLD);
        // bottom border
        MPI_Send(&(board[conv(1, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 0, sqrt_p), Bottom, MPI_COMM_WORLD);
        // topleft corner
        MPI_Send(&(board[conv(ln * 8, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, -1, sqrt_p), TopLeft, MPI_COMM_WORLD);
        // topright corner
        MPI_Send(&(board[conv(ln * 8, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 1, sqrt_p), TopRight, MPI_COMM_WORLD);
        // bottomright corner
        MPI_Send(&(board[conv(1, ln, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 1, sqrt_p), BottomRight, MPI_COMM_WORLD);
        // bottomleft corner
        MPI_Send(&(board[conv(1, 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, -1, sqrt_p), BottomLeft, MPI_COMM_WORLD);


        for (int i = 1; i <= ln * 8; i++)
        {
            MPI_Recv(&(board[conv(i, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, -1, sqrt_p), Right, MPI_COMM_WORLD, &status);
            MPI_Recv(&(board[conv(i, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 0, 1, sqrt_p), Left, MPI_COMM_WORLD, &status);
        }

        MPI_Recv(&(board[conv(ln * 8 + 1, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 0, sqrt_p), Bottom, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, 1, ln)]), ln, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 0, sqrt_p), Top, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(ln * 8 + 1, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, -1, sqrt_p), BottomRight, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, 1, sqrt_p), TopLeft, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(ln * 8 + 1, ln + 1, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, 1, 1, sqrt_p), BottomLeft, MPI_COMM_WORLD, &status);
        MPI_Recv(&(board[conv(0, 0, ln)]), 1, MPI_UNSIGNED_CHAR, conv2(mpirank, -1, -1, sqrt_p), TopRight, MPI_COMM_WORLD, &status);


        for (int i = 1; i <= ln * 8; i++)
        {
            for (int j = 1; j <= ln; j++)
            {
                uint8_t tl = board[conv(i + 1, j - 1, ln)];
                uint8_t l = board[conv(i, j - 1, ln)];
                uint8_t bl = board[conv(i - 1, j - 1, ln)];
                uint8_t t = board[conv(i + 1, j, ln)];
                uint8_t c = board[conv(i, j, ln)];
                uint8_t b = board[conv(i - 1, j, ln)];
                uint8_t tr = board[conv(i + 1, j + 1, ln)];
                uint8_t r = board[conv(i, j + 1, ln)];
                uint8_t br = board[conv(i - 1, j + 1, ln)];

                // most significant bit
                uint8_t alive_num = GetBit(tl, 7) + GetBit(l, 7) + GetBit(bl, 7) + GetBit(t, 0) + GetBit(b, 0) + GetBit(t, 1) + GetBit(c, 1) + GetBit(b, 1);
                if (GetBit(c, 0) == 1)
                {
                    if (alive_num == 2 || alive_num == 3)
                        SetBit(board_new[conv(i, j, ln)], 0);
                    else
                        ClearBit(board_new[conv(i, j, ln)], 0);
                }
                else
                {
                    if (alive_num == 3)
                        SetBit(board_new[conv(i, j, ln)], 0);
                    else
                        ClearBit(board_new[conv(i, j, ln)], 0);
                }

                // bits 1-6
                for (int bit = 1; bit <= 6; bit++)
                {
                    alive_num = GetBit(t, bit - 1) + GetBit(t, bit) + GetBit(t, bit + 1) + GetBit(c, bit - 1) + GetBit(c, bit + 1) + GetBit(b, bit - 1) + GetBit(b, bit) + GetBit(b, bit + 1);
                    if (GetBit(c, bit) == 1)
                    {
                        if (alive_num == 2 || alive_num == 3)
                            SetBit(board_new[conv(i, j, ln)], bit);
                        else
                            ClearBit(board_new[conv(i, j, ln)], bit);
                    }
                    else
                    {
                        if (alive_num == 3)
                            SetBit(board_new[conv(i, j, ln)], bit);
                        else
                            ClearBit(board_new[conv(i, j, ln)], bit);
                    }
                }

                // least significant bit
                alive_num = GetBit(tr, 0) + GetBit(r, 0) + GetBit(br, 0) + GetBit(t, 7) + GetBit(b, 7) + GetBit(t, 6) + GetBit(c, 6) + GetBit(b, 6);
                if (GetBit(c, 7) == 1)
                {
                    if (alive_num == 2 || alive_num == 3)
                        SetBit(board_new[conv(i, j, ln)], 7);
                    else
                        ClearBit(board_new[conv(i, j, ln)], 7);
                }
                else
                {
                    if (alive_num == 3)
                        SetBit(board_new[conv(i, j, ln)], 7);
                    else
                        ClearBit(board_new[conv(i, j, ln)], 7);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        swap(board, board_new);
    }
    if (mpirank == 0)
    {
        cout << "Time:" << MPI_Wtime() - tt << "s" << endl;
        // for (int i = 1; i <= ln * 8; i++)
        // {
        //     for (int j = 1; j <= ln; j++)
        //     {
        //         for (int k = 0; k <= 7; k++)
        //         {
        //             cout << GetBit(board[conv(i, j, ln)], k);
        //         }
        //     }
        //     cout << endl;
        // }
    }
    MPI_Finalize();
    free(board);
    free(board_new);
    return 0;
}