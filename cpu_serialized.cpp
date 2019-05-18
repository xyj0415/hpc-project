#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <iostream>
#include "utils.h"

using namespace std;

inline int conv(int i, int j, int n)
{
    return i * (n + 2) + j;
}

int main(int argc, char* argv[])
{
    int n, max_iters, p, mpirank; 

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &max_iters);

    int board_size = (n + 2) * (n + 2);

    srand(time(0));
    uint8_t* board = (uint8_t*) malloc(sizeof(uint8_t) * board_size);
    uint8_t* board_new = (uint8_t*) malloc(sizeof(uint8_t) * board_size);

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            board[conv(i, j, n)] = rand() & 1;
        }
    }
    Timer t;
    t.tic();
    for (int iter = 0; iter < max_iters; iter++)
    {
        for (int i = 1; i <= n; i++)
        {
            board[conv(i, 0, n)] = board[conv(i, n, n)];
            board[conv(i, n + 1, n)] = board[conv(i, 1, n)];
            board[conv(0, i, n)] = board[conv(n, i, n)];
            board[conv(n + 1, i, n)] = board[conv(1, i, n)];
        }
        board[conv(0, 0, n)] = board[conv(n, n, n)];
        board[conv(0, n + 1, n)] = board[conv(n, 1, n)];
        board[conv(n + 1, 0, n)] = board[conv(1, n, n)];
        board[conv(n + 1, n + 1, n)] = board[conv(1, 1, n)];

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                uint8_t alive_num = 0;
                for (int xdir = -1; xdir <= 1; xdir++)
                {
                    for (int ydir = -1; ydir <= 1; ydir++)
                    {
                        if (xdir != 0 || ydir != 0)
                            alive_num += board[conv(i + xdir, j + ydir, n)];
                    }
                }
                if (board[conv(i, j, n)] == 0)
                {
                    board_new[conv(i, j, n)] = alive_num == 3 ? 1 : 0;
                }
                else
                {
                    board_new[conv(i, j, n)] = (alive_num == 2 || alive_num == 3) ? 1 : 0;
                }
            }
        }
        swap(board, board_new);
    }
    cout << "Time:" << t.toc() << "s" << endl;
    // for (int i = 1; i <= n; i++)
    // {
    //     for (int j = 1; j <= n; j++)
    //     {
    //         cout << (char)(board_new[conv(i, j, n)] + '0');
    //     }
    //     cout << endl;
    // }
    free(board);
    free(board_new);
    return 0;
}