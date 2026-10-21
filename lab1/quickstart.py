"""Minimalny test środowiska MPI według załączonego Quickstartu."""

import socket

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(
    f"my rank is: {rank}, out of {size} processes at node {socket.gethostname()}",
    flush=True,
)
