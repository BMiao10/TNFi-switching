import os
import sys

import dask
import dask.dataframe as dd

from dask_jobqueue import SGECluster
from dask.distributed import LocalCluster, Client
from dask.distributed import Client

import socket
hostname = socket.gethostname()
print(hostname)

import logging
logger = logging.getLogger('distributed.scheduler')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('distributed.core')
logger.setLevel(logging.ERROR)

def load_register_table(data_asset, table, **kwargs):
    return dd.read_parquet(f'/wynton/protected/project/ic/data/parquet/{data_asset}/{table}/', **kwargs)

def load_local_cluster(n_workers=8, memory_limit='60gb'):
    cluster = LocalCluster(n_workers=n_workers, memory_limit=memory_limit)
    client = Client(cluster)

    print(client.dashboard_link)

def load_cluster(cores=4, queue="long.q", memory="24GiB", walltime='04:00:00', scale=400):
    """
    Wrapper for loading cluster
    >>load_cluster(cores=4, queue="long.q", memory="64GiB", walltime='04:00:00', scale=400)
    """
    i = 0
    while i<100:
        try:
            cluster =  SGECluster(
                queue = 'long.q',
                cores = 8,
                memory = '40GiB',
                walltime = '04:00:00',
                death_timeout = 60,
                local_directory = f'{os.getcwd()}/dask_temp',
                log_directory = f'{os.getcwd()}/dask_temp/dask_log',
                python = sys.executable,
                resource_spec='x86-64-v=3',
                scheduler_options = {
                    'host': ''
                }
            )
        except:
            i += 1
            print(i)
        else:
            print(f'Using Port {40000 + i}...')
            break

    cluster.scale(scale)
    client = Client(cluster)
    print(client.dashboard_link)