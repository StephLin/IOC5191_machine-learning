import multiprocessing as mp
import subprocess as sp

N_CPUS = mp.cpu_count()


def task(perplexity):
    command = ' '.join([
        'python3', 'HW07_2_tSNE.py',
        str(perplexity), '--enable-part-1', '--enable-part-2',
        '--enable-part-3'
    ])
    sp.check_output(command, shell=True)


if __name__ == '__main__':
    with mp.Pool(N_CPUS) as pool:
        pool.map(task, [5, 10, 15, 20, 25, 30])