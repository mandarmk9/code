#!/usr/bin/env python3
import os
import shutil
import time
import numpy as np
import subprocess

temp_dir = '/vol/aibn31/data1/mandar/code/cosmo_sim_1d/multi_sim_setup_skel/'
sim_name = '/multi_k_sim/'


nstart = 44
nruns = 8
A = [-0.05, 1, -0.04, 2, -0.03, 3, -0.02, 4, -0.01, 5, -1, 11]
# from zel import initial_density
# x = np.arange(0, 1, 1/1000)
# a_sc = 1 / np.max(initial_density(x, A, 1))
# print(a_sc)

def ics_write(path, A, nstart, phi):
    k1_line = '    const int k1 = {};'.format(A[1])
    k2_line = '    const int k2 = {};'.format(A[3])
    k3_line = '    const int k3 = {};'.format(A[5])
    k4_line = '    const int k4 = {};'.format(A[7])
    k5_line = '    const int k5 = {};'.format(A[9])
    k6_line = '    const int k6 = {};'.format(A[11])

    amp1_line = '    const double amp1 = {};'.format(A[0])
    amp2_line = '    const double amp2 = {};'.format(A[2])
    amp3_line = '    const double amp3 = {};'.format(A[4])
    amp4_line = '    const double amp4 = {};'.format(A[6])
    amp5_line = '    const double amp5 = {};'.format(A[8])
    amp6_line = '    const double amp6 = {};'.format(A[10])

    phi_line = '    const double phi = {};'.format(phi)
    new_line_list = [k1_line, k2_line, k3_line, k4_line, k5_line, k6_line,
                    amp1_line, amp2_line, amp3_line, amp4_line, amp5_line, amp6_line, phi_line]

    file = path + 'ics.hh'
    shutil.move(file, file + '~')
    source = open(file + '~', 'r')
    destination = open(file, 'w')
    line_num = 0
    for line in source:
        line_num += 1
        if nstart <= line_num < nstart+int(len(new_line_list)):
            new_line = new_line_list[int(line_num - nstart)]
            destination.write(new_line + "\n")
        else:
            destination.write(line)

    source.close()
    destination.close()
    try:
        os.remove(file + '~')
        print('Cleaning up...')
    except:
        pass
    return None

def writer(temp_dir, sim_path, sim_name, A, nruns=8, nstart=44):
    sim_dir = sim_path + sim_name
    files = os.listdir(temp_dir)
    try:
        print('Making new directory for {} with {} runs'.format(sim_name[1:-1], nruns))
        os.makedirs(sim_dir, 0o755)
    except:
        pass

    runs, paths, phi_list = [], [], []
    for j in range(nruns):
        run = 'run{}/'.format(j+1)
        print('\nSetting up {}'.format(run[:-1]))
        path = sim_dir + run
        phi = '{}.0 * M_PI / {}'.format(j, nruns)

        try:
            os.makedirs(path, 0o755)
        except:
            pass
        for file in files:
            shutil.copy(temp_dir + file, path)
        ics_write(path, A, nstart, phi)
    return None

def create_run(temp_dir, sim_name, A, nruns, nstart):
    t0 = time.time()
    sim_path = '/vol/aibn31/data1/mandar/code/cosmo_sim_1d/'
    sim = sim_path + sim_name
    if os.path.exists(sim):
        print(sim)
        val = input('This run already exists, do you want to overwrite? [y/n]: ')
        valid_input = False
        while not valid_input:
            if val.lower() == 'y':
                print('Deleting old files...')
                writer(temp_dir, sim_path, sim_name, A, nruns, nstart)
                valid_input = True
            elif val.lower() == 'n':
                print('User interruption. No files were changed.')
                valid_input = True
            else:
                print('Invalid input, please enter y/n')
    else:
        writer(temp_dir, sim_path, sim_name, A, nruns, nstart)
    t1 = time.time()
    print('Run created! This took {}s'.format(np.round(t1-t0, 5)))

create_run(temp_dir, sim_name, A, nruns, nstart)


# astart, aend, numsteps, numpar, numout = 0.5, 6.0, 80000, 125000, 1600
astart, aend, numsteps, numpar, numout = 0.25, 5.0, 80000, 62500, 1600

compile_code = 'g++ -O3 -std=c++14 cosmo_sim_1d.cc -o cosmo_sim_1d'
run_code = './cosmo_sim_1d  -a {} -A {} -s {} -n {} -l {} -m'.format(astart, aend, numsteps, numpar, numout)
for n in range(nruns):
    print('\nCompiling run{}'.format(n+1))
    subprocess.run(compile_code, shell=True, check=True, cwd='/vol/aibn31/data1/mandar/code/cosmo_sim_1d/'+sim_name+'/run{}/'.format(n+1))
    print('Creating new tmux pane...')
    subprocess.run('tmux new-session -d -s {}run{} {}'.format(sim_name, n+1, run_code), shell=True, check=True, cwd='/vol/aibn31/data1/mandar/code/cosmo_sim_1d/'+sim_name+'/run{}/'.format(n+1))
    print('Done! Please check that the run has executed correctly.')
