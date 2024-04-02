import multiprocessing as mp
import time
from fileutils import *

'''
walk folders to find all dataset files
'''
def walkmonthdays(sfolder, pattern, walktype='walk'):
    dayfiles = []
    for dayf in find_files(sfolder, pattern, listtype="walk"):
        dayfiles += [dayf]
    return dayfiles

def new_process(func, proclist, args):
    q = mp.Queue()
    proclist += [{'proc': mp.Process(target=func, args=args), 'queue': q}]
    proclist[-1]['proc'].start()

def par_files(func, days, pthreads, args):
    procs = []
    proctimetotal = 0
    dayscompleted = []
    #print(days)
    for cpu in range(pthreads):
        d = days.pop()
        dayscompleted += [d]
        #print('initial proc')
        new_process(func, procs, tuple([d]+args))
    while len(procs) > 0:
        time.sleep(0.1)
        for p in procs:
            try:
                proctimetotal += p['queue'].get_nowait()
            except:
                pass
            if not p['proc'].is_alive():
                #print('remove, tot procs: %d' % len(procs))
                procs.remove(p)
                #print('tot procs: %d' % len(procs))
        while len(procs) < pthreads:
            if len(days) == 0: break
            #print('new proc')
            d = days.pop()
            dayscompleted += [d]
            new_process(func, procs, tuple([d]+args))
    return proctimetotal