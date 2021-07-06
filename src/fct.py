import json
import matplotlib.pyplot as plt
import numpy as np
import os
import neurokit2 as nk


def ppg_to_text(itp_ppg, fd):
    FOLDER_PATH = './PURE/' + fd
    fn = FOLDER_PATH + "./" + fd + ".txt"
    np.savetxt(fn, itp_ppg, fmt = '%.9e', delimiter = "\t\t")
    
def std_ppg_to_text(itp_ppg, fd):
    std_ppg = nk.standardize(itp_ppg)
    FOLDER_PATH = './PURE/' + fd
    fn = FOLDER_PATH + "./" + fd + "_std.txt"
    np.savetxt(fn, std_ppg, fmt = '%.9e', delimiter = "\t\t")
    
def get_data(fd):
    FOLDER_PATH = './PURE/' + fd + '/'+fd
    JSON_PATH = './PURE/' + fd + '/' + fd + '.json'
    with open(JSON_PATH, "r") as st_json:
        data_json = json.load(st_json)

    time_ns = list()
    ppg = list()
    for dat in data_json['/FullPackage']:
        time_ns.append(dat['Timestamp'])
        ppg.append(dat['Value']['waveform'])

    time_ns = np.array(time_ns)
    ppg = np.array(ppg)
    fl = os.listdir(FOLDER_PATH)
    fl = sorted(fl)
    imgt = list()
    for fn in fl:
        imgt.append(int(os.path.splitext(fn)[0][5:]))
    imgt =  np.array(imgt)

    imgt2 = (imgt - imgt[0])/1e6
    time_ns2 = (time_ns - imgt[0])/1e6
    return imgt2, time_ns2, ppg

def interpolation_ppg(imgt, time_ns, ppg, normalize = True):
    itp_ppgs = list()
    ppg_idx1 = 0
    ppg_idx2 = 1
    ppg_idx3 = 2
    err = 0
    for img_idx in range(1, len(imgt)-1):
#         print(len(itp_ppgs)) 
        diff1 = 0;diff2= 0
        if time_ns[ppg_idx3] - imgt[img_idx] > imgt[img_idx] - time_ns[ppg_idx1]:
            while imgt[img_idx] - time_ns[ppg_idx1] < 0:
                if ppg_idx1==0: break
                ppg_idx1 -= 1 
                ppg_idx2 -= 1 
                ppg_idx3 -= 1
            diff1 = (imgt[img_idx] - time_ns[ppg_idx1]) / (time_ns[ppg_idx2] - time_ns[ppg_idx1])
            diff2 = (time_ns[ppg_idx2] - imgt[img_idx]) / (time_ns[ppg_idx2] - time_ns[ppg_idx1])
            new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx1] * diff2
            itp_ppgs.append(new_ppg)
        else:
            if time_ns[ppg_idx3] - imgt[img_idx] < imgt[img_idx] - time_ns[ppg_idx2]:
                ppg_idx1 += 1 
                ppg_idx2 += 1 
                ppg_idx3 += 1
            diff1 = (time_ns[ppg_idx3] - imgt[img_idx]) / (time_ns[ppg_idx3] - time_ns[ppg_idx2])
            diff2 = (imgt[img_idx] - time_ns[ppg_idx2]) / (time_ns[ppg_idx3] - time_ns[ppg_idx2])
            if diff1 < 0 or diff2 < 0:
                diff1 = (imgt[img_idx] - time_ns[ppg_idx1]) / (time_ns[ppg_idx2] - time_ns[ppg_idx1])
                diff2 = (time_ns[ppg_idx2] - imgt[img_idx]) / (time_ns[ppg_idx2] - time_ns[ppg_idx1])
                new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx1] * diff2
            else:
                new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx3] * diff2
            itp_ppgs.append(new_ppg)
        if ppg_idx2 + 2 < len(time_ns):
            time_delay = time_ns[ppg_idx2+2]-time_ns[ppg_idx2] - 15 * err
    #         print(err, time_delay)
            if time_delay < 40:
                step = 2
                err = 0
            elif time_delay < 65:
                step = 1
                err += 1
            else:
                err += 1
            ppg_idx1 += step 
            ppg_idx2 += step 
            ppg_idx3 += step
        if ppg_idx3 >= len(time_ns):
            ppg_idx3 -= 1
            break
    if abs(imgt[len(imgt)-1] - time_ns[-1]) > abs(imgt[len(imgt)-1] - time_ns[ppg_idx3]):
        end_idx = ppg_idx3
    elif abs(imgt[len(imgt)-1] - time_ns[-1]) > abs(imgt[len(imgt)-1] - time_ns[ppg_idx2]):
        end_idx = ppg_idx2
    else:
        end_idx = -1
    itp_ppgs.append(ppg[end_idx])
    
    if normalize:
        itp_ppgs = nk.standardize(itp_ppgs)
    return itp_ppgs