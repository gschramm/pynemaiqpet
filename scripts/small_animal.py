import pymirc.viewer as pv
import pymirc.fileio as pf
import numpy as np
import pynemaiqpet.nema_smallanimal as nsa

from scipy.ndimage import find_objects

import argparse

parser = argparse.ArgumentParser(description='NEMA small animal IQ scan analyzer')
parser.add_argument('dcm_dir', help = 'absolute path of input dicom directory')
args = parser.parse_args()

dcm = pf.DicomVolume(args.dcm_dir)
vol = dcm.get_data()

vol_aligned = nsa.align_nema_2008_small_animal_iq_phantom(vol, dcm.voxsize)
roi_vol = nsa.nema_2008_small_animal_pet_rois(vol_aligned, dcm.voxsize)
nsa.nema_2008_small_animal_iq_phantom_report(vol_aligned, roi_vol)

th = 0.3*np.percentile(vol_aligned, 99.9)
bbox = find_objects(vol_aligned > th)[0]
vi = pv.ThreeAxisViewer([vol_aligned[bbox],vol_aligned[bbox]], [None,roi_vol[bbox]], 
                        voxsize = dcm.voxsize, ls = '')
