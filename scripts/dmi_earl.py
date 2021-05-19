# small demo that shows how to fit radial profiles of NEMA sphere phantom to estimate
# resolution of a PET reconstruction

# we do the fit 3 times with different settings (fixing a different amount of parameters)
# the 3rd approach should be the most accurate if the sphere radii are known

import os
import pymirc.fileio as pmf
import pymirc.viewer as pv

import pynemaiqpet
import pynemaiqpet.nema_wb as nema

from scipy.ndimage import gaussian_filter
from glob import glob

from argparse import ArgumentParser


parser = ArgumentParser(description = 'Analyze GE PET NEMA in terms of recovery')
parser.add_argument('dcm_dir_pattern', help = 'dicom direcotry pattern')
parser.add_argument('--dcm_file_pattern', default = '*.dcm', help = 'dicom file pattern')
parser.add_argument('--injected_signal', default = None, type = float, 
                    help = 'injected signal in the spheres. if not given, taken from biggest sph.')
parser.add_argument('--sm_fwhm_mm', default = 5, type = float, 
                    help = 'FWHM of Gaussian kernel (mm) used for additional post-smoothing')
parser.add_argument('--force_smoothing', action = 'store_true', 
                    help = 'force smoothing. even for already smoothed data')
parser.add_argument('--earl_version', default = 2, type = int, choices = [1,2],
                    help = 'EARL version to use for limits in recovery plots') 
args = parser.parse_args()


# injected sphere activity concentration at start of acq.
injected_signal = args.injected_signal
sm_fwhm_mm      = args.sm_fwhm_mm
force_smoothing = args.force_smoothing
earlversion     = args.earl_version

dcm_dir_pattern  = args.dcm_dir_pattern
dcm_file_pattern = args.dcm_file_pattern

#------------------------------------------------------------------------------------------------

dcm_dirs = glob(dcm_dir_pattern)


for dcm_dir in dcm_dirs:

  print(os.path.basename(dcm_dir))

  # load example data set included in package
  dcm_pattern = os.path.join(dcm_dir, dcm_file_pattern)
  dcm         = pmf.DicomVolume(dcm_pattern)
  vol         = dcm.get_data() 
  voxsize     = dcm.voxsize 
  
  # FWHM of Gaussian kernel to apply before analysis
  dcm_hdr = dcm.firstdcmheader

  # check if the data was already post-smoothed by analysing private GE tags
  # for trans-axial and axial filter
  if ((dcm_hdr[0x0009,0x10ba].value == 0) and (dcm_hdr[0x0009,0x10db].value == 0)) or force_smoothing:
    sm_str = f'_{sm_fwhm_mm}mm_ps'
  else:
    sm_fwhm_mm = 0
    sm_str = ''
  
  # post smooth the image
  if sm_fwhm_mm > 0:
    print(f'Applying additional isotropic Gaussian post-smoothing of {sm_fwhm_mm}mm')
    vol = gaussian_filter(vol, sigma = sm_fwhm_mm / (2.35*voxsize))
  
  #-------------------------------------------------------------------------------------------------
  # do a fit where we force all spheres to have the same signal (assuming that the activity
  # concentration in all sphere was the same
  # and we fix the radii of all spheres (using the values from the NEMA specs)
  
  # try also doing the fit without fixing the radii
  fitres, sphere_results = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True,
                                                            Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.],
                                                            showBGROI = False,
                                                            Sfix = injected_signal)
  
  print('fit with same signal and fixed radii')
  print(sphere_results)
  sphere_results.to_csv(os.path.join(dcm_dir,f'{dcm.firstdcmheader.SeriesDescription}{sm_str}.csv'.replace(' ','_')))  
  
  #-------------------------------------------------------------------------------------------------
  # show the results
  
  fig = nema.show_WB_NEMA_profiles(fitres)
  fig.savefig(os.path.join(dcm_dir,f'{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles_EARL_{earlversion}.png'.replace(' ','_')))  
  
  # plot the max and a50 recoveries
  # the 2nd argument should be the true (expected) activity concentration in the spheres
  # and also show the limits given by EARL (vesion 2)

  if injected_signal is None:
    ref_signal = sphere_results.signal.values[0]
  else:
    ref_signal = injected_signal

  fig2 = nema.show_WB_NEMA_recoveries(sphere_results, ref_signal, 
                                      earlversion = earlversion)
  fig2.suptitle(f'{dcm.firstdcmheader.SeriesDescription}{sm_str}')
  fig2.tight_layout(pad = 2)

  fig2.savefig(os.path.join(dcm_dir,f'{dcm.firstdcmheader.SeriesDescription}{sm_str}_RCs_EARL_{earlversion}.png'.replace(' ','_')))  
