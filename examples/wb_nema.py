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

# load example data set included in package
# replace with your own data if needed (e.g. dcm_pattern = 'my/dcm_dir/*.dcm')
dcm_pattern = os.path.join(os.path.dirname(pynemaiqpet.__file__),'data','pet_recon_2','*.dcm')
dcm         = pmf.DicomVolume(dcm_pattern)
vol         = dcm.get_data() 
voxsize     = dcm.voxsize 

# FWHM of Gaussian kernel to apply before analysis
sm_fwhm_mm = 5.

# post smooth the image
if sm_fwhm_mm > 0:
  vol = gaussian_filter(vol, sigma = sm_fwhm_mm / (2.35*voxsize))

#-------------------------------------------------------------------------------------------------
# do a fit where we force all spheres to have the same signal (assuming that the activity
# concentration in all sphere was the same
# and we fix the radii of all spheres (using the values from the NEMA specs)

# try also doing the fit without fixing the radii
fitres, sphere_results = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True,
                                                          Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.])

print('fit with same signal and fixed radii')
print(sphere_results)

# you can save the results table to csv
# sphere_results.to_csv('myresults.csv')

#-------------------------------------------------------------------------------------------------
# show the results

fig = nema.show_WB_NEMA_profiles(fitres)

# plot the max and a50 recoveries
# the 2nd argument should be the true (expected) activity concentration in the spheres
# and also show the limits given by EARL (vesion 2)
fig = nema.show_WB_NEMA_recoveries(sphere_results, sphere_results['signal'].values[0], 
                                   earlversion = 2)

# show the volume 
pv.ThreeAxisViewer(vol, voxsize=voxsize)
