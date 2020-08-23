# small demo that shows how to fit radial profiles of NEMA sphere phantom to estimate
# resolution of a PET reconstruction

# we do the fit 3 times with different settings (fixing a different amount of parameters)
# the 3rd approach should be the most accurate if the sphere radii are known

import os
import pymirc.fileio as pmf
import pymirc.viewer as pv

import pynemaiqpet
import pynemaiqpet.nema_pet as nema

from scipy.ndimage import gaussian_filter

# load example data set included in package
# replace with your own data if needed (e.g. dcm_pattern = 'my/dcm_dir/*.dcm')
dcm_pattern = os.path.join(os.path.dirname(pynemaiqpet.__file__),'data','pet_recon_1','*.dcm')
dcm         = pmf.DicomVolume(dcm_pattern)
vol         = dcm.get_data() 
voxsize     = dcm.voxsize 

# FWHM of Gaussian kernel to apply before analysis
sm_fwhm_mm = 4.5

# post smooth the image
if sm_fwhm_mm > 0:
  vol = gaussian_filter(vol, sigma = sm_fwhm_mm / (2.35*voxsize))

#-------------------------------------------------------------------------------------------------
# (1) do a completely unconstrained fit 
# not a good idea since all sphere usually have the same signal (activity concentration)
# and the fitted resolution and radius are highly correlated
fitres1, sphere_results1 = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize)

print('unconstrained fit')
print(sphere_results1)

fig1a = nema.show_WB_NEMA_profiles(fitres1)

#-------------------------------------------------------------------------------------------------
# (2) do a completely unconstrained fit
# remember that fitted resolution and radius are highly correlated
fitres2, sphere_results2 = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True)

print('fit with same signal')
print(sphere_results2)

fig2a = nema.show_WB_NEMA_profiles(fitres2)

#-------------------------------------------------------------------------------------------------
# (3) do a fit where we force all spheres to have the same signal and we fix
# the radii of all spheres (using the values from the NEMA specs)
# another approach would be to use the fitted R values from (2)
fitres3, sphere_results3 = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True,
                                                            Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.])

print('fit with same signal and fixed radii')
print(sphere_results3)

fig3a = nema.show_WB_NEMA_profiles(fitres3)

# plot the max and a50 recoveries
# the 2nd argument should be the true (expected) activity concentration in the spheres
fig3b = nema.show_WB_NEMA_recoveries(sphere_results3, sphere_results3['signal'].values[0])

# show the volume 
pv.ThreeAxisViewer(vol, voxsize=voxsize)
