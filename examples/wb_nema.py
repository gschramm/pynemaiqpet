import pymirc.fileio as pmf
import pymirc.viewer as pv

import pynemaiqpet.nema_pet as nema

dcm     = pmf.DicomVolume('../../pymirc/data/nema_petct/PT/*.dcm')
vol     = dcm.get_data() 
voxsize = dcm.voxsize 

fitres, sphere_results = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True, sm_fwhm = 5,
                                                          Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.])

print(sphere_results)

fig1 = nema.show_WB_NEMA_profiles(fitres)
fig2 = nema.show_WB_NEMA_recoveries(sphere_results, sphere_results['signal'].values[0])
