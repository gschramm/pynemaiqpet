import os
import argparse

def wb_nema_iq():
  parser = argparse.ArgumentParser(description='NEMA WB IQ scan analyzer')

  parser.add_argument('dcm_dir', help = 'absolute path of input dicom directory')
  parser.add_argument('--dcm_pattern', default = '*', help = 'file pattern for files in the dcm_dir')
  parser.add_argument('--fwhm_mm', default = 0, help = 'FWHM (mm) of Gaussian filter applied to the input volumes before the analysis', type = float)
  parser.add_argument('--radii_mm', default = [None], help = 'The radii (mm) of the 6 spheres (seperated by blanks)). If not given this is set the values "18.5 14.0 11.0 8.5 6.5 5.0" are used', nargs = '+')
  parser.add_argument('--signal', default = None, help = 'Fixed signal in [Bq/ml] (or the units of the volume) used when fitting all spheres. If not provided, the fitted value from the biggest sphere is used for all spheres', type = float)
  parser.add_argument('--wall_mm', default = 1.5, help = 'Fixed glass wall thickness (mm). If not provided 1.5mm is used.', type = float)
  parser.add_argument('--earl', default = 2, help = 'EARL version to use for limits in plots', type = int, choices = [1,2])
  parser.add_argument('--true_act_conc', default = None, help = 'True activity concentration in the spheres in [Bq/ml] (or the units of the volume). If not given, it is obtained from the fitted signal of the biggest sphere.', type = float)
  parser.add_argument('--output_dir',  help = 'name of the output directory', default = None)
  parser.add_argument('--show', help = 'show the results', action = 'store_true')
  parser.add_argument('--verbose', help = 'print (extra) verbose output', action = 'store_true')
  
  args = parser.parse_args()
  
  #-------------------------------------------------------------------------------------------------
  # load modules
  import matplotlib.pyplot as plt
  from scipy.ndimage import gaussian_filter
  
  import pymirc.fileio as pmf
  import pymirc.viewer as pv
  
  import pynemaiqpet
  import pynemaiqpet.nema_wb as nema
  
  #-------------------------------------------------------------------------------------------------
  # parse input parameters
  
  dcm_dir       = args.dcm_dir
  dcm_pattern   = args.dcm_pattern 
  sm_fwhm_mm    = args.fwhm_mm 
  Rfix          = args.radii_mm
  Sfix          = args.signal
  dfix          = args.wall_mm
  earlversion   = args.earl
  true_act_conc = args.true_act_conc
  output_dir    = args.output_dir
  show          = args.show
  verbose       = args.verbose
  
  if Rfix[0] is None:
    Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.]
  elif Rfix[0] == 'fit':
    Rfix = None
  else:
    if len(Rfix) != 6:
      raise ValueError('When manually specifying the sphere radii, 6 values must be given.')
    Rfix = [float(x) for x in Rfix]
  
  #-------------------------------------------------------------------------------------------------
  # load the dicom volume
  
  dcm         = pmf.DicomVolume(os.path.join(dcm_dir, dcm_pattern))
  vol         = dcm.get_data() 
  voxsize     = dcm.voxsize 
  
  # post smooth the image
  if sm_fwhm_mm > 0:
    vol = gaussian_filter(vol, sigma = sm_fwhm_mm / (2.35*voxsize))
  
  #-------------------------------------------------------------------------------------------------
  # do a fit where we force all spheres to have the same signal (assuming that the activity
  # concentration in all sphere was the same)
  
  # try also doing the fit without fixing the radii
  fitres, sphere_results = nema.fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True, Rfix = Rfix,
                                                            Sfix = Sfix, dfix = dfix, showBGROI = True)
  
  if verbose:
    print('fit with same signal and fixed radii')
    print(sphere_results)
  
  if output_dir is not None:
    os.makedirs(output_dir, exist_ok = True)
    sphere_results.to_csv(os.path.join(output_dir, 'fit_results.csv'))
  
  #-------------------------------------------------------------------------------------------------
  # show the results
  
  fig = nema.show_WB_NEMA_profiles(fitres)
  
  # plot the max and a50 recoveries
  # the 2nd argument should be the true (expected) activity concentration in the spheres
  # and also show the limits given by EARL (vesion 2)
  
  if true_act_conc == None:
    true_act_conc = sphere_results['signal'].values[0]
  
  fig2 = nema.show_WB_NEMA_recoveries(sphere_results, true_act_conc, earlversion = earlversion)
  
  # show the volume 
  vi = pv.ThreeAxisViewer(vol, voxsize=voxsize)
  
  # save plots
  if output_dir is not None:
    fig.savefig(os.path.join(output_dir,'sphere_profiles.pdf'))
    fig2.savefig(os.path.join(output_dir,'recoveries.pdf'))
    vi.fig.savefig(os.path.join(output_dir,'volume.png'))
  
  if show:
    plt.show()
