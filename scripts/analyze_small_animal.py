import pymirc.viewer as pv
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pynemaiqpet.nema_smallanimal as nsa
import pandas as pd

from scipy.signal   import argrelextrema
from scipy.ndimage import find_objects, center_of_mass

import argparse

from pathlib import Path
#-----------------------------------------------------------------------------------------
# input parameter

# master directory
mpath  = Path.home() / 'Downloads/NEMA_Nifti_GS/'
# recursive search for nifti files to analyze
fnames     = sorted(list(mpath.rglob('*Cropped*/*.nii')))
# output directory for results
output_dir = Path('results') 
# verbose output
verbose    = True
# method for ROI to calculate recovery for hot rods (max or mean)
lp_voxel   = 'mean'
#-----------------------------------------------------------------------------------------

if not output_dir.exists():
  output_dir.mkdir(parents = True, exist_ok = True)

df = pd.DataFrame()

for nifti_file in fnames:
  if verbose:
    print(nifti_file)

  rel_path = nifti_file.relative_to(mpath)

  # read the PET volume from dicom
  nii = nib.load(nifti_file)
  nii = nib.as_closest_canonical(nii)
  vol = np.flip(nii.get_fdata().squeeze(),(0,1))
  voxsize = nii.header['pixdim'][1:4]

  flipz = False

  # flip volume in z direction if necessary
  if center_of_mass(vol)[2] < 0.5*vol.shape[2]:
    vol = np.flip(vol,2)
    flipz = True

  phantom_name =  nsa.get_phantom_name(vol, voxsize)
  if verbose:
    print(phantom_name)

  # align the PET volume to "standard" space (a digitial version of the phantom)
  vol_aligned = nsa.align_nema_2008_small_animal_iq_phantom(vol, voxsize, version = phantom_name)
  
  # generate the ROI label volume
  roi_vol = nsa.nema_2008_small_animal_pet_rois(vol_aligned, voxsize, phantom = phantom_name, lp_voxel = lp_voxel)
  
  # generate the report
  res = nsa.nema_2008_small_animal_iq_phantom_report(vol_aligned, roi_vol)

  # reformat the results such that we can append them to a single data frame
  res_reformatted = {}
  
  for key, val in res.items():
    if isinstance(val,np.ndarray):
      for ia, aval in enumerate(val):
        res_reformatted[f'{key}_{ia+1}'] = aval
    else:
      res_reformatted[key] = val
 
  df = df.append(pd.DataFrame(res_reformatted, index = [rel_path]))

  #---------------------------------------------------------------------------------------------------
  # plots

  odir = output_dir / rel_path.parent

  odir.mkdir(parents=True, exist_ok=True)

  if flipz:
    vol_aligned = np.flip(vol_aligned,2)
    roi_vol     = np.flip(roi_vol,2)
  
  # show the aligned volume and the ROI volume
  vi = pv.ThreeAxisViewer([vol_aligned,vol_aligned], [None,roi_vol**0.1], voxsize = voxsize, ls = '')
  fig_file = odir / (rel_path.stem + '_uniform_rois.png')
  vi.fig.savefig(fig_file)
  if verbose:
    print(f'wrote {fig_file}')
  plt.close(vi.fig)

  # show the summed rod planes
  bbox_uni = find_objects(vol_aligned >= 0.5*np.percentile(vol_aligned,99))
  bbox_rods = find_objects(roi_vol == 4)
  
  rod_img = vol_aligned[bbox_uni[0][0],bbox_uni[0][1],bbox_rods[0][2]].mean(2).T
  roi_img = roi_vol[bbox_uni[0][0],bbox_uni[0][1],bbox_rods[0][2]].mean(2).T
  
  fig, ax = plt.subplots(1,2, figsize = (10,5))
  ax[0].imshow(rod_img, cmap = plt.cm.Greys, interpolation='bilinear', vmax = res['uniform_ROI_mean'])
  ax[0].contour(roi_img > 0, levels = 1, cmap = 'spring')
  ax[1].imshow(rod_img**0.1, cmap = plt.cm.Greys, interpolation='bilinear')
  ax[1].contour(roi_img > 0, levels = 1, cmap = 'spring')
  
  ax[0].set_title('rod image')
  ax[1].set_title('rod image^0.1 (compressed contrast)')
  
  for axx in ax.ravel():
    axx.set_axis_off()

  fig.tight_layout()
  fig.show()
  fig_file = odir / (rel_path.stem + '_rod_rois.png')
  fig.savefig(fig_file)
  if verbose:
    print(f'wrote {fig_file}')
  plt.close(fig)

#-------------------------------------

# write all results to a single csv file
csv_file = output_dir / 'results.csv'
df.to_csv(csv_file)
if verbose:
  print(f'wrote {csv_file}')
