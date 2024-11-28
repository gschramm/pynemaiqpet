import numpy as np
import pylab as py
import pandas as pd
import matplotlib.patches as patches

from scipy.ndimage import label, labeled_comprehension, find_objects, gaussian_filter
from scipy.ndimage import binary_erosion, median_filter
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

from scipy.special  import erf
from scipy.signal   import argrelextrema, find_peaks_cwt

from lmfit    import Model

def find_background_roi(vol, voxsize, Rcenter = 82, edge_margin = 28., showROI= False):
  """ find the 2D background ROI for a NEMA sphere phantom

  Parameters
  ----------

  vol : 3d numpy array
    containing the volume

  voxsize: 1d numpy array 
    containing the voxel size (mm)

  Rcenter: float (optional)
    the radius (mm) of the sphere part that is not considered - default 82.

  edge_margin : float (optional)
    margin (mm) to stay away from the boarder of the phantom - default 28.

  showROI : bool (optional)
    whether to show 2D background ROI in a figure - default False

  Returns
  -------

  A tuple containing the indices of the background voxels.
  """

  # find the background value from a histogram analysis
  h  = np.histogram(vol[vol>0.01*vol.max()].flatten(),200)
  bg = 0.5*(h[1][np.argmax(h[0])] + h[1][np.argmax(h[0]) + 1]) 
  
  # get the axial slice with the maximum activity (spheres)
  zprof = vol.sum((0,1))
  sphere_sl = np.argmax(zprof)
  
  sphere_2d_img = vol[...,sphere_sl]
  
  bg_mask = binary_fill_holes(median_filter(np.clip(sphere_2d_img,0,0.8*bg), size = 7) > 0.5*bg)
  
  # erode mask by ca. 2.8 cm to stay away from the boundary
  nerode = int(edge_margin / voxsize[0])
  if (nerode % 2) == 0: nerode += 1
  
  bg_mask = binary_erosion(bg_mask, np.ones((nerode,nerode))) 
  
  # set center where spheres are to 0
  com = center_of_mass(binary_fill_holes(sphere_2d_img < 0.5*bg))
  
  x = voxsize[0]*(np.arange(vol.shape[0]) - com[0]) - 5
  y = voxsize[1]*(np.arange(vol.shape[1]) - com[1])
  
  X,Y = np.meshgrid(x,y)
  R = np.sqrt(X**2 + Y**2)
  
  bg_mask[R<=Rcenter] = 0
  
  # generate voxel indices for background voxels
  tmp = np.zeros(vol.shape, dtype = np.int8)
  tmp[...,sphere_sl] = bg_mask
  bg_inds = np.where(tmp == 1)

  if showROI:
    fig, ax = py.subplots(figsize = (5,5))
    ax.imshow(sphere_2d_img.T, cmap = py.cm.Greys, vmin = 0, vmax = np.percentile(sphere_2d_img,99.9))
    ax.contour(bg_mask.T)
    ax.set_title(f'background ROI slice {sphere_sl}')
    fig.tight_layout()
    fig.show()

  return bg_inds

#--------------------------------------------------------------------------------------------------

def gausssphere_profile(z = np.linspace(0, 2, 100), 
                        Z = 0.8):
    """ Radial profile of a sphere convolved with a 3D radial symmetric Gaussian

     Parameters
     ----------
     z : 1D numpy float array 
       normalized radial coordinate    (r / (sqrt(2) * sigma))

     Z : float
       normalized radius of the sphere (R / (sqrt(2) * sigma))

    Returns
    -------
    1D numpy array
    """

    sqrtpi = np.sqrt(np.pi)
    
    P = np.zeros_like(z)

    inds0 = np.argwhere(z == 0)
    inds1 = np.argwhere(z != 0)
    
    P[inds0] = erf(Z) - 2 * Z * np.exp(-Z**2) / sqrtpi

    P[inds1] = ( 0.5 * (erf(z[inds1] + Z) - erf(z[inds1] - Z)) - 
                (0.5/sqrtpi) * ((np.exp(-(z[inds1] - Z)**2) - np.exp(-(z[inds1] + Z)**2)) / z[inds1]))
    
    return P

#--------------------------------------------------------------------------------------------------

def glasssphere_profile(r,
                        R    = 18.5,
                        FWHM = 5,
                        d    = 1.5,
                        S    = 10.0,
                        B    = 1.0):
    """ Radial profile of a hot sphere with cold glass wall in warm background

    Parameters
    ----------
    r : 1D numpy float array 
      array with radial coordinates

    R : float, optional
      the radius of the sphere

    FWHM : float, optional
      the full width at half maximum of the points spread function

    d : float, optional
      the thickness (diameter) of the cold glass wall

    S : float, optional
      the signal in the sphere

    B : float, optional
      the signal in the background

    Returns
    -------
    1D numpy float array
    """
    sqrt2 = np.sqrt(2)

    sigma = FWHM / (2*np.sqrt(2*np.log(2)))  
    Z     = R / (sigma*sqrt2)
    w     = d / (sigma*sqrt2)
    z     = r / (sigma*sqrt2)

    P = S*gausssphere_profile(z, Z) - B*gausssphere_profile(z, Z + w) + B

    return P

#--------------------------------------------------------------------------------------------------

def plot1dprofiles(vol,
                   voxsizes):
    """ Plot profiles along the x, y and z axis through the center of mass of a sphere

    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes :  3 component array 
      with the voxel sizes
    """ 
    # now we have to find the activity weighted center of gravity of the sphere
    # to do so we do a coarse delineation of the sphere (30% over bg)
    bg    = np.mean(vol[:,:,0])
    absth = relth*(vol.max() - bg) + bg

    mask              = np.zeros_like(vol, dtype = np.uint8)
    mask[vol > absth] = 1

    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax will not be included in the fit
    rmax = np.min((i0.max(),i1.max(),i2.max()))/2

    # first try to get the center of mass via the coarse delineation 
    weights       = vol[mask == 1]
    summedweights = np.sum(weights)
 
    c0 = np.sum(i0[mask == 1]*weights) / summedweights  
    c1 = np.sum(i1[mask == 1]*weights) / summedweights  
    c2 = np.sum(i2[mask == 1]*weights) / summedweights  

    r  = np.sqrt((i0 - c0)**2 + (i1 - c1)**2 + (i2 - c2)**2)

    # second try to get the center of mass
    # use weights from a smoothed volume 
    sigmas = 4 / (2.355*voxsizes)
    vol_sm = gaussian_filter(vol, sigma = sigmas)

    weights       = vol_sm[r <= rmax]
    summedweights = np.sum(weights)

    d0 = np.sum(i0[r <= rmax]*weights) / summedweights  
    d1 = np.sum(i1[r <= rmax]*weights) / summedweights  
    d2 = np.sum(i2[r <= rmax]*weights) / summedweights  

    r  = np.sqrt((i0 - d0)**2 + (i1 - d1)**2 + (i2 - d2)**2)

    if plot1dprofiles:
        spherecenter = np.unravel_index(np.argmin(r),r.shape)
        prof0 = vol[:, spherecenter[1], spherecenter[2]]
        prof1 = vol[spherecenter[0], :, spherecenter[2]]
        prof2 = vol[spherecenter[0], spherecenter[1], :]

        dims = vol.shape

        prof02 = vol.sum(axis = (1,2)) / (dims[1]*dims[2])
        prof12 = vol.sum(axis = (0,2)) / (dims[0]*dims[2])
        prof22 = vol.sum(axis = (0,1)) / (dims[0]*dims[1])

        c0 = sum(prof0*voxsizes[0]*np.arange(len(prof0))) / sum(prof0)
        c1 = sum(prof1*voxsizes[1]*np.arange(len(prof1))) / sum(prof1)
        c2 = sum(prof2*voxsizes[2]*np.arange(len(prof2))) / sum(prof2)

        #bg = 0.5*(prof2[0:3].mean() + prof2[-3:].mean())

        #fwhm0 = np.argwhere(prof0 - bg > 0.5*(prof0.max() - bg))[:,0].ptp()*voxsizes[0]
        #fwhm1 = np.argwhere(prof1 - bg > 0.5*(prof1.max() - bg))[:,0].ptp()*voxsizes[1]
        #fwhm2 = np.argwhere(prof2 - bg > 0.5*(prof2.max() - bg))[:,0].ptp()*voxsizes[2]

        fig1d, ax1d = py.subplots(1)
        ax1d.plot(voxsizes[0]*np.arange(len(prof0)) - c0, prof0, label = 'x')
        ax1d.plot(voxsizes[1]*np.arange(len(prof1)) - c1, prof1, label = 'y')
        ax1d.plot(voxsizes[2]*np.arange(len(prof2)) - c2, prof2, label = 'z')
        py.legend()

#--------------------------------------------------------------------------------------------------

def get_sphere_center(vol, 
                      voxsizes,
                      relth = 0.25):
    """ Get the center of gravity of a single hot sphere in a volume
    
    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes : 3 component array 
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse 
      delination of the sphere - default 0.25
    """
    # now we have to find the activity weighted center of gravity of the sphere
    # to do so we do a coarse delineation of the sphere (30% over bg)
    bg    = np.mean(vol[:,:,0])
    absth = relth*(vol.max() - bg) + bg

    mask              = np.zeros_like(vol, dtype = np.uint8)
    mask[vol > absth] = 1

    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax will not be included in the fit
    rmax = np.min((i0.max(),i1.max(),i2.max()))/2

    # first try to get the center of mass via the coarse delineation 
    weights       = vol[mask == 1]
    summedweights = np.sum(weights)
 
    c0 = np.sum(i0[mask == 1]*weights) / summedweights  
    c1 = np.sum(i1[mask == 1]*weights) / summedweights  
    c2 = np.sum(i2[mask == 1]*weights) / summedweights  

    r  = np.sqrt((i0 - c0)**2 + (i1 - c1)**2 + (i2 - c2)**2)

    # second try to get the center of mass
    # use weights from a smoothed volume 
    sigmas = 4 / (2.355*voxsizes)
    vol_sm = gaussian_filter(vol, sigma = sigmas)

    weights       = vol_sm[r <= rmax]
    summedweights = np.sum(weights)

    d0 = np.sum(i0[r <= rmax]*weights) / summedweights  
    d1 = np.sum(i1[r <= rmax]*weights) / summedweights  
    d2 = np.sum(i2[r <= rmax]*weights) / summedweights  

    sphere_center = np.array([d0, d1, d2])

    return sphere_center

#--------------------------------------------------------------------------------------------------

def fitspheresubvolume(vol,
                       voxsizes,
                       relth          = 0.25,
                       Rfix           = None,
                       FWHMfix        = None,
                       dfix           = None,
                       Sfix           = None,
                       Bfix           = None,
                       wm             = 'dist',
                       cl             = False,
                       sphere_center  = None):
    """Fit the radial sphere profile of a 3d volume containg 1 sphere

    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes : 3 component array 
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse 
      delination of the sphere

    dfix, Sfix, Bfix, Rfix : float, optional 
      fixed values for the wall thickness, signal, background and radius

    wm : string, optinal   
      the weighting method of the data (equal, dist, sqdist)

    cl : bool, optional
      bool whether to compute the confidence limits (this takes very long)
 
    sphere_center : 3 element np.array 
      containing the center of the spheres in mm
      this is the center of in voxel coordiantes multiplied by the voxel sizes

    Returns
    -------
    Dictionary
      with the fitresults (as returned by lmfit)
    """                   

    if sphere_center is None: sphere_center = get_sphere_center(vol, voxsizes, relth = relth)
      
    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    rmax = np.min((i0.max(),i1.max(),i2.max()))/2
    r  = np.sqrt((i0 - sphere_center[0])**2 + (i1 - sphere_center[1])**2 + (i2 - sphere_center[2])**2)

    data = vol[r <= rmax].flatten()
    rfit = r[r <= rmax].flatten()

    if (Rfix == None): Rinit = 0.5*rmax
    else: Rinit = Rfix
 
    if (FWHMfix == None): FWHMinit = 2*voxsizes[0]  
    else: FWHMinit = FWHMfix
 
    if (dfix == None): dinit = 0.15  
    else: dinit = dfix

    if (Sfix == None): Sinit = data.max() 
    else: Sinit = Sfix  
 
    if (Bfix == None): Binit = data.min()
    else: Binit = Bfix 

    # lets do the actual fit
    pmodel = Model(glasssphere_profile)
    params = pmodel.make_params(R = Rinit, FWHM = FWHMinit, d = dinit, S = Sinit, B = Binit)

    # fix the parameters that should be fixed
    if Rfix    != None: params['R'].vary    = False
    if FWHMfix != None: params['FWHM'].vary = False
    if dfix    != None: params['d'].vary    = False
    if Sfix    != None: params['S'].vary    = False
    if Bfix    != None: params['B'].vary    = False

    params['R'].min    = 0
    params['FWHM'].min = 0
    params['d'].min    = 0
    params['S'].min    = 0
    params['B'].min    = 0

    if   wm == 'equal' : weigths = np.ones_like(rfit)
    elif wm == 'sqdist': weights = 1.0/(rfit**2) 
    else               : weights = 1.0/rfit 

    weights[weights == np.inf] = 0

    fitres = pmodel.fit(data, r = rfit, params = params, weights = weights)
    fitres.rdata = rfit
    if cl: fitres.cls   = fitres.conf_interval()

    # calculate the a50 mean
    fitres.a50th     = fitres.values['B'] + 0.5*(vol.max() - fitres.values['B'])
    fitres.mean_a50  = np.mean(data[data >= fitres.a50th])

    # calculate the mean
    fitres.mean = np.mean(data[rfit <= fitres.values['R']]) 

    # calculate the max
    fitres.max = data.max()

    # add the sphere center to the fit results
    fitres.sphere_center = sphere_center

    return fitres

#--------------------------------------------------------------------------------------------------

def plotspherefit(fitres, ax = None, xlim = None, ylim = None, unit = 'mm', showres = True):
    """Plot the results of a single sphere fit 

    Parameters
    ----------
    fitres : dictionary
      the results of the fit as returned by fitspheresubvolume

    ax : matplotlib axis, optional
      to be used for the plot

    xlim, ylim : float, optional
      the x/y limit

    unit : str, optional
      the unit of the radial coordinate

    showres : bool, optional
      whether to add text about the fit results in the plot
    """                   

    rplot = np.linspace(0, fitres.rdata.max(),100)
    
    if ax   == None: fig, ax = py.subplots(1)
    if xlim == None: xlim = (0,rplot.max())

    ax.plot(fitres.rdata, fitres.data, 'k.', ms = 2.5)
    
    ax.add_patch(patches.Rectangle((0, 0), fitres.values['R'], fitres.values['S'], 
                  facecolor = 'lightgrey', edgecolor = 'None'))
    x2  = fitres.values['R'] + fitres.values['d']
    dx2 = xlim[1] - x2 
    ax.add_patch(patches.Rectangle((x2, 0), dx2, fitres.values['B'], 
                  facecolor = 'lightgrey', edgecolor = 'None'))

    ax.plot(rplot, fitres.eval(r = rplot), 'r-')
    ax.set_xlabel('R (' + unit + ')')
    ax.set_ylabel('signal')

    ax.set_xlim(xlim)
    if ylim != None: ax.set_ylim(ylim)

    if showres:
        ax.text(0.99, 0.99, fitres.fit_report(), fontsize = 6, transform = ax.transAxes, 
                            verticalalignment='top', horizontalalignment = 'right')

#--------------------------------------------------------------------------------------------------

def NEMASubvols(input_vol,    
                voxsizes,     
                relTh    = 0.2,
                minvol   = 300,
                margin   = 9,  
                nbins    = 100,
                zignore  = 38,
                bgSignal = None):
    """ Segment a complete NEMA PET volume with several hot sphere in different subvolumes containing
        only one sphere

    Parameters
    ----------
    input_vol : 3D numpy array
      the volume to be segmented

    voxsizes  a 1D numpy array 
      containing the voxelsizes

    relTh : float, optional
      the relative threshold used to find spheres

    minvol : float, optional
      minimum volume of spheres to be segmented (same unit as voxel size^3)

    margin : int, optional
      margin around segmented spheres (same unit as voxel size)

    nbins : int, optional
      number of bins used in histogram for background determination

    zignore : float, optional
     distance to edge of FOV that is ignored (same unit as voxelsize) 

    bgSignal : float or None, optional
      the signal intensity of the background
      if None, it is auto determined from a histogram analysis

    Returns
    -------
    list
      of slices to access the subvolumes from the original volume
    """ 

    vol = input_vol.copy()

    xdim, ydim, zdim = vol.shape
    
    minvol = int(minvol / np.prod(voxsizes))
    
    dx = int(np.ceil(margin / voxsizes[0]))
    dy = int(np.ceil(margin / voxsizes[1]))
    dz = int(np.ceil(margin / voxsizes[2]))

    nzignore = int(np.ceil(zignore / voxsizes[2]))
    vol[:,:,:nzignore]  = 0   
    vol[:,:,-nzignore:] = 0   
 
    # first do a quick search for the biggest sphere (noisy edge of FOV can spoil max value!)
    histo = py.histogram(vol[vol > 0.01*vol.max()], nbins) 
    #bgSignal = histo[1][argrelextrema(histo[0], np.greater)[0][0]]
    if bgSignal is None:
      bgSignal = histo[1][find_peaks_cwt(histo[0], np.arange(nbins/6,nbins))[0]]
    thresh   = bgSignal + relTh*(vol.max() - bgSignal)
    
    vol2               = np.zeros(vol.shape, dtype = int)
    vol2[vol > thresh] = 1
    
    vol3, nrois = label(vol2)
    rois = np.arange(1, nrois + 1)
    roivols = labeled_comprehension(vol, vol3, rois, len, int, 0)
   
    i = 1
    
    for roi in rois: 
        if(roivols[roi-1] < minvol): vol3[vol3 == roi] = 0
        else:
            vol3[vol3 == roi] = i
            i = i + 1
    
    nspheres     = vol3.max()
    spherelabels = np.arange(1, nspheres + 1)
    
    bboxes = find_objects(vol3)
    
    nmaskvox = list()
    slices   = list()    
 
    for bbox in bboxes:
        xstart = max(0, bbox[0].start - dx)
        xstop  = min(xdim, bbox[0].stop + dx + 1)
    
        ystart = max(0, bbox[1].start - dy)
        ystop  = min(xdim, bbox[1].stop + dy + 1)
    
        zstart = max(0, bbox[2].start - dz)
        zstop  = min(xdim, bbox[2].stop + dz + 1)
    
        slices.append((slice(xstart,xstop,None), slice(ystart,ystop,None), slice(zstart,zstop,None)))

        nmaskvox.append((xstop-xstart)*(ystop-ystart)*(zstop-zstart))

    # sort subvols acc to number of voxel
    slices   = [ slices[kk] for kk in np.argsort(nmaskvox)[::-1] ]

    return slices

#--------------------------------------------------------------------------------------------------

def findNEMAROIs(vol, voxsizes, R = None, relth = 0.25, bgth = 0.5):
  """
  image-based ROI definition in a NEMA IQ scan 

  Arguments
  ---------

  vol      ... the volume to be segmented

  voxsizes ... a numpy array of voxelsizes in mm


  Keyword arguments
  ----------------
  
  R        ... (np.array or list) with the sphere radii in mm
               default (None) means [18.5,14.,11.,8.5,6.5,5.]       

  relth    ... (float) relative threshold to find the spheres
               above background (default 0.25)

  bgth     ... (float) relative threshold to find homogenous
               background around spheres

  Returns
  -------

  A volume containing labels for the ROIs:
  1      ... background
  2 - 7  ... NEMA spheres (large to small)
  """

  if R is None: R = np.array([18.5,14.,11.,8.5,6.5,5.])

  slices   = NEMASubvols(vol, voxsizes)
  labelvol = np.zeros(vol.shape, dtype = np.uint8) 
  bgmask   = np.zeros(vol.shape, dtype = np.uint8) 
  
  for i in range(len(slices)):
    subvol        = vol[slices[i]]
    sphere_center = get_sphere_center(subvol, voxsizes, relth = relth)
  
    i0, i1, i2 = np.indices(subvol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]
  
    r  = np.sqrt((i0 - sphere_center[0])**2 + (i1 - sphere_center[1])**2 + (i2 - sphere_center[2])**2)
  
    if i == 0: bg = subvol[r > (R[i] + 7)].mean()
  
    submask = np.zeros(subvol.shape, dtype = np.uint8)
    submask[r <= R[i]] = i + 2
  
    labelvol[slices[i]] = submask
  
  # calculate the background
  bgmask[vol > bgth*bg]  = 1
  bgmask[labelvol >= 2] = 0
  
  bgmask_eroded = binary_erosion(bgmask, iterations = int(15/voxsizes[0]))
  labelvol[bgmask_eroded] = 1

  return(labelvol)

#--------------------------------------------------------------------------------------------------

def fit_WB_NEMA_sphere_profiles(vol,        
                                voxsizes,
                                sm_fwhm   = 0,
                                margin    = 9.0,
                                dfix      = 1.5,
                                Sfix      = None,
                                Rfix      = None,
                                FWHMfix   = None,
                                wm           = 'dist',
                                nmax_spheres = 6,
                                sameSignal   = False,
                                showBGROI    = False):
    """ Analyse the sphere profiles of a NEMA scan

    Parameters
    ----------
    vol : 3D numpy array
      the volume to be segmented

    voxsizes :  a 3 element numpy array 
      of voxelsizes in mm

    sm_fwhm : float, optional
      FWHM of the gaussian used for post-smoothing (mm)

    dfix, Sfix : float, optional
      fixed values for the wall thickness, and signal

    Rfix : 1D numpy array, optional
      a 6 component array with fixed values for the sphere radii (mm)

    margin: float, optional
      margin around segmented spheres (same unit as voxel size)

    wm : str, optional
      the weighting method of the data (equal, dist, sqdist)

    nmax_spheres: int, optional
      maximum number of spheres to consider (default 6)

    sameSignal : bool, optional
      whether to forace all spheres to have the signal from the biggest sphere

    showBGROI : bool ,optional
      whether to show the background ROI in a separate figure - default False 

    Returns
    -------
    Dictionary
        containing the fitresults
    """ 

    unit = 'mm'   

    if sm_fwhm > 0:
        print('\nPost-smoothing with ' + str(sm_fwhm) + ' mm')
        sigma    = sm_fwhm / 2.355
        sigmas   = sigma / voxsizes
        vol      = gaussian_filter(vol, sigma = sigmas)

    # find the 2D background ROI
    bg_inds = find_background_roi(vol, voxsizes, showROI = showBGROI)
    bg_mean = vol[bg_inds].mean()
    bg_cov  = vol[bg_inds].std() / bg_mean

    slices  = NEMASubvols(vol, voxsizes, margin = margin, bgSignal = bg_mean)

    subvols = list()
    for iss, ss in enumerate(slices): 
      if iss < nmax_spheres:
        subvols.append(vol[ss])
     
    if Rfix == None: Rfix = [None] * len(subvols)
    if len(Rfix) < len(subvols): Rfix = Rfix + [None] * (len(subvols) - len(Rfix))

    # initial fit to get signal in the biggest sphere
    if (Sfix == None) and (sameSignal == True):
        initfitres = fitspheresubvolume(subvols[0], voxsizes, dfix = dfix, Bfix = bg_mean, 
                                        FWHMfix = FWHMfix, Rfix = Rfix[0], wm = wm)
        Sfix = initfitres.params['S'].value

    # fit of all spheres
    fitres = []
    for i in range(len(subvols)):
      fitres.append(fitspheresubvolume(subvols[i], voxsizes, dfix = dfix, Sfix = Sfix, 
                                       Bfix = bg_mean, FWHMfix = FWHMfix, Rfix = Rfix[i]))

    # summary of results
    fwhms = np.array([x.values['FWHM'] for x in fitres])
    Rs    = np.array([x.values['R'] for x in fitres])
    Bs    = np.array([x.values['B'] for x in fitres])
    Ss    = np.array([x.values['S'] for x in fitres])
   
    sphere_mean_a50  = np.array([x.mean_a50  for x in fitres])  
    sphere_mean      = np.array([x.mean      for x in fitres])  
    sphere_max       = np.array([x.max       for x in fitres])  

    sphere_results = pd.DataFrame({'R':Rs,'FWHM':fwhms,'signal':Ss, 'mean_a50':sphere_mean_a50,
                                   'mean':sphere_mean, 'max':sphere_max, 
                                   'background_mean': bg_mean, 'background_cov': bg_cov}, 
                                    index = np.arange(1,len(subvols)+1))

    return fitres, sphere_results

#-------------------------------------------------------------------------------------------
def show_WB_NEMA_profiles(fitres):

  nsph = len(fitres)

  rmax = fitres[0].rdata.max()
  fig, axes = py.subplots(2,3, figsize = (18,8.3))

  ymax = 1.05*max([x.max for x in fitres])

  for i in range(nsph):
    plotspherefit(fitres[i], ylim = (0, ymax), 
                  ax = axes[np.unravel_index(i,axes.shape)], xlim = (0,1.5*rmax))
  if nsph < 6:
    for i in np.arange(nsph,6):
        ax = axes[np.unravel_index(i,axes.shape)]
        ax.set_axis_off()

  fig.tight_layout()
  fig.show()

  return fig

#-------------------------------------------------------------------------------------------
def show_WB_NEMA_recoveries(sphere_results, true_activity, 
                            earlcolor = 'lightgreen', earlversion = 2):

  unit = 'mm'   

  a50RCs = sphere_results['mean_a50'].values / true_activity
  maxRCs = sphere_results['max'].values / true_activity
  Rs     = sphere_results['R'].values

  fig2, axes2 = py.subplots(1,2, figsize = (6,4.), sharex = True)

  # for the EARL limits see
  # http://earl.eanm.org/cms/website.php?id=/en/projects/fdg_pet_ct_accreditation/accreditation_specifications.htm

  if earlversion == 1:
    RCa50_min  = np.array([0.76, 0.72, 0.63, 0.57, 0.44, 0.27]) 
    RCa50_max  = np.array([0.89, 0.85, 0.78, 0.73, 0.60, 0.43]) 
    RCmax_min  = np.array([0.95, 0.91, 0.83, 0.73, 0.59, 0.34]) 
    RCmax_max  = np.array([1.16, 1.13, 1.09, 1.01, 0.85, 0.57]) 
  elif earlversion == 2:
    RCa50_min  = np.array([0.85, 0.82, 0.80, 0.76, 0.63, 0.39]) 
    RCa50_max  = np.array([1.00, 0.97, 0.99, 0.97, 0.86, 0.61]) 
    RCmax_min  = np.array([1.05, 1.01, 1.01, 1.00, 0.85, 0.52]) 
    RCmax_max  = np.array([1.29, 1.26, 1.32, 1.38, 1.22, 0.88]) 

  # add the EARL limits
  if earlcolor != None:
    for i in range(len(Rs)): 
      axes2[0].add_patch(patches.Rectangle((Rs[i] - 0.5, RCa50_min[i]), 1, 
                         RCa50_max[i] - RCa50_min[i], 
                         facecolor = earlcolor, edgecolor = 'None'))
      axes2[0].set_title(f'EARL version {earlversion}')


  axes2[0].plot(Rs, a50RCs, 'ko')
  axes2[0].set_ylim(min(0.25,0.95*a50RCs.min()), max(1.02,1.05*a50RCs.max()))
  axes2[0].set_xlabel('R (' + unit + ')')
  axes2[0].set_ylabel('RC a50')

  # add the EARL limits
  if earlcolor != None:
    for i in range(len(Rs)): 
      axes2[1].add_patch(patches.Rectangle((Rs[i] - 0.5, RCmax_min[i]), 1, 
                         RCmax_max[i] - RCmax_min[i], 
                         facecolor = earlcolor, edgecolor = 'None'))
      axes2[1].set_title(f'EARL version {earlversion}')

  axes2[1].plot(Rs, maxRCs, 'ko')
  axes2[1].set_ylim(min(0.29,0.95*maxRCs.min()), max(1.4,1.05*maxRCs.max()))
  axes2[1].set_xlabel('R (' + unit + ')')
  axes2[1].set_ylabel('RC max')

  fig2.tight_layout()
  fig2.show()

  return fig2
