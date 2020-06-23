#-------------Modules---------------

import numpy as np
import healpy as hp

#------------Functions--------------

def smooth2reso(wt_reso,ori_reso,Hmap): 
    
    """
    Function which smooth a HEALPy map, using a gaussian, to desired resolution taking into account its original resultion
    given by ori_reso. 

    Parameters
    ----------
     
    wt_reso : float
        desired resolutions of the map in arcmin. 
    ori_reso : float
        original resolution of the map in arcmin. 
    Hmap : array
        Map we want to smooth.         
        
    Returns
    -------
    array
        The smoothed map.

    """
                  
    FWHM_i = m.sqrt(wt_reso**2 - ori_reso**2) 
    FWHM_f = FWHM_i/ 60 * np.pi/180
    Fmap = hp.smoothing(Hmap, fwhm=FWHM_f, beam_window=None, pol=False, iter=0, lmax=None,mmax=None)
    
    return Fmap
    
    
 def mergemaps(maps_array,wt_reso,dic_reso): 
    
    """
    Function which merge a list of map into a single matrix.  

    Parameters
    ----------
    
    maps_array ; array 
        Array containing all the healpy maps we want to perform the ILC on. 
    wt_reso : float
        desired resolutions of the maps. 
    dic_reso : dictionary
        dictionary containing the list of all the resolutions of each of the maps.
        
    Returns
    -------
    ndarray
        Containing in each of its column a map. 

    """
    
    for i in range(len(maps_array)):
            
        Hmap = smooth2reso(wt_reso=wt_reso,ori_reso=dic_reso[i],Hmap=maps_array[i])
        
        if i == 0:
        
            stacked = [Hmap]
            
        else: 
    
            stacked = np.concatenate((stacked,[Hmap]),axis=0)
    
    return stacked   
    
    
