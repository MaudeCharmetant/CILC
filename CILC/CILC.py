#-------------Modules---------------

import numpy as np
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as cst
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#------------Physics--------------

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255)
T_CMB = cosmo.Tcmb0.si.value #
k_B = cst.k_B.value
h = cst.h.value
c = cst.c.value
sig_T = cst.sigma_T.value #Thomson cross section in m2 
me = cst.m_e.value #Electron mass in kg 

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
                  
    FWHM_i = np.sqrt(wt_reso**2 - ori_reso**2) 
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
    
    
def map2fields(maps_array,It,nfields,wt_reso,dic_reso,median,gauss,mask,dic_freq): 
    
    """
    Function which read maps at different frequencies, smooth them to a desired resolution and merge them 
    into a big matrix composed of each pixel for each healpy field and for each frequency. 

    Parameters
    ----------
    
    maps_array : array 
        Containing all the healpy maps we want to perform the ILC on.
    It : int 
        Number of pixels per field.
    nfields : int 
        Number of fields. 
    wt_reso : float
        Desired resolutions of the maps in arcmin. 
    dic_reso : dictonary 
        Dictonary containing all 
    median : bool 
        If True remove the median of each field. Otherwise fit a Gaussian to the pixel histogram 
        and remove the bias of each field.
         
    Returns
    -------
    ndarray
        Containing in the row the number of pixels in a field, each column is a field and the tird dimesion
        is the number of frequencies.  

    """
        
    cube = np.empty([int(It),int(nfields),int(len(dic_freq))])

    for i in range(len(maps_array)):
                 
        Fmap = smooth2reso(wt_reso=wt_reso,ori_reso=dic_reso[i],Hmap=maps_array[i])
        
        if mask is not None: 
            
            Fmap *= mask 
        
        else: 
            
            Fmap = Fmap 
        
        k=0
        for j in range(0,len(Fmap),It):  
                
                #if median == True: 
                    
                median = np.median(Fmap[j:j+It])
                cube[0:It,k,i]+= Fmap[j:j+It] - [median]*(It)
                k = k+1   
                    
                #else: 
                    
                #offset = sz.create_histogram(Fmap[j:j+It-1], int(np.sqrt(np.size(Fmap[j:j+It-1]))), fit=True, plot=False) 
                #cube[0:It-1,k,i]+= Fmap[j:j+It-1] - [offset[0]]*(It-1)
                #k = k+1  

    return cube 

def covcorr_matrix(data,rowvar,mask,dic_freq):
    
    """
    Function which compute the covariance matrix and the correlation matrix.  

    Parameters
    ----------
    
    data : array
        Array containing the data we want to take the covariance matrix of. 
    rowvar : bool
        If rowvar is True (default), then each row represents a variable, with observations in the columns. 
        Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
        observations.
    mask : bool
        If True compute the covariance matrix only on the non-masked pixels. Otherwise on all the array.
    dic_freq : dictonary 
        Dictonary containing all the frequencies we want to compute the covariance matrix on.
        
    Returns
    -------
    tuple
        containing [0] : the covariance matrix [1] : the correlation matrix

    """
    
    #Compute the covariance matrix : 
    if mask is not None: 
        for i in range(len(dic_freq)):
            not_masked = np.where(data[:,i] != 0)[0]
        data = data[not_masked]
        
    cov_matrix = np.cov(data,y = None, rowvar = rowvar) #Compute the covariance matrix 

    #Compute the correlation matrix : 
    corr_matrix = np.corrcoef(data, y=None, rowvar=rowvar) #Compute the correlation matrix 
    
    return (cov_matrix,corr_matrix)

def D_I_CMB(x):
    
    """
    Function which compute the CMB spectral shape. 

    Parameters
    ----------
    
    x : array
        Frequency range over which the CMB spectral shape will be computed. 
        
    Returns
    -------
    array
        Array contaning the variarion of intensity produced by CMB over the fequencies. 

    """
    
    #Compute Delta I :  
    x_nu = np.array((cst.h.value*x)/(cst.k_B.value*T_CMB)) 
    A=np.array((2*cst.h.value*x**3)/(cst.c.value**2))      
    Delta_I = A/(np.exp(x_nu)-1)
    
    #Give feedback to the operator : 
    print("Delta I as been computed ")
    
    return   Delta_I

def D_I_tSZ(x,y):
    
    """
    Function which compute the tSZ spectral shape. 

    Parameters
    ----------
    
    x : array
        Frequency range over which the tSZ spectral shape will be computed. 
    y : float
        Value of the Compton-y parameter assumed here. 
        
    Returns
    -------
    array
        Array contaning the Variarion of intensity produced by tSZ over the fequencies. 

    """
    
    #Compute Delta I : 
    I_0 = (2*(cst.k_B.value*T_CMB)**3)/(cst.h.value*cst.c.value)**2    
    x_nu = np.array((cst.h.value*x)/(cst.k_B.value*T_CMB))    
    Delta_I = np.array(I_0*y*(x_nu**4)*(np.exp(x_nu)/(np.exp(x_nu)-1)**2)*((x_nu*(np.exp(x_nu)+1)/(np.exp(x_nu)-1))-4))
    Delta_I = Delta_I * T_CMB * ((np.exp(x_nu)-1)**2) / (x_nu**4*np.exp(x_nu)) #To make tSZ in K_CMB units. 
    
    #Give feedback to the operator : 
    print("Delta I as been computed ")
    
    return   Delta_I

def mixing_vector_CMB(dic_freq):
    
    """
    Function which compute the multiplying vector to transform a y-map into a tSZ(f). 

    Parameters
    ----------
    
    dic_freq : dic
        Dictonary containing the frequencies we want to get a tSZ map of.  
        
    Returns
    -------
    array
        Array contaning the multiplying vector. 

    """       
    
    #Initilisation : 
    freq = np.arange(0,1000)*1e9
    mix_vect = []
    
    # Compute the spectral shape of tSZ : 
    Delta_I = D_I_CMB(freq)

    #For each frequency channel, compute Delta_I : 
    for i in range(len(dic_freq)):
        
        mix_vect.append(Delta_I[dic_freq[i]]*(1e20))
        
    #Give feeback to the operator :    
    print('The mixing vector is : ',mix_vect)

    return mix_vect

def mixing_vector_tSZ(dic_freq):
    
    """
    Function which compute the multiplying vector to transform a y-map into a tSZ(f). 

    Parameters
    ----------
    
    dic_freq : dic
        Dictonary containing the frequencies we want to get a tSZ map of.  
        
    Returns
    -------
    array
        Array contaning the multiplying vector. 

    """       
    
    #Initilisation : 
    freq = np.arange(0,1000)*10**9  
    mix_vect = []
    
    # Compute the spectral shape of tSZ : 
    Delta_I = D_I_tSZ(freq,1)

    #For each frequency channel, compute Delta_I : 
    for i in range(len(dic_freq)):
        
        mix_vect.append(Delta_I[dic_freq[i]]*(1e20))
        
    #Give feeback to the operator :    
    print('The mixing vector is : ',mix_vect)

    return mix_vect

def CILC_weights(mix_vect_tSZ,mix_vect_CMB,data,cov_matrix,k,nside_tess):   
    
        
    """
    Function which compute the weights used by the ILC and do the ILC.

    Parameters
    ----------    
    mix_vect_tSZ : list 
        List containing all the values of the tSZ mixing vector. 
    mix_vect_CMB : list 
        List containing all the values of the CMB mixing vector.
    data : array 
        Array containing all the maps we want to apply the CILC on. 
    cov_matrix : array 
        Array containing the covariance matrix. 
    k : int 
        field k, when sky is divided into fields, we apply the CILC on each of the patches separatly.
    nside_tess : int 
        the number of pixels generated by this nside is used to separated the maps into fields.
        I recommend nside_tess = 4.
        
    Returns
    -------
    array
        Containing the CMB map. 

    """ 

    #Compute the weight of the ILC : 
    inv_cov = np.linalg.inv(cov_matrix) #Take the inverse of the covariance matrix. 
    p1 = (np.transpose(mix_vect_tSZ) @ inv_cov @ mix_vect_tSZ) * (np.transpose(mix_vect_CMB) @ inv_cov)
    p2 = ( np.transpose(mix_vect_CMB) @ inv_cov @  mix_vect_tSZ ) * ( np.transpose(mix_vect_tSZ) @ inv_cov )
    p3 =  (np.transpose(mix_vect_CMB) @ inv_cov  @ mix_vect_CMB) * (np.transpose(mix_vect_tSZ) @ inv_cov  @ mix_vect_tSZ)
    p4 = (np.transpose(mix_vect_CMB) @ inv_cov  @ mix_vect_tSZ)**2
    
    CILC_weight = (p1 - p2) / (p3 - p4)
    
    print('CILC weights are : ',CILC_weight)
    y=0  # Initialisation of the y map. 
    
    #Compute y map : 
    for i in range(0,len(mix_vect_tSZ)): #Go through all the frequencies.
        
        if nside_tess == 0: 
            map_Ti = data[i]
        else: 
        
            map_Ti = data[:,k,i]
        y = y + CILC_weight[i] * map_Ti  

    return y

def ILC_weights(mix_vect,data,cov_matrix,k,nside_tess):   
    
        
    """
    Function which compute the weights used by the ILC and do the ILC.

    Parameters
    ----------    
    mix_vect : list 
        List containing all the values of the mixing vector. 
    data : array 
        Array containing all the maps we want to apply the ILC on. 
    cov_matrix : array 
        Array containing the covariance matrix. 
    k : int 
        field k, when sky is divided into fields, we apply the CILC on each of the patches separatly.
    nside_tess : int 
        the number of pixels generated by this nside is used to separated the maps into fields.
        I recommend nside_tess = 4. If nside_tess = 0 then compute the ILC ove the full non-separated sky. 

    
    Returns
    -------
    array
        Cotaining the Compton-y map. 

    """ 
        
    inv_cov = np.linalg.inv(cov_matrix) #Take the inverse of the covariance matrix. 
    ILC_weight = (inv_cov @ mix_vect) / (np.transpose(mix_vect) @ inv_cov @ mix_vect) #Compute the weights
    print('ILC weights are : ',ILC_weight)
    y=0  # Initialisation of the y map. 
    
    #Compute y map : 
    for i in range(0,len(mix_vect)): #Go through all the frequencies.
        
        if nside_tess == 0: 
            map_Ti = data[i]
        else: 
        
            map_Ti = data[:,k,i]
        y = y + ILC_weight[i] * map_Ti  

    return y

def All_sky_ILC(dic_freq,maps_array,nside_map,nside_tess,wt_reso,dic_reso,median,gauss,CILC,mask):
    
    """
    Code which perform all the steps of an ILC or CILC. 

    Parameters
    ----------    
    dic_freq : dictonary
        Dictonary containing all the frequencies of the map we want to apply the ILC or CILC on. 
    maps_array : arrray 
        Array containing all the maps we want to apply the ILC on. 
    nside_map : int 
        Nside of the all the maps we want to apply the ILC on. 
    nside_tess : int 
        Nside that define the number of field we will separate the all sky map in. To apply the ILC or CILC
        on each of this field separatly. I recommend nside_tess = 4.
    wt_reso : float 
        Desired resolution of all the maps in arcmin. The maps will all the smoothed to this resolution. 
    dic_reso : dictonary 
        Dictonary contaning all the resolution of each of the maps we want to apply the ILC or CILC on.
    median : bool 
        If True will remove the ILC?CILC offset by substracing the median of each field to itself. 
    gauss : bool 
        If True will remove the offset of the ILC/CILC byt fitting a gaussian to the pixel histogran
        and susbtracting the mean of this gaussian to each field,         
    CILC : bool 
        If True the code will perform a CILC instead of and ILC. 
    mask : array 
        Array containing the mask if data are masked. 
    
    Returns
    -------
    array
        Cotaining the Compton-y map. 

    """ 
    
    if nside_tess == 0:
        
        Cube_map = mergemaps(maps_array=maps_array,wt_reso=wt_reso,dic_reso=dic_reso)
        
        if mask is not None: 
            
            Cube_map *= mask
            
        else: 
            
            Cube_map = Cube_map  
            
        cov_mat = covcorr_matrix(Cube_map,rowvar=True,mask=mask,dic_freq=dic_freq)
        
        if CILC == False: 
            
            mix = mixing_vector_tSZ(dic_freq=dic_freq)        
            fmap = ILC_weights(mix_vect=mix,data=Cube_map,cov_matrix=cov_mat[0],k=0,nside_tess=nside_tess)
            
            if mask is not None: 
                
                fmap *= mask
                
            else: 
                
                fmap = fmap
        
        else: 
            
            mix_tSZ = mixing_vector_tSZ(dic_freq=dic_freq)  
            mix_CMB = [1]*len(dic_freq)
            fmap = CILC_weights(mix_vect_tSZ=mix_tSZ,mix_vect_CMB=mix_CMB,data=Cube_map,cov_matrix=cov_mat[0],
                                k=0,nside_tess=nside_tess)
            
            if mask is not None: 
                
                fmap *= mask
                
            else: 
                
                fmap = fmap
    else:

        nfields = hp.nside2npix(nside_tess) 
        npix = hp.nside2npix(nside_map)
        It = int(npix/nfields)
            
        #Create cube : 
        Cube_map = map2fields(maps_array=maps_array,It=It,nfields=nfields,wt_reso=wt_reso,dic_reso=dic_reso,
                                  median=median,gauss=gauss,mask=mask,dic_freq=dic_freq)

        fmap = np.zeros(npix)
        l=0

        for i in range(0,npix,It):
                
            cov_mat = covcorr_matrix(Cube_map[:,l,:],rowvar=False,mask=mask,dic_freq=dic_freq)
            
            if  CILC == False: 
            
                mix = mixing_vector_tSZ(dic_freq=dic_freq)
                y = ILC_weights(mix_vect=mix,data=Cube_map,cov_matrix=cov_mat[0],k=l,nside_tess=nside_tess)  
                fmap[i:i+It]=y
                l=l+1
                
                
            else:
                
                mix_tSZ = mixing_vector_tSZ(dic_freq=dic_freq)  
                mix_CMB = [1]*len(dic_freq)              
                compo = CILC_weights(mix_vect_tSZ=mix_tSZ,mix_vect_CMB=mix_CMB,data=Cube_map,cov_matrix=cov_mat[0],
                                k=l,nside_tess=nside_tess)
                fmap[i:i+It]=compo
                l=l+1
            
            if mask is not None: 
                
                fmap *= mask 
            
            else: 
                
                fmap = fmap
                    
                
    return fmap
