import numpy as np
import jax.numpy as jnp


from MIJax.MIJax import (
    reconstruct_highres_manual,
    detect_stars,
    cut_cutouts,
    psf0_from_cutouts,
    psf_photometry,
    refine_wcs,
    get_Gaia,
    get_Euclid_data,
    recompute_WCS,
    get_matrice_transformation,
)

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u



#Basics information relative to your event

#Target
targ_name = 'OB130341'
ra = 268.0312083333333
dec = -29.84611111111111

oversample = 2
padding = 10
directory = '/home/ebachele/Work/Euclid/OGLE-2013-BLG-0341/'

#Output filenames (all keyed off targ_name)
gaia_file = f'Gaia_{targ_name}.tbl'
resampled_file = f'resampled_{targ_name}.fits'
psf_file = f'psf_{targ_name}.fits'

ref_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

#Dowloading Gaia catalog around the target
res_gaia = get_Gaia(ref_coord,'./',gaia_file)


#Collecting data from the directory containing the stamps prepared by Manon Gilles scripts
Euclid_data = get_Euclid_data(directory)


#Refining the WCS for each stamps
NEWWCS = []  
centers = []
cats = []
matches = []

for ind,image in enumerate(Euclid_data[0]):
    
    newcs,shift,match,stars = recompute_WCS(Euclid_data[-2][ind],res_gaia,image)
    NEWWCS.append(newcs)
    cats.append(stars)
    matches.append(match)
    
    
#Selecting the reference image, where the target is the most in the center
dists = [(i[0]-image.shape[1]/2)**2+(i[1]-image.shape[0]/2)**2 for i in centers]
ref = np.argmin(dists)

#Computing transformation matrix between the image and the reference
matrices = []

for i,cat in enumerate(cats):
    mat = get_matrice_transformation(cats[ref],cat,matches[ref],matches[i],factor=oversample,center1=Euclid_data[0][0].shape[1]/2,center2=Euclid_data[0][0].shape[0]/2)
    matrices.append(mat)    
    
masks =  ~jnp.array(Euclid_data[-3]).astype(bool) 


#Opening or computing the stacking image with a factor oversample
try:

    reconstructed = fits.open(resampled_file)[0].data[0]
    reconstructed_sigma = fits.open(resampled_file)[0].data[1]

except:


    reconstructed, loss = reconstruct_highres_manual(jnp.array(Euclid_data[0]), np.array(matrices), oversample,weight =False,masks =  masks,guess = None,n_iters=25,lr=.01,bin_order=3, shift_order=-1,padding =padding,)
   
    reconstructed_sigma = np.sqrt(np.abs(reconstructed))

    hdu = fits.PrimaryHDU(data=[reconstructed,reconstructed_sigma])
    output_filename = resampled_file
    hdu.writeto(output_filename, overwrite=True)
    print(f"Image successfully saved to {output_filename}")
 




#Opening or estimating a basic psf
try:

    psf0 = fits.open(psf_file)[0].data
    
except:

    star_positions,full_catalog = detect_stars(np.array(reconstructed), 2, threshold_sigma=0.5, oversize=1,peakmax=30000)

    cutouts,cutouts_psf,rel_positions = cut_cutouts(np.array(reconstructed),star_positions[-300:-100]-0.5,size=15)
  

    psf0 = psf0_from_cutouts(np.array(cutouts_psf))
    fits.HDUList([ fits.PrimaryHDU(data=psf)]).writeto(psf_file,overwrite=True)



#Detect stars in the stack
star_positions,full_catalog = detect_stars(np.array(reconstructed), 2, threshold_sigma=0.5,peakmax=30000)

#Photometry of the stack
psfphot = psf_photometry(reconstructed,psf0,star_positions-0.5, error=reconstructed_sigma)

#Refining WCS of the stack
final_poses = np.c_[psfphot['x_fit'],psfphot['y_fit']]
final_wcs,final_match,_ = refine_wcs(res_gaia,final_poses-0.5,oversampled_wcs,shift=None,threshold = 1,factor=1,iteration = 5)

#Estimate ZP of the stack with Gaia G band
ZP = res_gaia[final_match[:,0]]['phot_g_mean_mag'].data.data+2.5*np.log10(np.array(psfphot['flux_fit'])[final_match[:,1]])

print('Zero Point',np.percentile(ZP,(16,50,84)))

#Save stack with updated WCS
header = final_wcs.to_header()
from astropy.io import fits

hdu = fits.PrimaryHDU(data=[reconstructed,reconstructed_sigma], header=header)


output_filename = resampled_file
hdu.writeto(output_filename, overwrite=True)
print(f"Image successfully saved to {output_filename}")