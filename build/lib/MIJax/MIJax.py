###Microlensing Images with JAX###

import jax
import jax.numpy as jnp
from jax import grad
import os
from jax.scipy.signal import convolve2d
from jax.numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import shift
from jax.scipy.ndimage import map_coordinates
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
import numpy as np
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.io import fits, ascii
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D
from skimage.registration import phase_cross_correlation
from astropy.wcs.utils import fit_wcs_from_points
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import distance
from functools import partial
from skimage.measure import ransac
from skimage.transform import warp, AffineTransform
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table
from scipy.ndimage import zoom
from scipy.optimize import least_squares
from scipy.ndimage import center_of_mass



class ImagePSF:

    def __init__(self, psf_size=21):
        self.size = psf_size
        self.hsize = psf_size // 2

        self.median_psf = None
        self.spline = None

        
        self.cutouts = None
        self.positions = None

        self.fluxes = None
        self.bkgs = None
        self.deltas = None

     
    # ---------------------------------------------------------
    # Cutout extraction
    # ---------------------------------------------------------
    def _extract_cutout(self, image, x, y,  max_center_offset=1):

        xi = int(round(x))
        yi = int(round(y))
        ny, nx = image.shape

        if xi - self.hsize < 0 or xi + self.hsize >= nx:
            return None
        if yi - self.hsize < 0 or yi + self.hsize >= ny:
            return None

        cut = image[
            yi - self.hsize:yi + self.hsize + 1,
            xi - self.hsize:xi + self.hsize + 1
        ]

        # sanity check: reject bad centering
        max_pos = np.unravel_index(np.argmax(cut), cut.shape)
        center = (self.hsize, self.hsize)

        offset = np.hypot(
            max_pos[0] - center[0],
            max_pos[1] - center[1]
        )

        if offset > max_center_offset:
            return None

        dx = x - xi
        dy = y - yi

        return cut.copy(), dx, dy

    # ---------------------------------------------------------
    # PSF spline
    # ---------------------------------------------------------
    def _build_spline(self,psf):
        y = np.arange(psf.shape[0])
        x = np.arange(psf.shape[1])

        spliner = RectBivariateSpline(
            x, y,
            psf,
            kx=5, ky=5, s=0
        )

        return spliner
        
    # ---------------------------------------------------------
    # Evaluate shifted PSF
    # ---------------------------------------------------------
    def evaluate(self, dx=0.0, dy=0.0):
       


        if self.oversample>1:

            y = np.arange(self.size*self.oversample)
            x = np.arange(self.size*self.oversample)


            psf = self.spline(y - dy*self.oversample, x - dx*self.oversample)
            
            psf = zoom(psf,1/self.oversample,order=5)*self.oversample**2
            
        else:
        
            y = np.arange(self.size)
            x = np.arange(self.size)


            psf = self.spline(y - dy, x - dx)
            
        return psf
        
   
    def residuals(self, param, im):

        dx, dy, flux, bg = param
        model = flux * self.evaluate(dx, dy) +bg

        return (im- model).ravel()/np.abs(im).ravel()**0.5
        
        
    # ---------------------------------------------------------
    # Main iterative fit
    # ---------------------------------------------------------
    def fit(self, image, sigma,positions0, n_iter=4, shift_max=0.5):

        positions = np.asarray(positions0, float)
        self.positions = positions0

        # -----------------------------
        # extract cutouts
        # -----------------------------
        self.oversample = 1
        positions = []
        positions.append(positions0)
        for i in range(n_iter):
            cuts, imgs, deltas = [], [], []

            for x, y, f in positions[-1]:
            
                if f <= 0 or not np.isfinite(f):
                    continue

                res = self._extract_cutout(image, x, y)
                if res is None:
                    continue

                cut, dx, dy = res
                img = shift(
                    zoom(cut / f, self.oversample, order=5),
                    (-dy * self.oversample, -dx * self.oversample),
                    mode='nearest',
                    order=5
                )

                imgs.append(img)
                cuts.append(cut)
                deltas.append([dx, dy])
            if len(imgs)<10:
                breakpoint()
                continue
            imgs = np.asarray(imgs)

            # Use previous iteration's PSF to reject outlier pixels
            if i > 0:

                diff = np.abs(imgs - self.median_psf)

                # Relative threshold, with a small floor to avoid problems
                # where the PSF is close to zero
                threshold = 10 * np.maximum(np.abs(self.median_psf), 1e-6)

                mask = diff < threshold

                # Mask rejected pixels rather than throwing away whole images
                imgs = np.where(mask, imgs, np.nan)

            self.cutouts = np.asarray(cuts)
            self.imgs = imgs
            self.deltas = np.asarray(deltas)

            # nanmedian ignores rejected pixels
            self.median_psf = np.nanmedian(self.imgs, axis=0)
        
#        for i in range(n_iter):
#            cuts, imgs, deltas = [], [] , []
#            for x, y, f in positions[-1]:
#            
#                if f <= 0 or not np.isfinite(f):
#                    continue
#                res = self._extract_cutout(image, x, y)
#                if res is None:
#                    continue
#                cut, dx, dy = res
#                img = shift(zoom(cut/f,self.oversample,order=5),(-dy*self.oversample,-dx*self.oversample),mode='nearest',order=5)
#                imgs.append(img)
#                cuts.append(cut)
#                deltas.append([dx, dy])


#            self.cutouts = np.asarray(cuts)
#            self.imgs = np.asarray(imgs)
#            self.deltas = np.asarray(deltas)
#            self.median_psf = np.median(self.imgs,axis=0)
#            
#            
            cy, cx = center_of_mass(self.median_psf)
            
            target_y = (self.median_psf.shape[0] - 1) / 2
            target_x = (self.median_psf.shape[1] - 1) / 2

            self.median_psf = shift(
            self.median_psf,
            (target_y - cy, target_x - cx),
            order=5,
            mode="nearest",
            )
            self.median_psf /= self.median_psf.sum()
            self.spline = self._build_spline(self.median_psf)
            #self.print_quality_metrics(central_region = self.hsize-2)

            psfphot2 = psf_photometry(image,self.median_psf,positions[-1], error=sigma,oversample = self.oversample)  
            SNR = psfphot2['flux_fit']/psfphot2['flux_err']

            quality_fit = np.nanpercentile(psfphot2['reduced_chi2'],75)
            
            
            valid_flags = (psfphot2['flags'] == 0) #& (psfphot2['reduced_chi2']<np.max((2,quality_fit)))

            if not np.any(valid_flags):
                raise ValueError("No stars passed the basic quality flags.")

            # 2. Dynamic SNR Thresholding (Adaptive to the field)
            all_valid_snr = SNR[valid_flags]
            # Adaptively pick a threshold: use the 75th percentile, but clamp it between 15 and 100
            #dynamic_snr_cutoff = np.clip(np.percentile(all_valid_snr, 75), 15, 100)
            dynamic_snr_cutoff = 10

            # 3. Spatial Isolation Check (Prevents blended profiles)
            from scipy.spatial import KDTree
            coords = np.c_[psfphot2['x_fit'], psfphot2['y_fit']]
            tree = KDTree(coords)
            # Find distance to the 2nd nearest neighbor (1st is the star itself)
            distances, _ = tree.query(coords, k=2)
            nearest_neighbor_dist = distances[:, 1] 

            min_isolation_distance = 3*2*self.psf_ee_radius() #self.hsize * 2.5 # Rule of thumb
            isolation_mask = nearest_neighbor_dist > min_isolation_distance

            # 4. Edge Proximity Mask
            ny, nx = image.shape
            edge_mask = (
                (psfphot2['x_fit'] > self.hsize) & 
                (psfphot2['x_fit'] < nx - self.hsize) &
                (psfphot2['y_fit'] > self.hsize) & 
                (psfphot2['y_fit'] < ny - self.hsize)
            )

            # 5. Combine everything into your final master mask
            mask = valid_flags & (SNR >= dynamic_snr_cutoff) & isolation_mask & edge_mask
            positions.append(np.c_[psfphot2['x_fit'],      psfphot2['y_fit'],psfphot2['flux_fit'],][mask])

        self.positions = positions
    # ---------------------------------------------------------
    # evaluation helpers
    # ---------------------------------------------------------
    def evaluate_shift(self, dx, dy):
        ix = int(np.round(dx))
        iy = int(np.round(dy))

        fx = dx - ix
        fy = dy - iy

        # fractional shift via spline
        psf = self.evaluate(fx, fy)

        ny, nx = psf.shape
        out = np.zeros_like(psf)

        # source slice
        sx0 = max(0, -ix)
        sx1 = min(nx, nx - ix)
        sy0 = max(0, -iy)
        sy1 = min(ny, ny - iy)

        # destination slice
        dx0 = max(0, ix)
        dx1 = dx0 + (sx1 - sx0)
        dy0 = max(0, iy)
        dy1 = dy0 + (sy1 - sy0)

        out[dy0:dy1, dx0:dx1] = psf[sy0:sy1, sx0:sx1]

        # renormalize (important after clipping edges)
        s = out.sum()
        if s > 0:
            out /= s

        return out
        
         
    def psf_ee_radius(self, frac=0.5):
        """
        Encircled-energy radius of a 2D PSF.

        Parameters
        ----------
        psf : 2D array
            PSF stamp
        frac : float
            Fraction of total flux (e.g. 0.5 = EE50)

        Returns
        -------
        r_frac : float
            Radius enclosing given flux fraction (in pixels)
        """

        psf = np.array(self.median_psf, dtype=float)

        # background subtraction
        psf = psf - np.median(psf)
        psf[psf < 0] = 0

        # center on peak
        y0, x0 = np.unravel_index(np.argmax(psf), psf.shape)

        y, x = np.indices(psf.shape)
        r = np.sqrt((x - x0)**2 + (y - y0)**2)

        # flatten and sort by radius
        r_flat = r.ravel()
        f_flat = psf.ravel()

        order = np.argsort(r_flat)
        r_sorted = r_flat[order]
        f_sorted = f_flat[order]

        cumsum = np.cumsum(f_sorted)
        total = cumsum[-1]

        if total <= 0:
            return np.nan

        target = frac * total

        return np.interp(target, cumsum, r_sorted)
    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------
    def print_quality_metrics(self,central_region=4):
        residual_sq_sum = 0
        max_residual = 0
        chi2 = 0
        ndof = 0
        nstars = 0
        npix = (self.cutouts[0][central_region:-central_region,central_region:-central_region].shape[0])**2
        radius = self.psf_ee_radius()
        central_region = int(central_region)
        kept = []
        for cut in self.cutouts:
           

            p0 = [0, 0, (cut).sum(), 0]

            res = least_squares(
                self.residuals,
                p0,
                args=((cut,)),
                method='lm'      # or 'trf'
            )
            
            dx,dy,flux,bkg = res.x
            psf = self.evaluate(dx, dy)
            
            residual = cut - flux*psf-bkg
            residual = residual[central_region:-central_region,central_region:-central_region]
            residual_sq_sum += np.sum(residual**2)
            max_residual = max(max_residual, np.max(np.abs(residual)))


            
            cutchi2 = np.sum((residual**2) / np.abs(cut)[central_region:-central_region,central_region:-central_region])
            
           
            
            chi2 += cutchi2
            ndof += npix - 1
            
            nstars += 1

        if nstars == 0:
            print("No stars for diagnostics")
            return


        rms = np.sqrt(residual_sq_sum / (nstars * npix))
        chi2_dof = chi2 / ndof if ndof > 0 else np.nan


        print("\nPSF QUALITY METRICS")
        print("-------------------")
        print(f"Stars used   : {nstars}")
        print(f"RMS residual : {rms:.3e}")
        print(f"Max residual : {max_residual:.3e}")
        print(f"Chi2 / dof   : {chi2_dof:.3f}")
        print(f"FWHM : {2*radius:.3f}")    



#Simulation functions for testing
def gaussian_psf(size, sigma, oversample=1, x0=0.0, y0=0.0):
    """
    Generate an oversampled Gaussian PSF with optional centroid shift.

    Parameters
    ----------
    size : int
        PSF size (number of pixels along one axis).
    sigma : float
        Gaussian sigma (in pixels).
    oversample : int
        Oversampling factor.
    x0, y0 : float
        Subpixel shift of the centroid in high-res coordinates.

    Returns
    -------
    psf : 2D array
        Normalized Gaussian PSF with shifted centroid.
    """
    """Generate an oversampled Gaussian PSF."""
    ax = np.arange(0,size)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-((xx-x0)**2 + (yy-y0)**2) / (2. * sigma**2))

    return psf / psf.sum()
    
    
def simulate_scene(size, oversample, n_stars, true_psf, background):
    """Simulate a high-res scene with stars and PSF."""
    scene = np.full((size * oversample, size * oversample), background, dtype=float)
    psf = true_psf
    psf_size = psf.shape[0]
    half_psf = psf_size // 2

    star_positions = [
        (np.random.uniform(0, size * oversample),
         np.random.uniform(0, size * oversample),
         np.random.uniform(8e4, 1e6))
        for _ in range(n_stars)
    ]

    for x, y, f in star_positions:
        # Integer part (nearest pixel)
        ix, iy = int(np.floor(x)), int(np.floor(y))

        # Subpixel offsets (fractional part)
        dx, dy = x - ix, y - iy

        # Shift PSF by fractional offsets only
        shifted = shift(psf, shift=(dy, dx), order=5, mode='nearest')

        # Define cutout placement
        x_start = ix - half_psf
        y_start = iy - half_psf
        x_end = x_start + psf_size
        y_end = y_start + psf_size

        # Clip to image bounds
        if x_start < 0 or y_start < 0 or x_end > scene.shape[1] or y_end > scene.shape[0]:
            continue

        scene[y_start:y_end, x_start:x_end] += shifted * f

    return scene, star_positions
    
    
    
def downsample_scene(image, factor, shifts):
    """Generate downsampled images from shifted high-res image."""
    low_res_images = []
    for dx, dy in shifts:
        shifted = shift_image_fft(image, dy*factor, dx*factor)
        #shifted = shift_image(img, dy, dx, order=1)
        low_res = resize_sum(shifted, factor)
        
        low_res_images.append(add_poisson_noise(low_res))
        
    return np.array(low_res_images)
    
def add_poisson_noise(images):
    """Apply Poisson noise to a stack of images."""
    return np.array([np.random.poisson(np.abs(img)) for img in images])
   

# Reconstruction functions

def resize_sum(img, factor):
    """
    Downsample by integer factor using sum (flux-conserving).
    
    Parameters
    ----------
    img : 2D array
        High-res image
    factor : int
        Downsampling factor
    
    Returns
    -------
    downsampled : 2D array
        Low-res image, flux-conserving
    """
    H, W = img.shape
    H_low, W_low = H // factor, W // factor
    #img = img[:H_low*factor, :W_low*factor] 
    img = jax.lax.dynamic_slice(img, (0,0), (H_low*factor, W_low*factor))# crop to multiple of factor
    img = img.reshape(H_low, factor, W_low, factor)
    return img.sum(axis=(1, 3))   
    
    
def shift_image_fft(img, dy, dx, pad=100):
    """
    Subpixel shift using Fourier shift theorem with optional zero-padding.
    
    Args:
        img : (H, W) array
        dy, dx : shifts in pixels (positive = down/right)
        pad : if True, pad with zeros to avoid wrap-around
        
    Returns:
        shifted image, same size as input
    """
    H, W = img.shape

    if pad > 0:
        img = jnp.pad(img, ((pad, pad), (pad, pad)))
        H2, W2 = img.shape
    else:
        H2, W2 = H, W

    mask = jnp.ones((H,W))
    
    # Frequency grid
    fy = fftfreq(H2)
    fx = fftfreq(W2)
    FX, FY = jnp.meshgrid(fx, fy)

    # Fourier shift
    img_f = fft2(img)
    phase = jnp.exp(-2j * jnp.pi * (dx * FX + dy * FY))
    

    shifted = jnp.real(ifft2(img_f * phase))
    #shifted = shifted.at[:int(dy), :].set(0)   # zero out first int(y) rows
    #shifted = shifted.at[:, :int(dx)].set(0)   # zero out first int(x) columns
    shifted_mask =     shift_image(mask,dy,dx,order=0).astype(bool)
    #shifted = jnp.where(shifted_mask, shifted, 0.0)


    #shifted = shifted.at[~shifted_mask].set(0)

    #breakpoint()
    if pad > 0:
        shifted = shifted[pad:-pad, pad:-pad]

    shifted = jnp.where(shifted_mask, shifted, 0)
    return shifted 
    
    
   
def shift_image(img, dy, dx, order=0):
    """Subpixel shift using real-space cubic interpolation (non-Fourier)."""

    if order==0:
    
        ny, nx = img.shape
        y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        coords = jnp.array([y - dy, x - dx])

        model =  map_coordinates(img, coords, order=0, mode="constant", cval=0.0)
    
    if order==1:
        ny, nx = img.shape
        y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        coords = jnp.array([y - dy, x - dx])

    
        model =  map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    
    if order==3:
    
        model =  shift_image_cubic(img, dy,dx)
        
    if order==-1:
    
        model = shift_image_fft(img, dy, dx,100)
        
    return model

def matrix_image(img,transfo, order=0):
    """Subpixel shift using real-space cubic interpolation (non-Fourier)."""
    
    if order==-1:
         return   fft_affine_resample(img, transfo,pad=img.shape[0])
    
         
    A_xy = transfo[:, :2]
    b_xy = transfo[:, 2]

    perm = jnp.array([1, 0])

    A = A_xy[perm][:, perm]
    b = b_xy[perm]

    ny, nx = img.shape

    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    coords_out = jnp.stack([y, x], axis=0).reshape(2, -1)

    # invert transform
    Ainv = jnp.linalg.inv(A)
    coords_in = Ainv @ (coords_out - b[:, None])
    coords_in = coords_in.reshape(2, ny, nx)
    
  
         
    if order==3:
        result = map_coordinates_cubic(img,coords_in, mode="constant",
                                cval=0.0
                                )
    else:
        result = map_coordinates(
                                img,
                                coords_in,
                                order=order,
                                mode="constant",
                                cval=0.0
                                )
                                
    return result

    
@partial(jax.jit, static_argnames=("pad",))
def fft_affine_resample(img: jnp.ndarray, transfo: jnp.ndarray, pad: int = 0) -> jnp.ndarray:
    """
    Apply a full affine transform to a 2D image using Fourier resampling.

    Args:
        img: (H, W) input image
        transfo: 2x3 affine matrix [[a, b, tx], [c, d, ty]]
        pad: optional zero-padding around image to avoid aliasing

    Returns:
        Transformed image of shape (H, W)
    """
    H, W = img.shape
    A = transfo[:, :2]   # 2x2 linear part
    b = transfo[:, 2]    # translation

    # Optional zero-padding
    if pad > 0:
        img = jnp.pad(img, ((pad, pad), (pad, pad)))
        H, W = img.shape

    # Fourier coordinates
    ky = jnp.fft.fftfreq(H)[:, None]  # (H,1)
    kx = jnp.fft.fftfreq(W)[None, :]  # (1,W)
    kx = jnp.broadcast_to(kx, (H, W))
    ky = jnp.broadcast_to(ky, (H, W))

    # 2D FFT of image
    img_fft = jnp.fft.fft2(img)

    # Map Fourier coordinates using inverse transpose of A
    AinvT = jnp.linalg.inv(A).T
    # Stack and apply AinvT
    u = jnp.stack([kx, ky], axis=-1)        # (H,W,2)
    u_resampled = u @ AinvT                 # (H,W,2)
    kx_res = u_resampled[..., 0]
    ky_res = u_resampled[..., 1]

    # Apply translation phase factor
    phase = jnp.exp(-2j * jnp.pi * (kx_res * b[0] + ky_res * b[1]))

    # Apply affine resampling in Fourier space, scale flux
    img_fft_resampled = img_fft * phase / jnp.abs(jnp.linalg.det(A))

    # Inverse FFT
    img_out = jnp.fft.ifft2(img_fft_resampled)
    img_out = jnp.real(img_out)

    # Crop padding if any
    if pad > 0:
        img_out = img_out[pad:-pad, pad:-pad]

    return img_out

    
def map_coordinates_cubic(
    img,
    coords_in,
    mode="constant",
    cval=0.0,
):
    """
    Bicubic interpolation for 2D images in JAX.

    img: (H, W)
    coords_in: (2, N)
               coords_in[0] = row coords
               coords_in[1] = col coords
    """

    order= 3

    H, W = img.shape
    rows = coords_in[0]
    cols = coords_in[1]

    r0 = jnp.floor(rows).astype(int)
    c0 = jnp.floor(cols).astype(int)

    dr = rows - r0
    dc = cols - c0

    # 4x4 neighborhood offsets
    offsets = jnp.array([-1, 0, 1, 2])

    def sample_point(r_base, c_base, dr, dc):
        result = 0.0

        for i in offsets:
            for j in offsets:
                rr = r_base + i
                cc = c_base + j

                if mode == "constant":
                    valid = (
                        (rr >= 0) & (rr < H) &
                        (cc >= 0) & (cc < W)
                    )
                    val = jnp.where(valid, img[rr, cc], cval)
                elif mode == "nearest":
                    rr = jnp.clip(rr, 0, H-1)
                    cc = jnp.clip(cc, 0, W-1)
                    val = img[rr, cc]
                else:
                    raise NotImplementedError("Mode not supported")

                w = _cubic_kernel(i - dr) * _cubic_kernel(j - dc)
                result += val * w

        return result

    return jax.vmap(sample_point)(r0, c0, dr, dc)
    
def _cubic_kernel(x, a=-0.5):
    """Keys cubic kernel (Catmull-Rom when a=-0.5)."""
    absx = jnp.abs(x)
    absx2 = absx**2
    absx3 = absx**3

    return jnp.where(
        absx <= 1,
        (a + 2)*absx3 - (a + 3)*absx2 + 1,
        jnp.where(
            absx < 2,
            a*absx3 - 5*a*absx2 + 8*a*absx - 4*a,
            0.0
        )
    )



def shift_image_cubic(img, dy, dx):
    H, W = img.shape

    y = jnp.arange(H)
    x = jnp.arange(W)
    Y, X = jnp.meshgrid(y, x, indexing="ij")

    Yf = Y - dy
    Xf = X - dx

    y0 = jnp.floor(Yf).astype(jnp.int32)
    x0 = jnp.floor(Xf).astype(jnp.int32)

    out = jnp.zeros_like(img)

    for j in range(-1, 3):
        for i in range(-1, 3):
            wy = cubic_kernel(Yf - (y0 + j))
            wx = cubic_kernel(Xf - (x0 + i))
            w = wy * wx

            yy = jnp.clip(y0 + j, 0, H-1)
            xx = jnp.clip(x0 + i, 0, W-1)

            out = out + w * img[yy, xx]

    return out
    
    
def simulate_lowres_from_model(model, shifts, factor, bin_order=0, shift_order=0):
    """Simulate low-res images from high-res model with PSF and shifts."""

    lowres_images = []


    h, w = model.shape
    out_shape = (h // factor, w // factor)

    #for (dx,dy) in shifts:
    for transfo in shifts:
        
        if transfo.shape != (2,3):
            dx,dy = transfo
            shifted = shift_image(model,dy*factor,dx*factor,order=shift_order)
        
        else:
            shifted = matrix_image(model,transfo,order=shift_order)
        im = resize_sum(shifted, factor)
        lowres_images.append(im)
        
    return jnp.stack(lowres_images)
    
    
def smoothness_penalty(model):
    """Compute squared Laplacian for smoothness."""
    lap = (
        model[1:-1, 2:] + model[1:-1, :-2] +
        model[2:, 1:-1] + model[:-2, 1:-1] -
        4 * model[1:-1, 1:-1]
    )
    return jnp.sum(lap ** 2)
    

def loss_fn_smooth(model, true_lowres, shifts,  factor, weight = False,masks=None,smooth_lambda=0.01,bin_order=0, shift_order=0,padding=0):
    """MSE + smoothness constraint."""

    mse = loss_fn(model, true_lowres, shifts,  factor,weight,masks,bin_order,shift_order,padding)
    #smooth = smoothness_penalty(model)
    return +mse#+0.1*smooth 

def loss_fn(model, true_lowres, shifts, factor,weight=False,masks = None,bin_order=0, shift_order=0,padding=0):


    simulated = simulate_lowres_from_model(model,  shifts, factor,bin_order, shift_order)
    
    mask = jnp.zeros(simulated.shape).astype(bool)
    mask = mask.at[:, padding:-padding, padding:-padding].set(True)
    mask = (simulated!=0) & (mask) & (true_lowres!=0)
   
    if masks is not None:
       mask = mask & masks
    resid = simulated - true_lowres
    return jnp.sum(jnp.where(mask, resid**2, 0.0))


def gaussian_kernel(size: int, sigma: float):
    """Normalized 2D Gaussian kernel."""
    ax = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / jnp.sum(kernel)


def gaussian_blur_fluxconserving(image, sigma=0.5):
    """Apply Gaussian blur but preserve total flux."""

    size = 15
    kernel = gaussian_kernel(size, sigma)
    
    blurred = convolve2d(image, kernel, mode="same")
   
    return blurred


#@partial(jax.jit, static_argnames=("n_iters","factor", "bin_order", "shift_order", "padding"))
def reconstruct_highres_manual(true_lowres,  shifts, factor, weight = False,masks = None,guess=None,n_iters=300, lr=1e-1,bin_order=0, shift_order=0,padding=0):
    """Reconstruct high-res model using manual gradient descent (JAX)."""

    M, H_low, W_low = true_lowres.shape
    H_high, W_high = H_low * factor, W_low * factor


    if guess is None:
        model = jnp.ones((H_high, W_high)) * jnp.median(true_lowres[0])
    else:
        model = guess
        
    shifts = jnp.array(shifts)
    #losses = []

    #loss_fn_smooth_jit = jax.jit(loss_fn_smooth, static_argnames=("factor","bin_order","shift_order","padding"))
    loss_fn_smooth_jit = loss_fn_smooth
    grad_fn = jax.grad(loss_fn_smooth_jit)

    smooth_lambda = 0
    for i in range(n_iters):
        loss = loss_fn_smooth_jit(model, true_lowres,  shifts, factor,weight, masks,smooth_lambda,bin_order, shift_order,padding)
        grads = grad_fn(model, true_lowres, shifts, factor,weight,masks,smooth_lambda,bin_order, shift_order,padding)
        
        model = model - lr * grads
        #sigma = 0.5 * (1 - i/n_iters)  # decrease blur over iterations
        #model = gaussian_blur_fluxconserving(model, sigma=sigma)
        model = gaussian_blur_fluxconserving(model, sigma=0.5) #Should be made optional

        #losses.append(loss)

        #if i % (n_iters // 10) == 0 or i == n_iters - 1:
        print(f"Iter {i:4d}: loss = {loss:.6f}")
        #print(i)
    
    model = np.array(model)
    
    return model,None #losses
  

#Stars/PSF functions
def detect_stars(image, psf_size, threshold_sigma=5.0, peakmax=55000,brightest=None,min_separation = 1):
    """
    Detect stars in a high-res image and extract cutouts.

    Parameters
    ----------
    image : 2D array
        High-resolution image.
    psf_size : int
        Size of the PSF cutout in pixels (high-res pixels).
    threshold_sigma : float
        Detection threshold in units of background RMS.


    Returns
    -------
    star_positions : list of tuples (x, y, flux)
        Detected star positions in pixel coordinates and their flux.
    cutouts : list of 2D arrays
        Square cutouts of each star of size `psf_size`.
    """

    work_image = image.copy()

    for i in range(1):

        # Estimate background RMS for thresholding
        values, counts = np.unique(np.round(image,0), return_counts=True)
        #bkg_sigma = mad_std(work_image)
        bkg = values[np.argmax(counts)]
        mmm = (values>0) & (values<bkg) & (counts>counts.max()/2.0)
        
        peak = np.argmax(counts)

        # refined mode
        y1, y2, y3 = counts[peak-1:peak+2]
        delta = 0.5*(y1-y3)/(y1 - 2*y2 + y3)
        bkg = values[peak] + delta

        # refined half-max
        halfmax = y2/2.0

        left = np.where(counts[:peak] < halfmax)[0][-1]

        x0, y0 = values[left], counts[left]
        x1, y1 = values[left+1], counts[left+1]

        x_half = x0 + (halfmax-y0)*(x1-x0)/(y1-y0)

        bkg_sigma = (bkg - x_half)/np.sqrt(2*np.log(2))

        print(bkg,bkg_sigma)

        #b#kg_sigma = bkg**0.5

        #mmm = np.abs(image-bkg)<3*bkg_sigma
        #bkg2,bkg_sigma2 = image[mmm].mean(),image[mmm].std()
        
        #bbkkgg = np.median(image)
        #mmm = np.abs(image - bbkkgg) < 3*np.sqrt(bbkkgg)
        #sigma = image[mmm].std()

        #for _ in range(1):
        #    print(bbkkgg, sigma)

        #    mask = np.abs(image - bbkkgg) < 3*sigma
         #   bbkkgg = np.percentile(image[mask],(16,50,84))

        #    sigma = (bbkkgg[2]-bbkkgg[0])/2
        #    bbkkgg = bbkkgg[1]
        #    sigma = image[mask].std()
        #print(bkg,bkg_sigma,bkg2,bkg_sigma2,bbkkgg,sigma)
        #breakpoint()
        #bkg = bbkkgg
        #bkg_sigma = sigma
        #print( values[np.argmax(counts)], values[np.argmax(counts)]**0.5,bkg,bkg_sigma,bbkkgg,sigma)
        cut =threshold_sigma*bkg_sigma

        #print('Star cut',cut)
        # DAOStarFinder: peak detection
        daofind = DAOStarFinder(fwhm=psf_size, threshold=cut,peak_max=peakmax,min_separation=min_separation, sharpness_range=(0.2,1.0))
        sources = daofind(work_image-bkg)
        
        order = sources['flux'].argsort()
        sources = sources[order]
        
        if sources is None:
            return [],[]
            
        else:
            
            stars = np.c_[sources['x_centroid'],sources['y_centroid'],sources['flux'],sources['mag']]
            mask = np.isfinite(stars[:,3])

            
            return stars[mask],sources, bkg
    
def cut_cutouts(image, stars, size=11): 
    
    star_positions = []
    cutouts = []
    cutouts2 = []
    rel_positions = []
    
    half_size = int(size // 2)
    for src in stars:
            x, y, flux,mag = src
            ix, iy = int(x), int(y)
            
            # Extract cutout
            y_start = max(iy - half_size, 0)+1
            y_end = y_start+size
            x_start = max(ix - half_size, 0)+1
            x_end = x_start+size
            
            
            # Extract cutout2
            y_start2 = max(iy - size, 0)+1
            y_end2 = y_start2+2*size+1
            x_start2 = max(ix - size, 0)+1
            x_end2 = x_start2+2*size+1
            
            
            
            cutout = image[y_start:y_end, x_start:x_end].copy()
            cutout2 =  image[y_start2:y_end2, x_start2:x_end2].copy()

            if (cutout.shape == (size,size)) & (cutout2.shape == (2*size+1,2*size+1)):

                cutouts.append(cutout)
                cutouts2.append(cutout2)
              
                x_rel = x - x_start-half_size
                y_rel = y - y_start-half_size
                rel_positions.append((x_rel, y_rel))

    return cutouts, cutouts2,rel_positions
    
    
def psf_photometry(image,psf,guess, error= None,n_odd=5, oversample = 1):
 
    from photutils.psf import PSFPhotometry, ImagePSF
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.table import Table
    init_tbl = Table()
    init_tbl['x_0'] = guess[:,0]
    init_tbl['y_0'] = guess[:,1]
    init_tbl['flux'] = guess[:,2]  
    psf_model = ImagePSF(psf,oversampling=oversample)
    #psf_model.x_0.fixed = True
    #psf_model.y_0.fixed = True
    psf_phot = PSFPhotometry(
    psf_model=psf_model,
    fitter=LevMarLSQFitter(),xy_bounds = [1,1],
    fit_shape=n_odd, progress_bar=True,
    )


    result = psf_phot(image, init_params=init_tbl,error=error,)

    return result
    
def psf0_from_cutouts(cutouts):
    
    psfs = [i/i.sum() for i in cutouts]
    
    reconstructed_psf = np.median(psfs,axis=0)
    reconstructed_psf /= reconstructed_psf.sum()
    psf = np.copy(reconstructed_psf) 
       

    return psf
    
 
def refine_psf(image,psf,poses,error,iterations = 20):

    poses2 = np.copy(poses)
    poses3 = np.copy(poses)
    alpha = 0.8


    for i in range(iterations):

        cutouts,cutouts_psf,rel_positions = cut_cutouts(np.array(image),poses3,size=int(psf.shape[0]/2))
        
        psfnew = psf0_from_cutouts(np.array(cutouts_psf))
        psf = (1-alpha)*psf+alpha*psfnew
        psf /= psf.sum()
        
        psfphot = psf_photometry(image,psf,poses2, error=error)

        mask2 = psfphot['reduced_chi2']<np.nanpercentile(psfphot['reduced_chi2'],50)+0.1

        poses2 = np.c_[psfphot['x_fit'],psfphot['y_fit'],psfphot['flux_fit'],-2.5*np.log10(psfphot['flux_fit'])]
        poses3 = poses2[mask2]-0.5

        print(i,len(poses3),np.nanmedian(psfphot['reduced_chi2'][mask2]),np.nanmin(psfphot['reduced_chi2'][mask2]),np.nanmedian(psfphot['qfit']),np.nanmin(psfphot['qfit']))

        
    return psf
    
#Data access  

def get_Gaia(ref_coord, Gaia_directory, Gaia_name,size=1.5, limit= 100000):

    path = Gaia_directory + Gaia_name

    try:
        res_gaia = Table.read(path, format='ascii.ecsv')

    except:

#        width = u.Quantity(size, u.arcmin)
#        height = u.Quantity(size, u.arcmin)

#        Gaia.ROW_LIMIT = limit
#        columns = [
#                "source_id",
#                "ra",
#                "dec",
#                "parallax",
#                "pmra",
#                "pmdec",
#                "phot_g_mean_mag",
#                "phot_bp_mean_mag",
#                "phot_rp_mean_mag",
#                "ruwe",
#            ]
#        res_gaia = Gaia.query_object_async(
#            coordinate=ref_coord,
#            width=width,
#            height=height,
#            columns=columns
#        )

#        res_gaia.write(path, format='ascii.ecsv', overwrite=True)
               # VizieR query configuration
        Vizier.ROW_LIMIT = limit

        columns = [
            "Source",
            "RA_ICRS",
            "DE_ICRS",
            "Plx",
            "pmRA",
            "pmDE",
            "Gmag",
            "BPmag",
            "RPmag",
            "RUWE",
        ]

        v = Vizier(
            columns=columns,
            row_limit=limit
        )

        # Query a square region
        result = v.query_region(
            ref_coord,
            width=size * u.arcmin,
            height=size * u.arcmin,
            catalog="I/355/gaiadr3"
        )

        if len(result) == 0:
            return Table()

        res_gaia = result[0]

        # Rename columns to match the Gaia archive output
        res_gaia.rename_column("Source", "source_id")
        res_gaia.rename_column("RA_ICRS", "ra")
        res_gaia.rename_column("DE_ICRS", "dec")
        res_gaia.rename_column("Plx", "parallax")
        res_gaia.rename_column("pmRA", "pmra")
        res_gaia.rename_column("pmDE", "pmdec")
        res_gaia.rename_column("Gmag", "phot_g_mean_mag")
        res_gaia.rename_column("BPmag", "phot_bp_mean_mag")
        res_gaia.rename_column("RPmag", "phot_rp_mean_mag")

        res_gaia.write(path, format="ascii.ecsv", overwrite=True)

    # Mask bad G-band data
    res_gaia = res_gaia[
        np.isfinite(res_gaia["phot_g_mean_mag"])
    ]

    return res_gaia
    #masking bad data    
    res_gaia = res_gaia[np.isfinite(res_gaia['phot_g_mean_mag'])]
    
    return res_gaia
   
    
def get_Euclid_data(data_directory):
    "Expect data with same format as defined by Manon Gilles scripts"



    data = [data_directory+i for i in os.listdir(data_directory) if 'sci.fits' in i]

    #tweakreg.TweakReg(data)

    #breakpoint()
    images = []
    masks = []
    WCS = []

    names = []
    cuts = []
    size = 11
    cats = []
    count = 0


    for dat in data[:]:
        
        aa = fits.open(dat)
        header = aa[0].header


        names.append(dat)
        image = aa[0].data
        shape = image.shape
        www = wcs.WCS(aa[0].header)
        WCS.append(www)
        images.append(image)

        
    images = crop_images_to_smallest(images)

    rmss = []
    for i,dat in enumerate(data):


        try:
            aa = fits.open(dat.replace('sci','rms'))
            header = aa[0].header

            names.append(dat)
            image = aa[0].data

            rmss.append(image)
        except:
            rmss.append(np.ones(images[i].shape))
            
    rmss = crop_images_to_smallest(rmss)

    flags = []


    for i,dat in enumerate(data):


        try:
            aa = fits.open(dat.replace('sci','flg'))
            header = aa[0].header

            names.append(dat)
            image = aa[0].data
            flag  = (image%2==1) | (image==2**1)| (image==2**2)| (image==2**3)| (image==2**4)| (image==2**5)| (image==2**7)| (image==2**8)| (image==2**9)| (image==2**12)| (image==2**15)| (image==2**17)| (image==2**21)| (image==2**22)

            flags.append(flag)

        except:
            flags.append(np.ones(images[i].shape))

    flags = crop_images_to_smallest(flags)


    bkgs =[]
    for i,dat in enumerate(data):


        try:
            aa = fits.open(dat.replace('sci','bkg'))
            header = aa[0].header

            names.append(dat)
            image = aa[0].data
          
            bkgs.append(image)

        except:
            bkgs.append(np.zero(images[i].shape))

    bkgs= crop_images_to_smallest(bkgs)
    
    
    with open(data_directory+'PSFs.pkl', 'rb') as f:
        psfs = pickle.load(f)

    
    
    return images, bkgs, rmss, flags, WCS,psfs
    
def crop_images_to_smallest(images):
    """
    Center-crop all images in a list to the size of the smallest image.
    
    Parameters:
        images (list of np.ndarray): List of images (H x W) or (H x W x C)
    
    Returns:
        list of np.ndarray: Cropped images with identical shape
    """
    if not images:
        return []

    # Get minimum height and width
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)

    cropped_images = []

    for img in images:
        h, w = img.shape[:2]

        # Compute top-left corner for center crop
        start_y = (h - min_h) // 2
        start_x = (w - min_w) // 2

        cropped = img[start_y:start_y + min_h,
                      start_x:start_x + min_w]

        cropped_images.append(cropped)

    return cropped_images
    
    
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
from scipy.optimize import linear_sum_assignment


def optimize_affine_matching(
    gaia_xy,
    detected_xy,
    radius=1.0,
    rotation_range=5.0,
    translation_range=200.0,
    scale_range=0.2,
    seed=42,
):
    """
    Optimise une transformation affine pour maximiser le nombre
    de correspondances uniques dans un rayon donné.

    gaia_xy : positions Gaia projetées par la WCS initiale
    detected_xy : positions des étoiles détectées
    radius : seuil de correspondance en pixels
    rotation_range : rotation maximale recherchée, en degrés
    translation_range : translation maximale recherchée, en pixels
    scale_range : variation d'échelle autorisée autour de 1
    """

    gaia_xy = np.asarray(gaia_xy, dtype=float)
    detected_xy = np.asarray(detected_xy, dtype=float)

    gaia_xy = gaia_xy[np.isfinite(gaia_xy).all(axis=1)]
    detected_xy = detected_xy[np.isfinite(detected_xy).all(axis=1)]

    tree = cKDTree(detected_xy)

    # Paramètres :
    # [a, b, c, d, tx, ty]
    # x' = a*x + b*y + tx
    # y' = c*x + d*y + ty

    def score(params):
        a, b, c, d, tx, ty = params

        transformed = gaia_xy @ np.array([
            [a, c],
            [b, d]
        ]) + np.array([tx, ty])

        distances, indices = tree.query(
            transformed,
            distance_upper_bound=radius
        )

        valid = indices < len(detected_xy)

        # Pénaliser les distances tout en maximisant les matches
        pairs = sorted(
            zip(distances[valid], np.where(valid)[0], indices[valid])
        )

        used_gaia = set()
        used_detected = set()
        n_matches = 0
        total_dist = 0

        for dist, gi, di in pairs:
            if gi not in used_gaia and di not in used_detected:
                used_gaia.add(gi)
                used_detected.add(di)
                n_matches += 1
                total_dist += dist

        return -n_matches + 1e-3 * total_dist

    bounds = [
        (1-scale_range, 1+scale_range),  # a
        (-scale_range, scale_range),     # b
        (-scale_range, scale_range),     # c
        (1-scale_range, 1+scale_range),  # d
        (-translation_range, translation_range),  # tx
        (-translation_range, translation_range),  # ty
    ]

    result = differential_evolution(
        score,
        bounds,
        seed=seed,
        maxiter=500,
        popsize=15,
        polish=False,
        atol=0.1
    )

    a, b, c, d, tx, ty = result.x

    matrix = np.array([
        [a, b, tx],
        [c, d, ty],
        [0, 0, 1]
    ])

    transformed = gaia_xy @ matrix[:2, :2].T + matrix[:2, 2]

    distances, indices = tree.query(
        transformed,
        distance_upper_bound=radius
    )

    valid = indices < len(detected_xy)

    candidates = sorted(
        zip(distances[valid], np.where(valid)[0], indices[valid])
    )

    used_gaia = set()
    used_detected = set()
    matches = []

    for dist, gi, di in candidates:
        if gi not in used_gaia and di not in used_detected:
            used_gaia.add(gi)
            used_detected.add(di)
            matches.append((gi, di, dist))

    matches = np.asarray(matches)

    return matrix, matches, -result.fun
    
    
def refine_affine_ransac(
    gaia_xy,
    detected_xy,
    matrix,
    radius=1.0,
    residual_threshold=1.0,
    max_trials=2000,
):
    """
    Raffine une transformation affine initiale avec RANSAC.

    Parameters
    ----------
    gaia_xy : ndarray (N, 2)
        Positions Gaia projetées en pixels.
    detected_xy : ndarray (M, 2)
        Positions des étoiles détectées.
    matrix : ndarray (3, 3)
        Transformation affine initiale.
    radius : float
        Rayon maximal pour associer deux étoiles, en pixels.
    residual_threshold : float
        Seuil des résidus RANSAC, en pixels.
    max_trials : int
        Nombre maximal d'itérations RANSAC.

    Returns
    -------
    model : AffineTransform
        Transformation affine raffinée.
    matches : ndarray (K, 3)
        Indices Gaia, indices détection et distances.
    """

    gaia_xy = np.asarray(gaia_xy, dtype=float)
    detected_xy = np.asarray(detected_xy, dtype=float)

    # Supprimer les positions invalides en conservant les indices d'origine
    valid_g = np.isfinite(gaia_xy).all(axis=1)
    valid_d = np.isfinite(detected_xy).all(axis=1)

    gaia_indices = np.flatnonzero(valid_g)
    detected_indices = np.flatnonzero(valid_d)

    gaia = gaia_xy[valid_g]
    detected = detected_xy[valid_d]

    # Appliquer la transformation affine initiale
    predicted = (
        gaia @ matrix[:2, :2].T
        + matrix[:2, 2]
    )

    # Recherche des correspondances candidates
    tree = cKDTree(detected)
    neighborhoods = tree.query_ball_point(predicted, radius)

    candidates = []

    for gi, neighbors in enumerate(neighborhoods):
        for di in neighbors:
            dist = np.linalg.norm(predicted[gi] - detected[di])
            candidates.append((dist, gi, di))

    # Apparier les étoiles de manière unique
    candidates.sort()

    used_g = set()
    used_d = set()
    pairs = []

    for dist, gi, di in candidates:
        if gi not in used_g and di not in used_d:
            used_g.add(gi)
            used_d.add(di)
            pairs.append((gi, di))

    pairs = np.asarray(pairs, dtype=int)

    if len(pairs) < 3:
        raise RuntimeError(
            f"Seulement {len(pairs)} correspondances candidates."
        )

    # RANSAC : estimer l'affine à partir des correspondances candidates
    model, inliers = ransac(
        (
            gaia[pairs[:, 0]],
            detected[pairs[:, 1]]
        ),
        AffineTransform,
        min_samples=3,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        rng=np.random.default_rng(42),
    )

    if model is None or inliers is None or inliers.sum() < 3:
        raise RuntimeError("RANSAC n'a pas trouvé de transformation fiable.")

    # Recalculer les correspondances avec la transformation raffinée
    predicted = model(gaia)

    neighborhoods = tree.query_ball_point(predicted, radius)
    candidates = []

    for gi, neighbors in enumerate(neighborhoods):
        for di in neighbors:
            dist = np.linalg.norm(predicted[gi] - detected[di])
            candidates.append((dist, gi, di))

    candidates.sort()

    used_g = set()
    used_d = set()
    matches = []

    for dist, gi, di in candidates:
        if gi not in used_g and di not in used_d:
            used_g.add(gi)
            used_d.add(di)

            matches.append((
                gaia_indices[gi],
                detected_indices[di],
                dist
            ))

    return model, np.asarray(matches)
    
    
def recompute_WCS(init_wcs,gaia_catalog,image,factor=1,psf_size=1,
        threshold_sigma=3,
        peakmax=40000,
        min_separation=1):


        star_positions,full_catalog, bkg = detect_stars(image, psf_size,threshold_sigma=threshold_sigma,
        peakmax=peakmax,
        min_separation=min_separation)  
        
        if factor>=2:
            #shift = [0,0]
            matrix = None
        else:
        
            #mod = build_image_model(init_wcs,gaia_catalog, shape=image.shape, psf_sigma=factor,factor=factor)
            #shift, error, diffphase = phase_cross_correlation(mod, image,upsample_factor=100)    
            
            radec = init_wcs.pixel_to_world(star_positions[:,0],star_positions[:,1])
            
            #gaia = np.column_stack((gaia_catalog['ra'].value,gaia_catalog['dec'].value))
            order = gaia_catalog['phot_g_mean_mag'].argsort().value[::-1]
            #gaia = gaia[order]

            # Project Gaia into the initial WCS pixel coordinates
            gaia_coords = init_wcs.world_to_pixel(SkyCoord( gaia_catalog['ra'][order],gaia_catalog['dec'][order]))


            gaia_xy = np.column_stack(gaia_coords)

            # Detected star positions
            cat_xy = star_positions[:, :2]
            gaia_coords = init_wcs.world_to_pixel(
            SkyCoord(
            ra=gaia_catalog["ra"],
            dec=gaia_catalog["dec"]
            )
            )

            gaia_xy = np.column_stack(gaia_coords)
            detected_xy = star_positions[:, :2]

            matrix, matches, score = optimize_affine_matching(
            gaia_xy,
            detected_xy,
            radius=1.0,
            rotation_range=5,
            translation_range=20,
            scale_range=0.02
            )

            model, matches_refined = refine_affine_ransac(
            gaia_xy,
            detected_xy,
            matrix,
            radius=1.0,
            residual_threshold=1.0
            )

            print("Transformation raffinée :")
            print(model.params)

            print("Correspondances initiales :", len(matches))
            print("Correspondances raffinées :", len(matches_refined))

            #model_robust, inliers = ransac((gaia_xy[-500:,],cat_xy[-500:,]), AffineTransform,  min_samples=3, residual_threshold=2.0, max_trials=1000)
            matrix = np.linalg.pinv(model.params)

        newcs,match,match_tot = refine_wcs(gaia_catalog,star_positions[:,:2][-2000:]-0.5,init_wcs,transform=matrix,factor=factor,threshold=1,iteration=5)
        indexes = np.arange(0,len(star_positions)).astype(int)[-2000:]
        match[:,1] = indexes[match[:,1]]

        return newcs,shift,match,star_positions
        
def refine_wcs(gaia,img,wcs_img,transform=None,threshold = 1,factor=1,iteration = 5):
    
    wc = wcs_img
    img2 = img.copy()
    if transform is not None:

        img2 = img2 @ transform[:2, :2].T + transform[:2, 2]
        
    for i in range(iteration):
    
        
        try:
            pixels = wc.world_to_pixel(SkyCoord(gaia['ra']*u.deg,gaia['dec']*u.deg))
        except:
            pixels = wc.world_to_pixel(SkyCoord(gaia['ra'],gaia['dec']))
        pixels = np.array(pixels).T*factor


        tree = cKDTree(img2)  # detected stars
        dist, idx = tree.query(pixels, distance_upper_bound=threshold*factor)

        valid = idx < len(img2)  # only matched stars
        matches = np.column_stack([np.arange(len(pixels))[valid], idx[valid],dist[valid]])
        matche = np.column_stack([np.arange(len(pixels))[valid], idx[valid]])
        try:
            try:
            
                new_wcs = fit_wcs_from_points(img2[matche[:,1],:2].T,SkyCoord(gaia[matche[:,0]]['ra']*u.deg,gaia[matche[:,0]]['dec']*u.deg))
                
            except:
            
                new_wcs = fit_wcs_from_points(img2[matche[:,1],:2].T,SkyCoord(gaia[matche[:,0]]['ra'],gaia[matche[:,0]]['dec']))
        except:
            breakpoint()
        print(i,len(matche),np.median(matches[:,2]))
        #
       
        wc = new_wcs
        factor = 1
        

    try:
        finalwcs = fit_wcs_from_points(img[matche[:,1],:2].T,SkyCoord(gaia[matche[:,0]]['ra']*u.deg,gaia[matche[:,0]]['dec']*u.deg))
    except:
        finalwcs = fit_wcs_from_points(img[matche[:,1],:2].T,SkyCoord(gaia[matche[:,0]]['ra'],gaia[matche[:,0]]['dec']))

    return finalwcs,matche,matches
  
  
def build_image_model(wcs, catalog, shape=(1000, 1000), psf_sigma=1.0, factor=1):
    
    ny, nx = shape
    image = np.zeros((ny, nx), dtype=np.float64)

    # Vectorized RA/Dec -> pixel coordinates
    coords = SkyCoord(
        ra=np.asarray(catalog["ra"]) * u.deg,
        dec=np.asarray(catalog["dec"]) * u.deg
    )

    x, y = wcs.world_to_pixel(coords)
    x *= factor
    y *= factor

    # Flux, default = 1
    try:
        flux = np.asarray(catalog["flux"], dtype=np.float64)
    except:
        flux = np.ones(len(catalog['ra']), dtype=np.float64)*100

    # Keep only sources whose PSF can contribute to the image
    r = 3.0 * psf_sigma

    valid = (
        (x >= -r) & (x < nx + r) &
        (y >= -r) & (y < ny + r)
    )

    x = x[valid]
    y = y[valid]
    flux = flux[valid]

    # Precompute coordinate offsets for the largest possible stamp
    radius = int(np.ceil(r))
    dx = np.arange(-radius, radius + 1)
    dy = np.arange(-radius, radius + 1)

    DX, DY = np.meshgrid(dx, dy)

    norm = 1.0 / (2.0 * psf_sigma**2)

    for xc, yc, f in zip(x, y, flux):
        ix = int(xc)
        iy = int(yc)

        # Local coordinates
        xx = ix + DX
        yy = iy + DY

        # Clip to image
        valid_x = (xx >= 0) & (xx < nx)
        valid_y = (yy >= 0) & (yy < ny)
        valid = valid_x & valid_y

        # Gaussian centered at the fractional pixel position
        gaussian = f * np.exp(
            -((xx - xc)**2 + (yy - yc)**2) * norm
        )

        image[yy[valid], xx[valid]] += gaussian[valid]

    return image
    
def get_matrice_transformation(cat1,cat2,match1,match2,factor=1,center1=0,center2=0):

    matches = []

    for i, a in enumerate(match1[:,0]):
        for j, b in enumerate(match2[:,0]):
            if a == b:
                matches.append((match1[i,1],match2[j,1]))

    matches = np.array(matches).astype(int)

    model_robust, inliers = ransac( (factor*(cat1[matches[:,0],:2]-center1),factor*(cat2[matches[:,1],:2]-center2)), AffineTransform, min_samples=3, residual_threshold=0.5, max_trials=100)
    
    matrice = model_robust.params[:2]
    return matrice
    
def get_wcs_matrice_transformation(wcs1,wcs2,gaia_catalog,oversample=1):

 
    # --- Reference WCS ---
    wcs_new1 = wcs1.deepcopy()

    if wcs_new1.wcs.has_cd():
        wcs_new1.wcs.cd /= oversample
    else:
        wcs_new1.wcs.cdelt /= oversample

    wcs_new1.wcs.crpix *= oversample

    # --- Current WCS ---
    wcs_new2 = wcs2.deepcopy()

    if wcs_new2.wcs.has_cd():
        wcs_new2.wcs.cd /= oversample
    else:
        wcs_new2.wcs.cdelt /= oversample

    wcs_new2.wcs.crpix *= oversample
    
    try:
        ref = wcs_new1.world_to_pixel(SkyCoord(gaia_catalog['ra']*u.deg,gaia_catalog['dec']*u.deg))
        tar = wcs_new2.world_to_pixel(SkyCoord(gaia_catalog['ra']*u.deg,gaia_catalog['dec']*u.deg))
        mat = np.linalg.lstsq(np.c_[ref[0],ref[1],[1]*len(ref[0])],np.c_[tar[0],tar[1]])
        
    except:
        ref = wcs_new1.world_to_pixel(SkyCoord(gaia_catalog['ra'],gaia_catalog['dec']))
        tar = wcs_new2.world_to_pixel(SkyCoord(gaia_catalog['ra'],gaia_catalog['dec']))
        mat = np.linalg.lstsq(np.c_[ref[0],ref[1],[1]*len(ref[0])],np.c_[tar[0],tar[1]])
        
    return mat[0].T
    
    
def plot_isocontours(image,contours=20):

    
    YY,XX = np.indices(image.shape)
    plt.imshow(np.log10(image),origin='lower',cmap='Greys')
    plt.contour(XX,YY,np.log10(image),cmap='Reds',levels=contours)
    
    plt.show()

def find_isolated_stars(stars, threshold):
    # Compute pairwise distances
    dists = distance.cdist(stars[:, :2], stars[:, :2])
    
    # Ignore self-distance by setting diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    
    # A star is isolated if ALL distances are greater than threshold
    isolated_mask = np.all(dists > threshold, axis=1)
    
    return  np.where(isolated_mask)[0]
    

def inject_stars(image, psf, n_stars, mag_range, zeropoint,
                 x_range=None, y_range=None, seed=None):
    """
    Inject synthetic stars into an image.

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    psf : 2D numpy array
        PSF kernel (should be normalized to sum=1).
    n_stars : int
        Number of stars to inject.
    mag_range : tuple (mag_min, mag_max)
        Magnitude range for injected stars.
    zeropoint : float
        Photometric zeropoint (mag = ZP - 2.5 log10(flux)).
    x_range, y_range : tuple or None
        Range of positions for injection. Defaults to full image.
    seed : int or None
        Random seed.

    Returns
    -------
    new_image : 2D numpy array
        Image with injected stars.
    catalog : structured array
        Table of injected star properties (x, y, mag, flux).
    """

    rng = np.random.default_rng(seed)
    ny, nx = image.shape

    if x_range is None:
        x_range = (0, nx)
    if y_range is None:
        y_range = (0, ny)

    # Sample magnitudes
    mags = rng.uniform(mag_range[0], mag_range[1], n_stars)

    # Convert magnitudes to flux
    fluxes = 10**((zeropoint - mags) / 2.5)

    # Sample positions
    xs = rng.uniform(x_range[0], x_range[1], n_stars)
    ys = rng.uniform(y_range[0], y_range[1], n_stars)

    # Output image
    new_image = image.copy()

    # PSF center
    py, px = np.array(psf.mean_psf.shape) // 2

    for x, y, flux in zip(xs, ys, fluxes):
        ix, iy = int(x), int(y)

        # Define stamp boundaries
        x0 = ix - px
        x1 = ix + (psf.mean_psf.shape[1] - px)
        y0 = iy - py
        y1 = iy + (psf.mean_psf.shape[0] - py)

        ppssff = psf.evaluate_shift(x-int(x), y-int(y))
        # Skip stars too close to edges
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue

        # Add scaled PSF
        new_image[y0:y1, x0:x1] += flux * ppssff

    # Build catalog
    catalog = np.zeros(n_stars, dtype=[
        ('x', float), ('y', float),
        ('mag', float), ('flux', float)
    ])
    catalog['x'] = xs
    catalog['y'] = ys
    catalog['mag'] = mags
    catalog['flux'] = fluxes

    return new_image, catalog
    
    
def monte_carlo_efficiency(image, psf, zeropoint,
                           psf_size=2, threshold_sigma=0.5,
                           peakmax=10000, min_separation=1,
                           mag_range=[15,30], bins=25,
                           match_radius=0.25,
                           n_stars_per_iterations=100,
                           n_iterations=20,
                           seed=None):

    rng = np.random.default_rng(seed)

    # Prepare bins
    if isinstance(bins, int):
        bins = np.linspace(mag_range[0], mag_range[1], bins)

    mag_centers = 0.5 * (bins[:-1] + bins[1:])

    N_injected_total = np.zeros(len(bins)-1, dtype=int)
    N_detected_total = np.zeros(len(bins)-1, dtype=int)

    all_stars = []
    all_delta_mag = []   # <<< NEW
    try:
        # Initial detection
        star_positions, full_catalog,bkg = detect_stars(
            np.array(image),
            psf_size=psf_size,
            threshold_sigma=threshold_sigma,
            peakmax=peakmax,
            min_separation=min_separation
        )
        all_stars.append(star_positions)
    except:
        pass
    

    # -------------------------
    # Monte Carlo loop
    # -------------------------
    eff_per_iter = []
    
    for i in tqdm(range(n_iterations), desc="Monte Carlo efficiency"):

        # Inject stars
        new_image, catalog = inject_stars(
            image, psf, n_stars_per_iterations, mag_range, zeropoint,
            seed=rng.integers(1e6)
        )

        # Detect stars
        star_positions, full_catalogn,bkg = detect_stars(
            np.array(new_image),
            psf_size=psf_size,
            threshold_sigma=threshold_sigma,
            peakmax=peakmax,
            min_separation=min_separation
        )

        all_stars.append(star_positions)

        # -------------------------
        # KDTree matching
        # -------------------------
        tree = cKDTree(np.c_[catalog['x'], catalog['y']])

        dist, idx = tree.query(
            star_positions[:, :2]-0.5,
            distance_upper_bound=match_radius
        )

        valid = np.isfinite(dist)

        det_idx = np.where(valid)[0]   # detected indices
        inj_idx = idx[valid]           # injected indices
        d = dist[valid]

        matched_det = []
        matched_inj = []

        if len(det_idx) > 0:

            # greedy unique assignment
            order = np.argsort(d)
            used_inj = set()

            for j in order:
                inj_id = inj_idx[j]
                det_id = det_idx[j]

                if inj_id not in used_inj:
                    used_inj.add(inj_id)
                    matched_inj.append(inj_id)
                    matched_det.append(det_id)

        # Build matched catalog safely
        matched_catalog = (
            catalog[matched_inj]
            if len(matched_inj) > 0
            else np.empty(0, dtype=catalog.dtype)
        )

        # -------------------------
        # Δmag computation (NEW)
        # -------------------------
        if len(matched_inj) > 0:
        
            delta_mag = (
                star_positions[matched_det, 3]+zeropoint
                - catalog[matched_inj]['mag']
            )
            all_delta_mag.append(delta_mag)

        # -------------------------
        # Histograms
        # -------------------------
        N_injected, _ = np.histogram(catalog['mag'], bins=bins)
        N_detected, _ = np.histogram(matched_catalog['mag'], bins=bins)

        N_injected_total += N_injected
        N_detected_total += N_detected
        eff_iter = N_detected / N_injected
        eff_per_iter.append(eff_iter)

    
    # -------------------------
    # Efficiency
    # -------------------------
    #efficiency = np.zeros_like(N_injected_total, dtype=float)
    #valid = N_injected_total > 0
    #efficiency[valid] = N_detected_total[valid] / N_injected_total[valid]

    #sigma = np.zeros_like(efficiency)
    #sigma[valid] = np.sqrt(
    #    efficiency[valid] * (1 - efficiency[valid]) / N_injected_total[valid]
    #)

    eff_per_iter = np.array(eff_per_iter)

    eff_mean = np.nanmean(eff_per_iter, axis=0)
    eff_std  = np.nanstd(eff_per_iter, axis=0)

    efficiency = eff_mean
    sigma = eff_std
    return (
        mag_centers,
        efficiency,
        sigma,
        N_injected_total,
        N_detected_total,
        all_stars,
    )
    
def ZP_offset(stars_ra,stars_dec,stars_mag,ref_ra,ref_dec,ref_mag,dist=10**-4):

    tree = cKDTree(np.c_[stars_ra,stars_dec])
    dist, idx = tree.query(np.c_[ref_ra,ref_dec],distance_upper_bound=dist)
    # valid matches
    valid = np.isfinite(dist) & (idx < len(stars_ra))

    # keep only brightest matched stars
    bright = stars_mag[idx[valid]] < np.nanpercentile(stars_mag,25)

    inj_idx = idx[valid][bright]
    det_idx = np.where(valid)[0][bright]

    delta_ZP = np.median(
        stars_mag[inj_idx] - ref_mag[det_idx]
    )

 
    return delta_ZP

