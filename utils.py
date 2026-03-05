import glob, os
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.io import fits
from datetime import datetime
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from lstchain import __version__ as lstchain_version

def IndexToDeclRa(index, nside):
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)

def DeclRaToIndex(decl, ra, nside):
    return hp.pixelfunc.ang2pix(
        nside, np.radians(-decl + 90.),
        np.radians(360. - ra)
    )
    
def healpix2map(healpix_data, ra_bins, dec_bins):
    
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)

    # Convert the latitude and longitude to theta and phi
    theta, phi = np.radians(90 - dec_grid), np.radians(ra_grid)
    
    nside = hp.npix2nside(len(healpix_data)) # nside of the grid

    # Convert theta, phi to HEALPix indices and create a 2D map using the HEALPix data
    hp_indices = hp.ang2pix(nside, theta, phi)

    return (healpix_data[hp_indices])

def get_hp_map_thresholds(healpix_data, threshold_percent=[0.9, 0.68]):
    
    # We sort the tresholds itself in descending order
    threshold_percent = np.sort(threshold_percent)[::-1]
    
    # Sort in descending order and normalize
    sorted_data = np.sort(healpix_data)[::-1] / np.sum(healpix_data)
    cumulative_sum = np.cumsum(sorted_data)

    # Find the values corresponding to the thresholds
    indexes_map = [np.searchsorted(cumulative_sum, t) for t in threshold_percent]
    # Then we find the thresholds
    threshold_maps = [sorted_data[min(index, len(sorted_data) - 1)] for index in indexes_map]
    
    return threshold_maps

def get_2d_map_hotspot(map_data_2d, ra_bins, dec_bins):
    
    # Computing coordinate of maximum probability
    max_prob_index = np.unravel_index(np.argmax(map_data_2d), map_data_2d.shape)
    
    max_prob_ra, max_prob_dec = ra_bins[max_prob_index[1]], dec_bins[max_prob_index[0]]
    max_prob_coords = SkyCoord(ra=max_prob_ra, dec=max_prob_dec, unit=u.deg, frame="icrs")
    return max_prob_coords

def add_pointing_hdu(input_file, output_file, obs_id, row):
    """
    Reads a FITS file, adds a POINTING BinTableHDU at index 3, and saves to a new file.
    
    Parameters:
    - input_file (str): Path to the original FITS file.
    - output_file (str): Path where the new FITS file will be saved.
    - obs_id (int): Observation ID to insert into the header.
    """
    
    # Data arrays
    time_data = np.array([
        datetime.fromisoformat((row.Index + " " + row.Time).replace("\"", "")).timestamp()
    ], dtype=">f8")
    ra_pnt_data = np.array([row.RA], dtype=">f8")
    dec_pnt_data = np.array([row.DEC], dtype=">f8")
    alt_pnt_data = np.array([row.ALT], dtype=">f8")
    az_pnt_data = np.array([row.AZ], dtype=">f8")

    created_time = datetime.now().isoformat().replace("T", " ")
    
    # Create the FITS Columns
    col1 = fits.Column(name="TIME", format="D", unit="s", array=time_data)
    col2 = fits.Column(name="RA_PNT", format="D", unit="deg", array=ra_pnt_data)
    col3 = fits.Column(name="DEC_PNT", format="D", unit="deg", array=dec_pnt_data)
    col4 = fits.Column(name="ALT_PNT", format="D", unit="deg", array=alt_pnt_data)
    col5 = fits.Column(name="AZ_PNT", format="D", unit="deg", array=az_pnt_data)

    cols = fits.ColDefs([col1, col2, col3, col4, col5])

    # Create the Binary Table HDU
    pointing_hdu = fits.BinTableHDU.from_columns(cols, name="POINTING")

    # Add the specific header keywords you requested
    hdr = pointing_hdu.header
    hdr["CREATOR"] = f"lstchain v{lstchain_version}"
    hdr["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    hdr["HDUVERS"] = "0.3"
    hdr["HDUCLASS"] = "GADF"
    hdr["ORIGIN"] = "CTA"
    hdr["TELESCOP"] = "CTA-N"
    hdr["CREATED"] = created_time
    hdr["HDUCLAS1"] = "POINTING"
    hdr["OBS_ID"] = obs_id
    hdr["MJDREFI"] = 58392
    hdr["MJDREFF"] = 0.0
    hdr["TIMEUNIT"] = "s"
    hdr["TIMESYS"] = "UTC"
    hdr["TIMEREF"] = "TOPOCENTER"
    hdr["GEOLON"] = -17.891497010000023
    hdr["GEOLAT"] = 28.761526109999995
    hdr["ALTITUDE"] = f"{row.ALT} deg"
    hdr["MEAN_ZEN"] = f"{row.ZD} deg"
    hdr["MEAN_AZ"] = f"{row.AZ} deg"
    hdr["B_DELTA"] = "98.55072 deg"

    # Open the original FITS file, insert the new HDU, and write to a new file
    with fits.open(input_file) as hdul:
        # Insert at index 3 (Pushing EFFECTIVE AREA to index 4, ENERGY DISPERSION to index 5)
        hdul.insert(3, pointing_hdu)
        
        # Save to the new destination
        hdul.writeto(output_file, overwrite=True)
        print(f"Successfully added POINTING HDU and saved to {output_file}")


def fix_events_extension(file_path, obs_id):
    with fits.open(file_path) as hdul:
        # Find the EVENTS extension regardless of trailing spaces
        events_idx = None
        for i, hdu in enumerate(hdul):
            if hdu.name.strip() == "EVENTS":
                events_idx = i
                break
        
        if events_idx is None:
            print(f"Skipping {file_path}: 'EVENTS' not found (Checked: {[h.name for h in hdul]})")
            return

        target_hdu = hdul[events_idx]
        hdr = target_hdu.header
        data = target_hdu.data

        hdr["OBS_ID"] = obs_id

        # --- Add Headers TIME-OBS and TIME-END ---
        mjd_ref = hdr.get("MJDREFI", 0) + hdr.get("MJDREFF", 0.0)
        ref_time = Time(mjd_ref, format="mjd", scale="utc")

        for key, t_val in [("TIME-OBS", "TSTART"), ("TIME-END", "TSTOP")]:
            t_seconds = hdr.get(t_val)
            if t_seconds is not None:
                hdr[key] = (ref_time + TimeDelta(t_seconds, format="sec")).iso.split(" ")[1]

        hdul.writeto(file_path, overwrite=True)
    
    print(f" -> Fixed 'EVENTS' in {file_path}")


def fix_header_floats(file_path):
    """
    Deletes string-based pointing keywords and re-inserts them as pure floats.
    This prevents the lstchain indexer from crashing.
    """
    with fits.open(file_path, mode='update') as hdul:
        # Loop through both EVENTS and POINTING if they exist
        for ext_name in ['EVENTS', 'POINTING']:
            if ext_name in hdul:
                hdr = hdul[ext_name].header
                
                # List of keywords that MUST be floats for the indexer
                keywords_to_fix = ['ALT_PNT', 'AZ_PNT', 'RA_PNT', 'DEC_PNT']
                
                for key in keywords_to_fix:
                    if key in hdr:
                        raw_val = hdr[key]
                        try:
                            # 1. Convert to float
                            clean_val = float(raw_val)
                            
                            # 2. DELETE the old key entirely to clear the string flag
                            comment = hdr.comments[key]
                            del hdr[key]
                            
                            # 3. RE-INSERT as a fresh float card
                            hdr[key] = (clean_val, comment)
                            
                        except (ValueError, TypeError):
                            print(f"Could not convert {key} in {ext_name}")

        # Flush ensures the changes are written to the physical disk
        hdul.flush()
    print(f" -> Fixed header types in {file_path}")

def fix_geographic_headers(file_path):
    """
    Ensures GEOLON, GEOLAT, and ALTITUDE are in the EVENTS extension.
    Uses standard LST-1 / Roque de los Muchachos coordinates.
    """
    with fits.open(file_path, mode='update') as hdul:
        # Find the EVENTS extension
        events_idx = None
        for i, hdu in enumerate(hdul):
            if hdu.name.strip() == "EVENTS":
                events_idx = i
                break
        
        if events_idx is None:
            print(f"Error: 'EVENTS' not found in {file_path}")
            return

        hdr = hdul[events_idx].header

        # Set the coordinates as pure floats (No strings!)
        # Using the values from your POINTING check but cleaned up
        hdr['GEOLON'] = (-17.891497, "Geographic longitude of telescope (deg)")
        hdr['GEOLAT'] = (28.761526, "Geographic latitude of telescope (deg)")
        hdr['ALTITUDE'] = (2199.88, "Geographic altitude of telescope (m)")

        # Save the changes to the file
        hdul.flush()
    
    print(f" -> Geographic headers added to EVENTS in {file_path}")
    
def add_bkg(data_store, obs_id, dir_dl3, dim_bkg, bkg_type):
    
    fname = glob.glob(os.path.join(dir_dl3, f"bkg_{bkg_type}_{dim_bkg}d_{str(obs_id)}.fits"))[0]
    hdul = fits.open(fname)
    # Adding the acceptance model to the HDU table
    data_store.hdu_table.add_row({
        "OBS_ID" : obs_id,
        "HDU_TYPE" : "bkg",
        "HDU_CLASS" : f"bkg_{dim_bkg}d",
        "FILE_DIR" : ".",
        "FILE_NAME" : os.path.basename(fname),
        "HDU_NAME" : "BACKGROUND",
        "SIZE" : hdul["BACKGROUND"].size,
    })
    return data_store


    