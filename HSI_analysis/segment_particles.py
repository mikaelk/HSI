# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:53:14 2020

@author: Specim
"""
import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import pandas as pd
import re
from argparse import ArgumentParser


def decision_tree(minima_dict, characterization, mean_bandwith):
    guess_dict = {}
    for roi in minima_dict.keys():
        guess_dict[roi] = {}
        for target in characterization.keys():
            guess_dict[roi][target] = 0.0
    # check each roi
    for roi, peaks in minima_dict.items():
        # check each minimum in this area
        for peak in peaks:
            # check charactaeristic bands
            for target, criteria in characterization.items():
                for value in criteria['hard']:
                    if (value - mean_bandwith*1.5) <= peak <= (value + mean_bandwith*1.5):
                        guess_dict[roi][target] = guess_dict[roi][target] + 1
                for value in criteria['soft']:
                    if (value - mean_bandwith*1.5) <= peak <= (value + mean_bandwith*1.5):
                        guess_dict[roi][target] = guess_dict[roi][target] + 0.01
    return guess_dict


def guess_wrapup(guess_dict):
    final_guess =  {}
    for roi, guesses in guess_dict.items():
        final_guess[roi] = ""
        highest = max(guesses.values())
        for compound, guess_count in guesses.items():
            if guess_count == highest:
                final_guess[roi] += f"{compound}, "
    return final_guess
        

def read_meta_file(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()
        settings_ = lines[1]
        settings_sobol_ = lines[2]
    
        settings_ = eval('{' + settings_.split('{')[1].split('}')[0] + '}')
        settings_sobol_ = eval('{' + settings_sobol_.split('{')[1].split('}')[0] + '}')

    return settings_,settings_sobol_


def read_merge_file(merge_file,str_lookup):
    with open(merge_file, 'r') as f:
        lines = f.readlines()
    
        for i1,line_ in enumerate(lines):
            if str_lookup in line_:
                print(line_)
                i_lookup = i1
                break
        i_delete_str = lines[i_lookup + 2] 
        i_merge_str = lines[i_lookup + 4]
    
        i_delete = [int(str_) for str_ in re.findall('\d+', i_delete_str)]
        i_merge = []
        for line_ in i_merge_str.split(','):
            i_merge_add = [int(str_) for str_ in re.findall('\d+', line_)]
            i_merge.append(i_merge_add)
    return i_delete,i_merge


def plot_example_spectra(files,indices_plot,materials,pixel_squeezed,ax):
    for file_,index_,material_ in zip(files,indices_plot,materials):
        data_ = pd.read_csv(file_)
        bands = np.array(data_.keys()[1:],dtype=float)
        reflectance = data_.iloc[index_,1:]
        reflectance_scaled = reflectance * (pixel_squeezed.mean()/reflectance.mean())
        
        ax.plot(bands, reflectance_scaled, '-', alpha=.4, label=material_)


def calculate_sobol_watershed(corrected_nparr,bands,mean_bandwidth,settings,
                              SAVE=True,POSTPROCESS=False):
    from skimage.filters import sobel
    from skimage.measure import label
    from skimage.segmentation import watershed#, expand_labels
    from skimage.color import label2rgb
    from skimage.measure import regionprops
    
    prefix=''
    if POSTPROCESS:
        print('Running sobol/watershed in postprocessing mode...')
        i_delete,i_merge=read_merge_file(merge_file,str_merge)
        prefix = 'post_'
    
    y_start = settings_sobol['y_start'] #cut the image if scan domain is too large
    y_end = settings_sobol['y_end']
    pixel_to_mm_estim = settings_sobol['pixel_to_mm_estim']
    pixel_area_estim = pixel_to_mm_estim**2
    min_particle_area = settings_sobol['min_particle_area']
    use_petri_mask = settings_sobol['use_petri_mask']
    petri_center = settings_sobol['petri_center']
    petri_radius = settings_sobol['petri_radius']
    threshold_background = settings_sobol['threshold_background']
    PLOT = settings_sobol['plot']
    
    cropped_scan = corrected_nparr[y_start:y_end,:,:]
    orig_scan = cropped_scan.mean(axis=2)
    if PLOT:
        plt.figure()
        plt.hist(orig_scan.flatten(),200)
        plt.xlabel('pixel intensity')
        plt.ylabel('frequency')
    
    mask_sobel = None
    X,Y = np.meshgrid(np.arange(orig_scan.shape[1]),
                      np.arange(orig_scan.shape[0]))
    if use_petri_mask: #circular mask corresponding to the petri dish centre and radius
        mask_sobel = np.sqrt((Y-petri_center[0])**2 + (X-petri_center[1])**2) < petri_radius   
    else:
        mask_sobel = np.ones(X.shape,dtype=bool)
    
    edges = sobel(orig_scan,mask=mask_sobel)
    
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(orig_scan)

    foreground, background = 1, 2
    markers[orig_scan < threshold_background] = background
    markers[orig_scan > threshold_background] = foreground
    
    ws = watershed(edges, markers,mask=mask_sobel)
    seg1 = label(ws == foreground)

    
    array_L = [] #particle length
    array_l = [] #particle width
    minima_dict = {}
    
    df_spectrum=pd.DataFrame(columns=bands) #save all spectra
    df_particle=pd.DataFrame(columns=['particle ID','material_peak','L','l','area','filled_area',
                                      'major_axis','minor_axis','equivalent_diameter','perimeter'])
    
    pixel_to_mm = 0
    particle_id_calibrate = None
    if type(settings_sobol['calibration_ID_pix']) == int:
        pixel_to_mm = settings_sobol['calibration_mm']/settings_sobol['calibration_ID_pix']
        print('calibration using given pixel/mm: pixel_to_mm=%f' %(pixel_to_mm) )
    elif type(settings_sobol['calibration_ID_pix']) == str:
        particle_id_calibrate = int(settings_sobol['calibration_ID_pix'])
        
        
    peak_prominence = settings['peak_prominence']
    smooth_window = settings['smooth_window']
    
    seg2 = seg1.copy()
    if POSTPROCESS:
        #create two lists: one with all particles, one to which segment it belongs
        #this way the segments can be merged in the postprocessing step
        list_valid_particle = [] #counter
        list_valid_segment = []
        counter_valid = 0
        for i1 in np.unique(seg1)[1:]:
            valid_particle = False
            copy_seg = seg1.copy()
            copy_seg[(copy_seg !=0) & (copy_seg != i1)] = 0
            particle_area_estim = len(copy_seg[copy_seg == i1])*pixel_area_estim
            if particle_area_estim > min_particle_area:
                list_valid_particle.append(counter_valid)        
                list_valid_segment.append(i1)
                counter_valid += 1
        
        for i1 in np.unique(seg2)[1:]:
            

            if i1 in list_valid_segment:
                i_valid = np.where(np.array(list_valid_segment) == i1)[0][0]
                i_particle = list_valid_particle[i_valid]
                
                #check if it needs to be deleted
                if i_particle in i_delete:
                    seg2[seg2 == i1] = 0
                
                #check if it needs to be merged:
                for list_merge in i_merge:
                    if i_particle in list_merge:
                        leading_number = list_merge[0]
                        i_leading_number = np.where(np.array(list_valid_particle) == leading_number)[0][0]
                        leading_segment = list_valid_segment[i_leading_number]
                        
                        seg2[seg2 == i1] = leading_segment
                        
            else:
                #invalid particles: set to 0
                seg2[seg2 == i1] = 0
            #postprocessing: delete non-particles
        
        
    
    c_valid_particle = 0 #counter
    for i1 in np.unique(seg2)[1:]:
        valid_particle = False
        copy_seg = seg2.copy()
        copy_seg[(copy_seg !=0) & (copy_seg != i1)] = 0
        mask_particle = (copy_seg == i1)
        particle_area_estim = len(copy_seg[copy_seg == i1])*pixel_area_estim
        if particle_area_estim > min_particle_area:
            valid_particle = True
        
        if valid_particle:
        
            #calculate vectors of maximum/minimum variation to estimate dimensions
            x_ = X[copy_seg == i1]
            y_ = Y[copy_seg == i1]
            i_rnd = np.random.randint(0,len(x_)-1,min(1000,len(x_)) )
            svd_ = np.linalg.svd(np.array([x_[i_rnd]-x_[i_rnd].mean(),y_[i_rnd]-y_[i_rnd].mean()]))
          
            scaling = 100
            cm = np.array([x_.mean(),y_.mean()]).T  
            vector_L = np.array([ cm + svd_[0][:,0]*scaling, cm - svd_[0][:,0]*scaling ])
            
            # rotmat = np.dot(svd_[0],np.array([[0,1],[1,0]]))
            rotated = svd_[0].dot(np.array([x_,y_]))
            L_pixels = rotated[0,:].max()-rotated[0,:].min() #estimated particle length
            l_pixels = rotated[1,:].max()-rotated[1,:].min() #estimated particle width
            
            if c_valid_particle == particle_id_calibrate:
                pixel_to_mm = settings_sobol['calibration_mm'] / L_pixels
                print('calibration using particle %i: pixel_to_mm=%f' %(c_valid_particle,pixel_to_mm))
                
            props = regionprops(copy_seg) #particle properties from scikit-image

            array_L.append( L_pixels )
            array_l.append( l_pixels )
    
            particle_spectrum = cropped_scan[mask_particle]
            pixel_squeezed = particle_spectrum.mean(axis=0)
            smoothed = signal.savgol_filter(pixel_squeezed, window_length=smooth_window, polyorder=2)
    
            df_spectrum.loc[c_valid_particle] = smoothed
    
            tmp_min_dict = {}
            tmp_min_dict[str(c_valid_particle)] = []
            
            fig,axes = plt.subplots(1,2)
            axes[0].plot(bands, pixel_squeezed,'k-', label='Particle')
            minimas,_ = signal.find_peaks(-smoothed, prominence=(peak_prominence, None))
            minima_dict[str(c_valid_particle)] = []
            for minimum in minimas:
                minima_dict[str(c_valid_particle)].append(bands[minimum])
                tmp_min_dict[str(c_valid_particle)].append(bands[minimum])
                axes[0].plot(bands[minimum], pixel_squeezed[minimum],'rx')
            
            material_list = guess_wrapup(decision_tree(tmp_min_dict, characterization, mean_bandwidth))[str(c_valid_particle)]

            particle_info = [c_valid_particle,material_list,L_pixels*pixel_to_mm,l_pixels*pixel_to_mm,props[0]['area']*pixel_to_mm**2,
                             props[0]['filled_area']*pixel_to_mm**2,props[0]['major_axis_length']*pixel_to_mm, props[0]['minor_axis_length']*pixel_to_mm,
                             props[0]['equivalent_diameter']*pixel_to_mm, props[0]['perimeter']*pixel_to_mm ] 
            df_particle.loc[c_valid_particle] = particle_info


            axes[0].set_title('Guess: %s' % material_list)
            axes[0].set_xlabel('Wavelength')
            axes[0].set_ylabel('Reflectance')
            
            spectra_files = [os.path.join(home_folder,spectra_folder,'06_petri_dark_LDPE_2020-12-08.spectrum'),
                             os.path.join(home_folder,spectra_folder,'06_petri_dark_PP_2020-12-08.spectrum'),
                             os.path.join(home_folder,spectra_folder,'06_petri_dark_PS_2020-12-06.spectrum')]
            plot_example_spectra(spectra_files,[1,3,4],['PE','PP','PS'],pixel_squeezed,axes[0])
            axes[0].legend()

            color1 = label2rgb(copy_seg, image=orig_scan, bg_label=0)
            axes[1].imshow(color1)
            axes[1].plot(cm[0],cm[1],'ro',markersize=2)
            axes[1].plot([vector_L[0,0],vector_L[1,0]],[vector_L[0,1],vector_L[1,1]],'r-')
            fig.suptitle('Particle %i' % c_valid_particle)
            
            fig2,ax2 = plt.subplots(1,figsize=(5,12))
            ax2.imshow(color1)
            ax2.plot(cm[0],cm[1],'ro',markersize=2)
            ax2.plot([vector_L[0,0],vector_L[1,0]],[vector_L[0,1],vector_L[1,1]],'r-')
            fig.suptitle('Segment %i' % c_valid_particle)
            
            if SAVE:
                fig.savefig(f"{output_dir}/{prefix}Summary_{c_valid_particle}.png", dpi=300, bbox_inches='tight')
                fig2.savefig(f"{output_dir}/{prefix}Segment_{c_valid_particle}.png", dpi=300, bbox_inches='tight')
        
            plt.close('all')
            c_valid_particle += 1
            
    array_L = np.array(array_L)*pixel_to_mm
    array_l = np.array(array_l)*pixel_to_mm

    if SAVE:
        spectrum_file = os.path.join(home_folder,spectra_folder,prefix+data_name+'.spectrum')
        df_spectrum.to_csv(spectrum_file)

        particle_file = os.path.join(home_folder,spectra_folder,prefix+data_name+'.particle_info')
        df_particle.to_csv(particle_file)
        
        meta_file = os.path.join(home_folder,spectra_folder,data_name+'.meta')
        with open(meta_file, 'w') as f:
            f.write('pixel to mm: %f, minimum area threshold: %f \n settings: %s \n settings sobol filter: %s' % (pixel_to_mm,min_particle_area,settings,settings_sobol))
        
    # Show the segmentations.
    if PLOT:
        fig, axes = plt.subplots(1)#nrows=1, ncols=2, figsize=(9, 5),sharex=True, sharey=True)
        
        color1 = label2rgb(seg2, image=orig_scan, bg_label=0)
        axes.imshow(color1)
        axes.set_title('Sobel+Watershed')
        plt.savefig(f"{output_dir}/{prefix}segmentation.png", dpi=300, bbox_inches='tight')
        plt.close('all')
    return minima_dict,array_L,array_l
        
        
    

def run_everything(coordinate_method, data_path, data_name, output_dir, 
                   characterization, settings, SAVE=True, POSTPROCESS=False):

    # initialize variables
    minima_dict = {}
    
    # setup filenames and path
    dark_raw = f"{data_path}/capture/DARKREF_{data_name}.raw"
    dark_hdr = f"{data_path}/capture/DARKREF_{data_name}.hdr"
    data_raw = f"{data_path}/capture/{data_name}.raw"
    data_hdr = f"{data_path}/capture/{data_name}.hdr"
    white_raw = f"{data_path}/capture/WHITEREF_{data_name}.raw"
    white_hdr = f"{data_path}/capture/WHITEREF_{data_name}.hdr"
    
    # create output directory according to 'output_dir'
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
    
    # open data with spectral.envi
    dark_ref = envi.open(dark_hdr, dark_raw)
    white_ref = envi.open(white_hdr, white_raw)
    data_ref = envi.open(data_hdr, data_raw)
    
    bands = data_ref.bands.centers
    mean_bandwidth = np.mean(data_ref.bands.bandwidths)
    
    # convert to numpy array
    white_nparr = np.array(white_ref.load(dtype=np.float16))
    dark_nparr = np.array(dark_ref.load(dtype=np.float16))
    data_nparr = np.array(data_ref.load(dtype=np.float16))

    # calculate mean of white and dark reference
    # axis 0, to collapse all row into one,
    # so only columns(pixels) and bands (spectralinfo) remain
    white_mean_0 = np.mean(white_nparr, axis=0,dtype=np.float32)
    dark_mean_0 = np.mean(dark_nparr, axis=0,dtype=np.float32)
    
    # I changed the calcultions to be done in-place, 
    # otherwise there is an out-of-memory error for larger images
    diff_white_dark = white_mean_0 - dark_mean_0
    for i1 in range(data_nparr.shape[0]):
        data_nparr[i1,:,:] -= dark_mean_0
        data_nparr[i1,:,:] /= diff_white_dark
    
    data_nparr = data_nparr.astype(np.float32)
    corrected_nparr = data_nparr # corrected_nparr.astype(np.float32)
    
    minima_dict,array_L,array_l = calculate_sobol_watershed(corrected_nparr,bands,
                                                            mean_bandwidth,settings,
                                                            SAVE=SAVE,POSTPROCESS=POSTPROCESS)
        
    return corrected_nparr



if __name__=="__main__":
    import configuration as cmp

    p = ArgumentParser(description="""Particle detection and segmentation""")
    p.add_argument('-name', '--name', default='01_NEMICAT_27_1',help='folder name')
    p.add_argument('-merge_file', '--merge_file', default=None, help='file with merging information')
    p.add_argument('-merge_str', '--merge_str', default=None, help='string for merging file information')
    
    args = p.parse_args()
    data_name = args.name
    print('analysing %s' % data_name)
    
    home_folder = cmp.home_folder
    spectra_folder = cmp.spectra_folder
    
    filename_merge = args.merge_file
    str_merge = args.merge_str
    
    SAVE=True    
    POSTPROCESS = False
    if filename_merge:
        POSTPROCESS = True
        print('postprocessing, file: %s, str: %s' % (filename_merge,str_merge) )
        
    coordinate_method = 'sobol' #'sobol' #roi, blob, sobol
    
    data_path = os.path.join(home_folder,data_name)
    output_dir = os.path.join(home_folder,data_name,'output')
    
    characterization = cmp.characterization
    
    if POSTPROCESS: #mode which can be used to delete non-particles and merge particles
        merge_file = os.path.join(home_folder,spectra_folder, filename_merge )
        meta_file = os.path.join(home_folder,spectra_folder,data_name+'.meta')
        if not os.path.exists(meta_file):
            raise RuntimeError('postprocessing requires a metadata file (%s), please run in normal mode first' % meta_file)
        settings,settings_sobol = read_meta_file(meta_file)
    else:
        merge_file = None
        
        settings = cmp.defaults            
        settings_sobol = cmp.settings_sobol
    

    corrected_nparr = run_everything(coordinate_method, data_path, data_name, 
                        output_dir, cmp.characterization, cmp.defaults, SAVE=SAVE,
                        POSTPROCESS=POSTPROCESS)
    
    plt.figure()
    plt.imshow(corrected_nparr[:,:,130])

