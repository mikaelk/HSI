# HSI
Hyperspectral imaging software for polymer particle detection and classification


Below instructions to run the segmentation and machine learning scripts.
I recommend copying the examples folder to e.g. an 'examples_test' folder:

#---------------------------------------------------------------------------------#
1.) Particle segmentation and saving their spectra

To run the example:
cd HSI_analysis

Edit configuration.py with the correct home_folder (in this case the path to the examples folder)

Download the example data from https://ndownloader.figshare.com/files/28398216?private_link=0b2ddf5d8bf12b2d5aeb
And extract the data to the specified home folder 
You should now have a '01_NEMICAT_27_1' folder in the home folder

In bash, run:
python segment_particles.py --name 01_NEMICAT_27_1

This segments the particles.

Then run postprocessing:
python segment_particles.py --name 01_NEMICAT_27_1 -merge_file 01_merge.txt -merge_str 27_1

Which if necessary merges or deletes erroneous particles
See also the 'microplastics_processing_manual.odt' document in the docs folder.

The 01_merge.txt file can be found in the examples/00_spectra folder. 
This file contains the particles that were identified to be erroneous, or particles which were identified to be 1 particle in reality, but segmented as multiple particles. This can happen when particles are e.g. translucent, and only parts are picked up with the camera. In this file, first the identifier for the folder is given. In the case of the 01_NEMICAT_27_1, the '27_1' is the unique identifier the script looks for in the merge file. In this specific case, the scan was quite successful, no particles needed to be deleted or merged.

The -merge_str is the unique identifier for the folder to be post-processed as explained above. 

#---------------------------------------------------------------------------------#
2.) Labelling particles, and using the machine learning classification algorithm

Run label_particles.py in the command line: 
python label_particles.py -name post_01_NEMICAT_27_1.spectrum

We use the post-processed spectrum file here. A labelling file will be created in the '00_labels' folder, specifying whether the peak/through locations are present in the spectrum


We then train and test the random forest classifier on the labelled files:
python train_and_test_classifier.py

This saves a .joblib file in the ML_folder specified in configuration.py
You can train the model on all labelled files by copying the files in HSI_paper_2021/00_labels to the examples/00_labels folder, or by changing the home folder in configuration.py to HSI_paper_2021.

I put the pre-trained model from the paper in the 00_ML_models, which you can use for now. In this case, you don't have to run the train_and_test_classifier.py file.



Finally, we can use the model to make predictions.
Run:
classify_particles.py -string_classify NEMICAT -ML_model RFC_2021052616.joblib
Where the 'NEMICAT' argument tells the program to predict polymer materials for all spectrum files containing 'NEMICAT'. 
The ML_model argument tells the program which pretrained model to take in the 00_ML_models folder

