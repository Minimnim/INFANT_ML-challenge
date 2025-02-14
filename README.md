Instructions for Using the Codes
This guide provides step-by-step instructions for successfully running the codes. The project consists of two main steps: preprocessing the data and generating features using MATLAB, and building and evaluating a model using Python in a Jupyter Notebook.

Step 1: Preprocessing and Feature Generation (MATLAB)
1. Open the MATLAB file provided (script.m).
2. Run the script in MATLAB. This step involves:
- Preprocessing the raw data.
	- ATTENTION: Please update the paths to the locations eeg.csv, codes, and eeg_grades.csv are saved.
	- All the codes should be saved in the same working directory.
- Generating relevant features from the preprocessed data.
- Combining the extracted features with corresponding EEG grades
	- Please update n regarding the number of eeg.csv files you are analysing.
- Separating the dataset into training and validation sets.
	- This part of code is written considering the structure of eeg_grades.csv file, file_ID in the first column, baby_ID in the second, and gardes in the fourth 	column. ATTENTION: Please update it if you are using a file with different structure.
3. Once the MATLAB script completes, necessary data files and feature files will be generated for further processing in Python.

This step was the same for all the submitted models. So it needs to be done once, and then different models can be tested on the pre-processed data. 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Step 2: Building and Evaluating the Model (Python)
1. Open the Jupyter Notebook file named script2.ipynb.
2. Follow the instructions within the notebook. This step includes:
- Loading required libraries
- Loading the preprocessed data and features generated by the MATLAB script.
	- ATTENTION: Please update the path to the location the output of matlab code, specifically 'training_bycode.csv' ('training_bycode_important_features.csv' for submitted models 11-17) has been saved.
- Building a predictive model using leave-one-out XGBoost on training dataset.
- Making predictions on validation dataset using the trained model.
	- ATTENTION: Please update the path to the location the output of matlab code, specifically 'validation_bycode.csv' ('validation_bycode_important_features.csv' for submitted models 11-17), has been saved.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Note
- Step 1 can be done once, since it is the same process for all the submited models. Step 2 is changing, regarding the hyperparameters, for each model. The model provided in Jupyter Notebook is an example of one of the models. The other are submitted as Colab files. The instruction for running them is the same as provided Jupyter Notebook.
- Please feel free to reach out for any questions or clarifications during the process.
