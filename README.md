# README
This repository contains the public versions of projects worked on by this user at Imperial College. This project was developed using Python version 3.7.6.

## AMS (Autonomous Modelling System) version 1
A desktop application to facilitate experiments in the metal forming group at Imperial. 
The following text describes how the original version (In the ams_submitted folder) works.

It does 4 different things:

1. Renames the raw data files collected during the experiment from the robotic arm (UR 10 from Universal Robots) to capture the experimental conditions. This allows the program to know the test conditions from the file name. Currently some manual checks are done on these files.
2. Processes the raw data files (position and force data in two separate files from each experiment) and obtains the necessary data to plot graphs of coefficient of friction against sliding distance. This utilises pandas to carry out all the necessary calculations. The processed raw data files are moved to a raw data bin folder so that the user knows which files have been processed. A new file is containing the processed data  iscreated in a processed data folder. All the relevant test conditions will be extracted from the file name and stored in the database. Each experiment will have a unique experimental id.
3. The user will have done repeats of experiments with the same test condition. The user decide which test conditions should be averaged and the program will extract the relevant files. A graph showing the averaged results (coefficient of friction against sliding distance) as well as the standard deviations envelops will be shown. As there may be anomalous data, there is a 'select' column in the database to indicate whether each set of data should be used in the process. The default value is 1, with 0 indicating it should not be used. A new file containing this data will be created and stored in an averaged data folder.
4. Once all the necessary data has been collected and averaged, the user will proceed to fit the interactive friction model to the data. Before this step, the user will have to input the initial guesses for the model constants and all the values for the relevant lubricant properties in the database. The user will select the test conditions to be used and the program will automatically extract and plot the model predictions along with the data in a window. The 6 different model parameters will be changed by user through sliders until a suitable fit has been obtained. This can then be saved to the database by clicking the save button. The fit then can then be automatically refined through the use of an inbuilt optimisation algorithm.

The application relies on the following folder structure:
```bash
application_folder
├───application.exe
├───friction_averaged_results
├───friction_processed_results
├───friction_raw_data
└───friction_raw_data_bin
```

As of the time of this writing the program is not compiled into a single .exe and is made up of multiple .py scripts. 
To run the program, the user should run 'python app.py' in the terminal. This launching a GUI version of the application.

## AMS (Autonomous Modelling System) version 2
The project was improved on in the following summer. The requirements had changed and the currently uploaded version is a work in progress.

The graphical user interface was overhauled and multiple quality of life improvements were made to the process with the removal of the need to label the input files being one of them. Instead, the user can label the test conditions of each experimental result through the graphical user interface provided.
It was made easier to cycle through the 
Averaging the data was 

To run this version of the program, the user should run 'python app.py' in the terminal. This launching a GUI version of the application.

## Model fitter
A desktop application to facilitate experiments in the metal forming group at Imperial. 
The following text describes how the model fitter (In the model_fitter folder) works.

To run this program, the use should run 'python fitting.py' in the terminal
