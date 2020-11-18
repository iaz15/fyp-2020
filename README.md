# README
This repository contains the public versions of projects worked on by this user at Imperial College. This project was developed using Python version 3.7.6.

## AMS (Autonomous Modelling System) version 1 - ams_submitted
A desktop application to facilitate experiments in the metal forming group at Imperial.

The following text describes how the original version of the project works.

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
To run the program, the user should run 'python app.py' in the terminal. This launches a GUI version of the application.

## AMS (Autonomous Modelling System) version 2 - ams_improved
The project was improved on in the following summer. The requirements had changed and the currently uploaded version is a work in progress. As of this moment, the Graphical User Interface (GUI) version (launched by running app.py) does not include usage of the database and only includes the data processing steps. To use the fully featured version, database_interaction.py should be run. This script contains all the steps from the original version of the program with the exception of file renaming and is fully functional. However, a way for the user to interact with it is currently not implemented. A sample dataset will be used when running this the program. Certain parts of the program are currently commented out in "\_\_name\_\_ == "\_\_main\_\_" section, but can be uncommented before running the code to examine its features (For instance, the automatic fitting section is commented out as it can take a while to finish running).

Some of the major changes made in this version of the program are listed below.

1. The graphical user interface was overhauled and multiple quality of life improvements were made to the process with the removal of the need to label the input files being one of them. Instead, the user can label the test conditions of each experimental result through the graphical user interface provided. With this, renaming of files is no longer necessary.
2. The database schema (design) was improved, making it easier to manage and carry out certain tasks. Averaging the data now only requires the user to specify the condition ID instead of all the different variable conditions (No need to specify speed, lubricant volume, etc).
3. The code was tidied up, usage of object oriented programming was introduced, and a number of redundant functions and lines were removed.

## Model Fitter - model_fitter
The following text describes how the model fitter works. This program was created improve on the model fitting capabilities of the AMS and make it more extensible to other models. Given any model (structured in a specific format as a class) and data to fit to, the program is able to generate a GUI to allow the user to manually adjust the model parameters through sliders as in the original version. Automatic model parameter tuning can subsequently be carried out. Instead of using a database, the data and test conditions is loaded from an excel spreadsheet as this was the simplest and preferred user method at the time.

With different model parameter initial values, optimisation methods, and termination criteria, the final set of parameters obtained after running the program can vary. To keep track of the multiple results from different runs, results are saved each time. The program will save a .png showing a visual represntation fitted model to the datasets as well as the model parameter values, summary statistics and other relevant information in a .json file. This also gives the opportunity for a person to review each result manually and select the best result as goodness of fit to multiple datasets can be subjective at times. An example of the program's output can be seen in friction_model_liquid_solid_results_0.json and friction_model_liquid_solid_results_0.png.

To run the model fitting program, the user should run fitting.py. In the "\_\_name\_\_ == "\_\_main\_\_" section there are 3 models to choose from as examples.

1. The fit_gaussian() function fits a model to a randomly generated gaussian dataset. 
2. The fit_liquid() function fits the interactive friction model specific to liquid only lubricants to sample experimental datasets.
3. The fit_liquid_solid() functions fits the interactive friction model specific to mixed liquid and solid lubricants to sample experimental datasets.

In this folder, there was also an attempt to create a flexible way to dynamically generate databases from models was made (in dynamic_database.py). This can be seen in dynamic_database.py.
