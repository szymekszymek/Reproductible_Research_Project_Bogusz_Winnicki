In order to run the original script, a few modifications had to be made
	1. Older version of 'xlrd' package needed to be instaalled (the newest one does not support reading xlsx files)
		- 'pip2 install xlrd==1.2.0'
	2. Line 120 - from 'jobsData = pd.read_csv(jobDataFile)' to 'jobsData = pd.read_excel(jobDataFile).rename(columns={"Title": "Title_jobsData"})'
		- 'read_csv' needed to be changed to 'read_excel' since the file was provided in the excel format
		- '.rename(columns={"Title": "Title_jobsData"})' needed to be added at the end of the line since this df is later merged with another df containing the 'Title' column that resulted in two columns 'Title_x' and 'Title_y', thus preventing the join in line 154 ('allscores = pd.merge(jobScores,wvJTS,on=['O*NET-SOC Code','Title'])')
	3. Line 134 & 148 - column selection needed to be removed, thus selecting all available columns
		- in both cases there was a selection of 5 columns (['O*NET-SOC Code','Job Description','Projected Growth (2014-2024)','Projected Job Openings (2014-2024)','Industries']) from the 'Task Statements.xlsx' file, but 4 of those columns were not presented in the provided file.
		
In order to run the script, we created a directory called 'cf_dir' with all data files and modified python script, navigated into this directory (e.g. "cd '/Users/Szymon/Desktop/UNI/Mag/4 semester/Reproductible Research/project'") and run the following command (in our case 'sudo' was required for permission reasons):
"sudo python2 'cfProcessor_AEAPnP.py' '../cf_dir' 'cf_report_DWAS_input.csv' 'cf_report_DWAS_physical_input.csv' 'Tasks to DWAs.xlsx' 'Task Ratings.xlsx' 'Task Statements.xlsx' 'OESfile.xlsx'"