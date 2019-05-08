to train a model

    python model2.py

to test a model's performance in active learning

    python experiment1.py

## Running Things On Cloud

There are 3 major players in the computation process of running on google

* Local Macine : Your local machine, this is your main small-scale testing and code editor / debugger
* Remote Machine : Google, where the heavy computation needs to happen
* GitHub : Communication between the two, where also experiment results will permanently be held

# Overall Workflow

First, make sure the code can be run over a small-scale instance locally, i.e. only 1% of the total data

Second, turn the full-scale running process into a python script (of diff
arguments in case of multiple cases), experiment.py and run\_experiment.py (holds arguments only)

Add the python script to the git repo

Use launch.py to boot up remove machine, which will do the following :

* pull github to get the python script
* run the full-scale experiment specified by python script, with specific arguments in runexperiment.py
* result stored as a special pickle specified by python script, the pickle contains meta-data on how its ran
* add and push the result to github
* remote machine commit suicide to save us money

# The Local Machine
two files of running the script in addition to all the code files.

* launch.py the main script to generate code for booting up remote instance and logistics of running killing
* experiment.py the main script that takes multiple arguments, pushed once never changed
* run\_experiment.py the main script responsible for the experiment configs (pushed multiple times depends on diff settings)

all results hold meta-data of the commands used to run them so there is no confusion of where they come from

# Remote Machine
* Should be an image that holds the "essential" big data that cannot be stored on github
* Other than that everything should be temporary and assumed to destroyed after running

# GitHub
Communication protocol of persistent information across the computes, i.e. epxeriment.py and the result of the experiments
