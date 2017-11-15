# deep_detektor
Automated Factual-Claim Detection in Danish Broadcasting

This project is part of the Deep Learning course at Technical University of Denmark, taught by professor Ole Winther. 
The project revolves around an automated "claim"-detection system based on subtitles from the danish TV-show "Debatten". The found claims are used later by the fact-checking TV-program "Detektor". 

Check the [wiki](https://github.com/sfvnDTU/deep_detektor/wiki) for references and other good stuff

Team members: phav, jepno, jehi and sfvn at dtu.dk


### Running system

Rules for ensuring runs of files:  
1. All paths are relative and stored in a file called `project_paths.py`.   
1. The runable files are to be placed in the `run_files`-directory and can be run with the main folder (`deep_detektor`)
    as working directory.  
1. Data is stored in a `data`-directory next to the repository (`../data` from working directory).  
1. The two directories `DeepFactData` and `DRDetektorAutomaticFactChecking` are to be copied directly into 
    the `data`-directory.
1. The final structure of the program-files is thus:
```
./
    deep_detektor/                          <- Repository
    data/
        DeepFactData/                       <- From data-service
        DRDetektorAutomaticFactChecking/    <- From data-service
```
