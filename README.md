EDF DAT LEARNING ENVIRONMENT!!!
=======================

this project is really just a place to save some useful test scripts and documents I've found useful.

included are :

- feedback.txt
    + this is a raw data file from the eyelink feedback experiment
- parsedat.py
    + this is a script that can parse any .dat file as long as it follows the scho18 schema. 
    + it was developed entirely to be a test script, just to figure out how things are stored, to check that I can write binary using the correct endian and such. 
    + It also ensures that I am using the correct schema. Turns out the DDL provided by the basement (http://www.neurophys.wisc.edu/comp/docs/schemas/) was not totally right... 
    + but the schema represented within parsedat.py is. because I can read any dat file! muahaha!
- parseeyelink.py
    + this was originally to come up with a way to parse an eyelink edf file. 
    + over time it grew into something that can more or less write a dat file from an edf file. it involves an absolute tonne of manually coding and configuration. 
    + check out the makefeedback_dat funciton from unittests at the bottom to see how that's done. 
    + a core concept here is that of the eyelink maps. which identify what specific messages in the edf file mean with referance to scho18 keywords.
    + we'll need to flesh out what specifically scho18 really really needs. 
- RL_hu_test.dat
    + a working dat file produced from parseeyelink.unittests.makefeedback_dat
- SCHOl18.txt
    + DDL describing SCHO18
- swigert_391.dat
    + raw dat file from swigert. there's no hippa complience issue. because all it has is his name and date of experiment. and his name is not cosidered identifying if he's a monkey.... which he is. 
- swigert_391_parsed.txt
    + a text output from parsedat.py 
    + describes what is actualy inside swigert_391.dat
- directory.yml
    + every dat file starts out with a directory which follws a certain schema. 
    + that schema is laid out here. 
- datfile structure.txt
    + describes how datafiles are organized
