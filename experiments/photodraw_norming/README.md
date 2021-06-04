This directory contains the code necessary to run the `photodraw_norming` norming study, which asks participants for subjective typicality ratings for each of the 1024 stimuli. 

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/photodraw_norming/stimuli/example_norming_trial.png"></img>
</p>

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the `photodraw_norming` directory
- run `node store.js` in the kiddraw directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
