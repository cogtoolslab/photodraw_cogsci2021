This directory contains the code necessary to run the `recogdraw_instance` experiment, which asks participants to match a photo-cued sketch to its cued image in a 32-afc setup. Since this task is only applicable to photo-cued sketches, we obtained ratings for 6,144 sketches.

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/recogdraw_instance/stimuli/recogdraw_instance_exampletrial.png"></img>
</p>

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the kiddraw directory
- run `node store.js` in the `recogdraw_instance` directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
