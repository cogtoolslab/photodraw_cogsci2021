This directory contains the code necessary to run the `recogdraw_category` experiment, which asks participants to match a sketch to its category in a 32-afc setup. We obtained ratings for 12,288 sketches.

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/recogdraw_category/stimuli/recogdraw_category_exampletrialpng.png"></img>
</p>

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the kiddraw directory
- run `node store.js` in the `recogdraw_instance` directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
