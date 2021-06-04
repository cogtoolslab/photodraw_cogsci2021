This directory contains the code necessary to run the instance-goal, text-cued factor of the `photodraw2x2` experiment, in which 96 participants were asked to imagine and draw a specific exemplar of the cued category.

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/instancedraw_text/stimuli/suprised_pikachu_face.gif"></img>
</p>

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the kiddraw directory
- run `node store.js` in the `instancedraw_text` directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
