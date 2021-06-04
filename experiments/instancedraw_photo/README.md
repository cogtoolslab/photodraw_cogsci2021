This directory contains the code necessary to run the instance-goal, photo-cued factor of the `photodraw2x2` experiment, in which participants 96 were asked to sketch the specific object they were shown. The actual stimuli in `categorydraw_photo` and `instancedraw_photo` are the same, but the instructions are different.

<p align="center" style="font-size: smaller">
  <img width="85%" src="https://github.com/cogtoolslab/photodraw_cogsci2021/blob/master/experiments/instancedraw_photo/stimuli/instance_photo_screencap.gif"></img>
</p>

### How to run the experiment
- SSH into user@cogtoolslab.org 
- run `node app.js --gameport XXXX` in the kiddraw directory
- run `node store.js` in the `instancedraw_photo` directory
- navigate to https://cogtoolslab.org:XXXX/index.html to demo the experiment

Note: you need to run `npm install` to get `node-modules` in the experiment directory.
