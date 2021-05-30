/**
 * jspsych-cued-drawing
 * Judy Fan
 *
 * plugin for displaying a cue and getting drawing response
 *
 **/ 

jsPsych.plugins["jspsych-cued-drawing"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "jspsych-cued-drawing",
    parameters: {
      category: {
        type: jsPsych.plugins.parameterType.STRING, 
        pretty_name: 'cue_label',
        default: undefined,
        description: 'The label used to cue drawing.'
      }, 
      cue_html: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'cue image HTML',
        default: '<img src="imageURL" id="cue_html">',
        array: true,
        description: 'The html of the image cue used to prompt drawing. Can create own style.'
      },      
      imageURL: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'imageURLs',
        default: undefined,
        array: true,
        description: 'The URL for the image cues.'
      },

      cue_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'cue_duration',
        default: 1000,
        description: 'How long to show the cue (in milliseconds).'
      }, 
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'prompt',
        default: null,
        description: 'What to display to the participant as the instructions.'
      }                       
    }
  }

  plugin.trial = function(display_element, trial) {    
    var sketchID = trial.gameID + '_' + trial.category // want to add trial.SONAID in case they refresh the page (keeps SONAID, changes gameID)

    // init global timestamps
    var startTrialTime, startResponseTime, startStrokeTime, endStrokeTime, endTrialTime;

    // global drawing params
    var drawingAllowed = true;
    var submitAllowed = false; // allow submission once there is at least one stroke
    var strokeColor = 'black';
    var strokeWidth = 5;
    var simplifyParam = 10;
    var currStrokeNum = 0;      
      
    // print errors if the parameters are not correctly formatted 
    if(typeof trial.category === 'undefined'){
      console.error('Required parameter "category" missing in jspsych-cued-drawing');
    }   

    // wait for a little bit for data to come back from db, then show_display
    setTimeout(function() {show_cue(); }, 400); 

    // wrapper function to show cue, this is called when you've waited as long as you
    // reckon is long enough for the data to come back from the db
    function show_cue() {    
      var html = ''; 
      
      html += '<div id="occluder2"><p style="padding-top: 5%;padding-bottom: 2%;"></div>'
      if (trial.practiceTrial === true) {
        html += '<div id="occluder"><p style="padding-top: 5%;padding-bottom: 2%;">Look at this <b>FACE</b>!</p>\
                 <img height = "300" src = "./stimuli/Surprised_Pikachu.png"></div>'      
      } else if (trial.category === "bread") {
        html += '<div id="occluder"><p style="padding-top: 5%;padding-bottom: 2%;">Look at this piece of <b>'+ trial.category.toUpperCase()+ '</b>!</p>\
                 <img height = "300" src = "' + trial.imageURL + '"></div>'      
      } else {
        //a_or_an = ['a','e','i','o','u'].includes(trial.category[0])      + (a_or_an ? 'an <b>' : 'a <b>')
        html += '<div id="occluder"><p style="padding-top: 5%;padding-bottom: 2%;">Look at this <b>' + (trial.category === 'car_(sedan)' ? 'car (sedan)' : trial.category).toUpperCase() + '</b>!</p>\
                 <img height = "300" src = "' + trial.imageURL + '"></div>'
      }

      // create sketchpad 
      html += '<div class="centered" id="sketchpad_container" style="display:none">'; // 
      html += '<canvas id="sketchpad" style="display:none"></canvas>';
      html += '</div>';

      if (trial.practiceTrial !== true) {
        html += '<p id="trialNum">trial '+ (trial.trialNum + 1) + " of " + trial.numTrials + '</p>';
      }

      html += '<div id="label_container" style="display:none"><p id="cue_label"> '+ (trial.category === 'car_(sedan)' ? 'car (sedan)' : trial.category) +'</p></div>';

      // display button to submit drawing when finished
      html += '<button id="submit_button" class="green" style="display:none"> submit </button>'

      // actually assign html to display_element.innerHTML
      display_element.innerHTML = html;

      // reset global current stroke number variable
      currStrokeNum = 0;

      // record trial start timestamp
      startTrialTime = Date.now();

      // warn against refreshing page
      window.onbeforeunload = function() {
        return "Data will be lost if you leave the page, and the experiment start over again. Are you sure?";
      };

      // add event listener to submit button once response window opens
      submit_button.addEventListener('click', end_trial);
      submit_button.disabled = true; // button is disabled until at least one stroke   

      // occluder to help with trial transitions
      if (trial.practiceTrial === true) {
        jsPsych.pluginAPI.setTimeout(function() {$('#occluder2').fadeOut('fast');},1500);
        jsPsych.pluginAPI.setTimeout(function() {$('#occluder').fadeOut('fast');},9500);
      } else {
        jsPsych.pluginAPI.setTimeout(function() {$('#occluder2').fadeOut('fast');},1500);
        jsPsych.pluginAPI.setTimeout(function() {$('#occluder').fadeOut('fast');},9500);
      }
      // instantiate new sketchpad
      sketchpad = new Sketchpad();
      var tool = new Tool();
      sketchpad.setupTool();

      // // if you need to reposition the label_container do so now
      // if (trial.condition == 'text') {
      //   var label = display_element.querySelector('#label_container');
      //   label.style.top = "300px";
      //   label.style.left = "100px";
      // }

      // show the cues
      jsPsych.pluginAPI.setTimeout(function() {
        $('#sketchpad_container').fadeIn('fast');
        $('#label_container').fadeIn('fast');        
        $('#cue_container').fadeIn('fast');       
      }, 100);

      // wait for the cue duration, then trigger display of the drawing canvas
      jsPsych.pluginAPI.setTimeout(function() {show_canvas();}, trial.cue_duration);

    }  

    function show_canvas() {       

      // record timestamp for start of response window
      startResponseTime = Date.now(); 

      // show the canvas
      $('#sketchpad').fadeIn('fast');

      // show the submit button
      $('#submit_button').fadeIn('fast');        

    }

    // send stroke data back to server to save to db
    function send_stroke_data(path) {
      // path.selected = false;
      var svgString = path.exportSVG({asString: false}).getAttribute('d');


      // specify other metadata
      stroke_data = _.extend({}, trial, {
          eventType: 'stroke',
          gameID: trial.gameID,
          //SONA_ID: trial.SONA_ID,
          prolificID : trial.prolificID,
          studyID : trial.studyID,
          sessionID : trial.sessionID,
          sketchID: sketchID,
          strokeID: sketchID + '_' + currStrokeNum,
          svg: svgString,
          arcLength: path.length,
          dbname: trial.dbname,
          colname: trial.colname,
          iterationName: trial.iterationName,             
          currStrokeNum: currStrokeNum,
          simplifyParam: simplifyParam,
          startResponseTime: startResponseTime,
          startStrokeTime: startStrokeTime,
          endStrokeTime: endStrokeTime,
          time: Date.now()
      });

	    console.log('stroke_data',stroke_data);

      // send stroke data to server
      socket.emit('stroke',stroke_data);    

    }

    // triggered either when submit button is clicked or time runs out
    // sends trial data to database
    function end_trial() {

      // disable button to prevent double firing
      submit_button.disabled=true;

      // sketch rendering to base64 encoding
      var dataURL = display_element.querySelector('#sketchpad').toDataURL();
      dataURL = dataURL.replace('data:image/png;base64,','');      

      // data saving
      var trial_data = _.extend({}, trial, {
          eventType: 'sketch',
          numStrokes: currStrokeNum,          
          pngData: dataURL,
          gameID: trial.gameID,   
          sketchID: sketchID,
          //SONA_ID: trial.SONA_ID,
          prolificID : trial.prolificID,
          studyID : trial.studyID,
          sessionID : trial.sessionID,
          dbname: trial.dbname,
          colname: trial.colname, 
          iterationName: trial.iterationName,   
          startTrialTime: startTrialTime, 
          endTrialTime: Date.now(),   
          totalTrialTime: Date.now() - startTrialTime
      });

      // clear the HTML in the display element
      display_element.innerHTML = '';
      
      // clear sketchpad canvas and reset drawing state vars
      project.activeLayer.removeChildren();
      
      // end trial
      jsPsych.finishTrial(trial_data);

    }


    ///////// CORE DRAWING FUNCTIONS ///////////

    function Sketchpad() {
      // initialize paper.js      
      paper.setup('sketchpad');
      // when the section below is commented out, it defaults to width 300px, height 150px
      paper.view.viewSize.width = 300; 
      paper.view.viewSize.height = 300; 
    };

    Sketchpad.prototype.setupTool = function() {    
      // initialize path and tool
      var path;

      // define mouse interaction events
      tool.onMouseDown = function(event) {        
        startStroke(event);        
      }

      tool.onMouseDrag = function(event) {
        if (drawingAllowed && !_.isEmpty(path)) {
          var point = event.point.round();
          currMouseX = point.x;
          currMouseY = point.y;
          path.add(point);
        }
      };      

      tool.onMouseUp = function (event) {
        endStroke(event);                
      }

      // startStroke
      function startStroke(event) {
          if (drawingAllowed) {
            startStrokeTime = Date.now();
            // If a path is ongoing, send it along before starting this new one
            if(!_.isEmpty(path)) {
              endStroke(event);
            }

            var point = (event ? event.point.round() :
             {x: currMouseX, y: currMouseY});
              path = new Path({
                segments: [point],
                strokeColor: strokeColor,
                strokeWidth: strokeWidth
              });
          }
        };

      // endStroke
      function endStroke(event) {
        // Only send stroke if actual line (single points don't get rendered)
        try {
          
          if (drawingAllowed && path.length > 1) {
            
            // allow submission of button if endStroke is called 
            submit_button.disabled=false;

            // record end stroke time
            endStrokeTime = Date.now();
            
            // Increment stroke num
            currStrokeNum += 1;

            // Simplify path to reduce data sent
            path.simplify(simplifyParam);

            // send stroke data to db.
            if (trial.practiceTrial !== true) {
              send_stroke_data(path);
            }

            // reset path
            path = [];
          } 

        } catch (error) {
          if (error instanceof TypeError) {
            console.log('caught it')
            console.log(error)
          }
        }
      }
    
    }

  };

  return plugin;
})();


