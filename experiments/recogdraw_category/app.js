global.__base = __dirname + '/';

var
    use_https     = true,
    argv          = require('minimist')(process.argv.slice(2)),
    https         = require('https'),
    fs            = require('fs'),
    app           = require('express')(),
    _             = require('lodash'),
    parser        = require('xmldom').DOMParser,
    XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest,
    sendPostRequest = require('request').post;


////////// EXPERIMENT GLOBAL PARAMS //////////

var researchers = ['A4SSYO0HDVD4E', 'A9AHPCS83TFFE', 'A17XT5MJVPU37V'];
//////////////////////////////////////////////

var gameport;

if(argv.gameport) {
  gameport = argv.gameport;
  console.log('using port ' + gameport);
} else {
  gameport = 8883;
  console.log('no gameport specified: using 8883\nUse the --gameport flag to change');
}

try {
  var   privateKey  = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/privkey.pem'), 
        certificate = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/cert.pem'),
        intermed    = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/chain.pem'),
        options     = {key: privateKey, cert: certificate, ca: intermed},
        server      = require('https').createServer(options,app).listen(gameport),
        io          = require('socket.io')(server);
} catch (err) {
  console.log("cannot find SSL certificates; falling back to http");
  var   server      = app.listen(gameport),
        io          = require('socket.io')(server);
}

app.get('/*', (req, res) => {
  serveFile(req, res);
});

io.on('connection', function (socket) {
  // tell client index num & gameID on connecting, and update mongo to not repeat indices 
  initializeWithIndex(socket);

  // write data to db upon getting current data
  socket.on('currentData', function(data) {
    console.log('currentData received: ' + JSON.stringify(data));
    // Increment games list in mongo here
    writeDataToMongo(data);
  });

  socket.on('stroke', function(data) {
    console.log('stroke data received: ' + JSON.stringify(data));
    // Increment games list in mongo here
    writeDataToMongo(data);
  });  

  socket.on('getStim', function(data) {
    sendSingleStim(socket, data);
  });
});

FORBIDDEN_FILES = ["auth.json"]
var serveFile = function(req, res) {
  var fileName = req.params[0];
  if(FORBIDDEN_FILES.includes(fileName)){
    // Don't serve files that contain secrets
    console.log("Forbidden file requested:" + filename);
    return; 
  }
  console.log('\t :: Express :: file requested: ' + fileName);
  return res.sendFile(fileName, {root: __dirname});
};

var UUID = function() {
  var baseName = (Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10));
  var template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
  var id = baseName + '-' + template.replace(/[xy]/g, function(c) {
    var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
  return id;
};

function initializeWithIndex(socket) {
  var gameid = UUID();
  sendPostRequest('http://localhost:6002/db/getIndex', {
    json: {
      dbname : 'photodraw',
      colname: 'recogdraw_category_stims',
      gameid: gameid
    }
  }, (error, res, body) => {
    if (!error && res.statusCode === 200) {
      // send index num (and id) to client
      socket.emit('onConnected',  {
        batchNum: body.sketch_ind,
        gameid: gameid
      });
    } else {
      console.log(`error getting stims: ${error} ${body}`); // not logged
    }
  });
}

function sendSingleStim(socket, data) {
  sendPostRequest('http://localhost:6002/db/getsinglestim', {
    json: {
      dbname: 'stimuli',
      colname: 'photodraw2',
      numTrials: 1,
      gameid: data.gameID
    }
  }, (error, res, body) => {
    if (!error && res.statusCode === 200) {
      socket.emit('stimulus', body);
    } else {
      console.log(`error getting stims: ${error} ${body}`);
      console.log(`falling back to local stimList`);
      socket.emit('stimulus', {
        stim: _.sampleSize(require('./photodraw2_meta.js'), 1)
      });
    }
  });
}

var writeDataToMongo = function(data) {
  sendPostRequest(
    'http://localhost:6002/db/insert',
    { json: data },
    (error, res, body) => {
      if (!error && res.statusCode === 200) {
        console.log(`sent data to store`);
      } else {
	      console.log(`error sending data to store: ${error} ${body}`);
      }
    }
  );
};
