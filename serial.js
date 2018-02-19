


const SerialPort = require('serialport')
const express = require('express')
const fs = require('fs')
const http = require('http')
const app = express()
const webSocketServer = require('websocket').server;


process.title = 'simcap'
console.log("__dirname:", __dirname)

var state = {
	serial: null,
	serialConnected: false,
	labels: ["A", "B", "C", "E", "F", "G", "H", "I", "J"],
	activeLabel: "B",
	lastData: undefined,
	runs: [],
	clients: []
}


function getRuns() {
	return fs.readdirSync( __dirname + '/logs' )
	.filter((x) => x[0] != '.' )
	.filter((x) => x.indexOf('log-') === 0 )
	.map((x) => {
		return x.split('-')[1].split('.')[0]
	})
}

function getState() {
	return {
		serialConnected: state.serialConnected,
		labels: state.labels,
		activeLabel: state.activeLabel,
		runs: getRuns(),
		lastData: state.lastData,
		clients: state.clients.length
	}
}


function setState(key, value) {
	if (key === 'activeLabel') {
		if (state[key] === value) state[key] = null;
		else state[key] = value;
	}
	updateClients()
}

function handleWSMessage(msg) {
	if(msg.type == 'utf8') {
		console.log('WS Message: ', msg );
		try {
			const json = JSON.parse(msg.utf8Data)
			if (json.toggle == 'serialConnected') toggleSerialConnection();
			if (json.set && json.set.length == 2) setState( json.set[0], json.set[1] )
		} catch(e) {
			console.error(e)
			console.log('This doesn\'t look like a valid JSON: ', msg);
			return;
		}
	}
}

function handleWSRequest (req) {
	console.log( 'WS Connection request ' + (new Date()) + '' + req.origin )
	const connection = req.accept( null, req.origin )
	const index = state.clients.push( connection ) -1
	connection.sendUTF( JSON.stringify( getState() ))

	connection.on('message', handleWSMessage )
	connection.on('close', function(con){
		console.log('WS Connection close', connection.remoteAddress, index )
		state.clients.splice( index, 1 )
	})
}


function toggleSerialConnection() {
	if (state.serialConnected) {
		state.serialConnected = false;
		state.serial.close( () => {
			state.serial = null;
			updateClients();
		})

	} else {
		const ID = Date.now()
		state.serialConnected = true
		state.runs.push( ID )
		state.serial = new SerialPort('/dev/ttyUSB0', {
			baudRate: 115200
		})

		var log = require('fs').createWriteStream( 'logs/log-' + ID + '.csv');

		state.serial.on('data', (data) => {
			console.log(data.toString());
			if (data.indexOf(',') > -1) {
				if (data.indexOf('heading') > -1 ) log.write( 'timestamp, label, ' + data)
				else {
					const dataLine = Date.now() + ', ' + state.activeLabel + ', ' + data 
					state.lastData = dataLine.split(',')
					updateClients();
					log.write( dataLine )
				}
			}
		});

		state.serial.on('error', (err) => {
			console.error(err)
		});
	}
}

function updateClients() {
	state.clients.forEach((destination) => {
        destination.sendUTF( JSON.stringify( getState() ) );
    });
}

const server = http.createServer( app )
const wss = new webSocketServer( { httpServer: server } )


wss.on('request', handleWSRequest )

app.use(express.static('public'))

app.get('/', (req, res) => res.sendFile('./public/index.html'))
app.get('/data', (req, res) => res.send("var DATA = " + JSON.stringify(state, null, null, 4) + ";") )
app.get('/runs', (req,res) => {
	res.json( getRuns() );
});
app.get('/runs/:runId', (req,res) => {
	fs.readFile( __dirname + '/logs/log-' + req.params.runId + '.csv' , 'utf8', function(err, contents) {
		res.send( err ? err : contents);	
	});
})


server.listen(3000, () => console.log('SIMCAP UI listening on port 3000!'))




