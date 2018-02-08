


const SerialPort = require('serialport')
const express = require('express')
const fs = require('fs')
const app = express()

var state = {
	serial: null,
	serialConnected: false,
	labels: ["A", "B", "C"],
	activeLabel: null,
	runs: []
}

app.use(express.static('public'))

app.get('/', (req, res) => res.sendFile('./public/index.html'))
app.get('/data', (req, res) => res.send("<code>" + JSON.stringify(state, null, null, 4) + "</code>") )
app.get('/runs/:runId', (req,res) => {
	fs.readFile( __dirname + '/logs/log-' + req.params.runId + '.csv' , 'utf8', function(err, contents) {
		res.send( err ? err : contents);	
	});
})

app.get('/toggle', (req, res) => {


	if (state.serialConnected) {
		state.serialConnected = false;
		state.serial.close( () => {
			// res.send("Completed /runs/" + state.runs[state.runs.length - 1])
			res.redirect('/')
			state.serial = null;
		})
	} else {
		const ID = Date.now()

		// res.send("Started /runs/" + ID)
		res.redirect('/')

		state.serialConnected = true
		state.runs.push( ID )
		state.serial = new SerialPort('/dev/ttyUSB0', {
			baudRate: 115200
		})

		var log = require('fs').createWriteStream( 'logs/log-' + ID + '.csv');

		state.serial.on('data', (data) => {
			console.log(data.toString());
			if (data.indexOf(',') > -1) {
				if (data.indexOf('heading') > -1 ) log.write( 'timestamp, ' + data)
				else log.write(Date.now() + ', ' + data)
			}
		});

		state.serial.on('error', (err) => {
			console.error(err)
		});
	}
})

app.listen(3000, () => console.log('SIMCAP UI listening on port 3000!'))





