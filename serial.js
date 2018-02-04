


const SerialPort = require('serialport')
const express = require('express')
const app = express()

var state = {
	serial: null,
	serialConnected: false,
	labels: ["A", "B", "C"],
	activeLabel: null,
	runs: []
}

app.use(express.static('public'))

app.get('/', (req, res) => res.json(state))
app.get('/runs/:runId', (req,res) => {
	res.send("Show run [" + req.params.runId  + "] ...");	
})

app.get('/toggle', (req, res) => {


	if (state.serialConnected) {
		state.serialConnected = false;
		state.serial.close( () => {
			res.send("Completed /runs/" + state.runs[state.runs.length - 1])
			state.serial = null;
		})
	} else {
		const ID = Date.now()
		res.send("Started /runs/" + ID)

		state.serialConnected = true
		state.runs.push( ID )
		state.serial = new SerialPort('/dev/ttyUSB0', {
			baudRate: 115200
		})

		var log = require('fs').createWriteStream( 'logs/log-' + ID + '.cvs');

		state.serial.on('data', (data) => {
			console.log(data.toString());
			log.write(Date.now() + ', ' + data)
		});

		state.serial.on('error', (err) => {
			console.error(err)
		});
	}
})

app.listen(3000, () => console.log('SIMCAP UI listening on port 3000!'))





