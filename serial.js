var SerialPort = require('serialport')
var serialPort = new SerialPort('/dev/ttyUSB0', {
	baudRate: 115200
})

var log = require('fs').createWriteStream( 'logs/log-' + Date.now() + '.cvs');

serialPort.on('data', function(data){
	console.log(data.toString());
	log.write(Date.now() + ', ' + data)
});



