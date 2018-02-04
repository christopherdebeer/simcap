const raspi = require('raspi');
const I2C = require('raspi-i2c').I2C;

raspi.init(() => {
	console.log("RASPI CONNECTED...")
	const i2c = new I2C();
	console.log("I2C: ", i2c )
	console.log("0x1E (mag):", i2c.readSync(0x1E)); // Read one byte from the magnetometer
	console.log("0x6B (acc/gyro):", i2c.readSync(0x6B)); // Read one byte from the accelerometer/gyroscope
});
