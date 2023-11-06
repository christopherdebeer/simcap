var log = [];
var state = 1;

function getData() {
    for (var i=0;i<log.length;i++)
        console.log(i+","+log[i]);
}


//NFC Detection
NRF.nfcURL("webble://christopherdebeer.github.io/simcap/src/web/GAMBIT/");
NRF.on('NFCon', function() {
    digitalPulse(LED2, 1, 500);//flash on green light
    console.log('nfc_field : [1]');
    NRF.setAdvertising({
        0x183e: [1],
    });
});
NRF.on('NFCoff', function() {
    digitalPulse(LED2, 1, 200);//flash on green light
    console.log('nfc_field : [0]');
    NRF.setAdvertising({
        0x183e: [0],
    });
});

//Movement Sensor
require("puckjsv2-accel-movement").on();
var idleTimeout;
Puck.on('accel',function(a) {
    // digitalWrite(LED1,1); //turn on red light
  if (idleTimeout) clearTimeout(idleTimeout);
  else
    if (state === 1) {
        console.log('movement : 1');
        log.push(`a:1}`)
        NRF.setAdvertising({
            0x182e: [1],
        });
    }
    idleTimeout = setTimeout(function() {
        idleTimeout = undefined;
        // digitalWrite(LED1,0);//turn off red light
        if (state === 1) {
            console.log('movement : 0');
            log.push(`a:0}`)
            NRF.setAdvertising({
                0x182e: [0],
            });
        }
    },500);  
});


//Magnetic Field Sensor
require("puckjsv2-mag-level").on();
Puck.on('field',function(m) {
    digitalPulse(LED2, 1, 200);//flash green light
    if (state === 1) {
        console.log('magnetic_field : [' + m.state + ']');
        NRF.setAdvertising({
            0x183a: [m.state],
        });
        log.push(`m:${m.state}`)
    }
});

//Button Press
//Turn Off/On MQTT Advertising
var pressCount = 0;
setWatch(function() {
    pressCount++;
    state = (pressCount+1)%2;
    if ((pressCount+1)%2) digitalPulse(LED3,1,1500); //long flash blue light
    else
        digitalPulse(LED3,1,100); //short flash blue light
    console.log('button_press_count : [' + pressCount + ']');
    console.log('button_state : [' + (pressCount+1) + ']');
    console.log('state: ' + state); 
    NRF.setAdvertising({
        0xFFFF : [pressCount],
        0x183c: [((pressCount+1)%2)],
    });
}, BTN, { edge:"rising", repeat:true, debounce:50 });