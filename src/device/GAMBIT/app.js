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
