// Tempo burner ring: an ESP32 + 24-LED WS2812 ring mounted around/behind a burner.
// Polls /api/live/<burner>?format=led once a second and shows the same colour
// language as the station screens:
//   amber fill = heating (fills as the pot approaches the boil)
//   red        = boiling, add ingredients when it blinks
//   teal fill  = cooking timer
//   green blink= done, plate now      yellow blink = foam, heat lowered
// Mount the ring >30 cm from the flame, behind a heat shield; the electronics
// never touch the stove's controls.
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>

const char* WIFI_SSID = "kitchen";
const char* WIFI_PASS = "change-me";
const char* URL = "http://tempo.local/api/live/B1?format=led";   // one ring per burner
const int PIN = 5, N = 24;
Adafruit_NeoPixel ring(N, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  ring.begin(); ring.setBrightness(90); ring.show();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(300);
}

void loop() {
  HTTPClient http; http.begin(URL);
  if (http.GET() == 200) {
    StaticJsonDocument<256> d;
    if (!deserializeJson(d, http.getString())) {
      int r = d["rgb"][0], g = d["rgb"][1], b = d["rgb"][2];
      const char* pattern = d["pattern"]; float fill = d["fill"] | 0.0;
      bool on = !(strcmp(pattern, "blink") == 0 && (millis() / 500) % 2);
      int lit = strcmp(pattern, "fill") == 0 ? max(1, (int)(fill * N)) : N;
      for (int i = 0; i < N; i++) ring.setPixelColor(i, (on && i < lit) ? ring.Color(r, g, b) : 0);
      ring.show();
    }
  } else {
    for (int i = 0; i < N; i++) ring.setPixelColor(i, i % 6 == 0 ? ring.Color(40, 40, 40) : 0);  // offline marker
    ring.show();
  }
  http.end();
  delay(500);
}
