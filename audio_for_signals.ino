// audio_for_signals.ino
// ===== ESP32-S3: TCP + HMAC (clave HEX) + Streaming WAV (16 kHz mono 16-bit) =====
#include <WiFi.h>
#include <ESP_I2S.h>
#include <mbedtls/md.h>   // HMAC-SHA256
#include <math.h>
#include <esp_wifi.h>     // razones de desconexión, país, protocolo

// ---------- Wi-Fi ----------
const char* SSID = "DAVID";  // cámbialo si usas hotspot
const char* PASS = "Daga#1404";

// Retry Wi-Fi
const uint32_t WIFI_CONNECT_TIMEOUT_MS = 10000;
const uint8_t  WIFI_MAX_RETRIES        = 5;
const uint32_t WIFI_BACKOFF_BASE_MS    = 1000;

// Datos del AP objetivo (para evitar NO_AP_FOUND)
uint8_t g_bssid[6];
int     g_channel = -1;
bool    g_have_ap = false;

// ---------- Servidor ----------
#define AUDIO_SERVER_IP   "192.168.137.1"
#define AUDIO_SERVER_PORT 12345

// ---------- Clave compartida (HEX, 64 chars = 32 bytes) ----------
static const char* SHARED_KEY_HEX = "83e15c0a2b6a0f6f3a040a9f9b21f3c77e2b6d7d6d6e1e4a2b8c1d2e3f405062";

// ---------- Audio ----------
#define SAMPLE_RATE   16000U
#define SAMPLE_BITS   16
#define NUM_CHANNELS  1
#define BUFFER_SIZE   256

#define ENABLE_DSP         1
#define VOLUME_GAIN        4.0f
#define ALPHA              0.5f
#define QUIET_THRESHOLD    .0f
#define GATE_RELEASE_MS    150

I2SClass   i2s;
WiFiClient audioClient;

float prevHP_in=0, prevHP_out=0, prevLP=0;
int   quietRun=0;
static int16_t buf[BUFFER_SIZE];

// -------- Util DSP --------
inline float highPass(float s){ float o=0.95f*(prevHP_out + s - prevHP_in); prevHP_in=s; prevHP_out=o; return o; }
inline float lowPass(float s){ float y=ALPHA*s + (1.0f-ALPHA)*prevLP; prevLP=y; return y; }
inline float applyNoiseGate(float x){
  const int gateReleaseSamples = (GATE_RELEASE_MS * SAMPLE_RATE) / 1000;
  if (fabsf(x) < QUIET_THRESHOLD){ if (++quietRun > gateReleaseSamples) return 0.0f; }
  else quietRun = 0;
  return x;
}

// Lee N muestras int16 LE desde I2S
size_t readSamples(int16_t* dst, size_t n){
  size_t needBytes = n * sizeof(int16_t);
  size_t got = 0;
  while (got < needBytes){
    int r = i2s.readBytes((char*)dst + got, needBytes - got);
    if (r < 0) continue;
    got += (size_t)r;
  }
  return got / sizeof(int16_t);
}

// TCP helpers con timeouts amplios
bool recvExact(uint8_t* dst, size_t n, uint32_t timeout_ms = 15000){
  uint32_t start = millis(); size_t got = 0;
  while (got < n){
    if (!audioClient.connected()) return false;
    int avail = audioClient.available();
    if (avail > 0){
      int r = audioClient.read(dst + got, n - got);
      if (r < 0) return false; got += r; start = millis(); continue;
    }
    if (millis() - start > timeout_ms) return false;
    delay(1); yield();
  }
  return true;
}

bool sendAll(const uint8_t* src, size_t n, uint32_t timeout_ms = 15000){
  uint32_t start = millis(); size_t sent = 0;
  while (sent < n){
    if (!audioClient.connected()) return false;
    int w = audioClient.write(src + sent, n - sent);
    if (w > 0){ sent += w; start = millis(); continue; }
    if (millis() - start > timeout_ms) return false;
    delay(1); yield();
  }
  return true;
}

bool hmac_sha256(const uint8_t* key, size_t keyLen,
                 const uint8_t* msg, size_t msgLen,
                 uint8_t out[32]){
  const mbedtls_md_info_t* md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  if (!md) return false;
  return mbedtls_md_hmac(md, key, keyLen, msg, msgLen, out) == 0;
}

// --- HEX -> bytes ---
bool hexToBytes(const char* hex, uint8_t* out, size_t outCap, size_t& outLen){
  size_t n = strlen(hex); if (n % 2 != 0 || outCap < n/2) return false; outLen = n/2;
  auto nib = [](char c)->int{ if('0'<=c&&c<='9')return c-'0'; c|=0x20; if('a'<=c&&c<='f')return 10+(c-'a'); return -1; };
  for (size_t i=0;i<n;i+=2){ int hi=nib(hex[i]), lo=nib(hex[i+1]); if(hi<0||lo<0) return false; out[i/2]=(uint8_t)((hi<<4)|lo); }
  return true;
}

// --- WAV header (44 bytes) ---
void generateWavHeader(uint8_t* h, uint32_t dataSize, uint32_t sr){
  uint32_t chunkSize  = dataSize + 36;
  uint32_t byteRate   = sr * NUM_CHANNELS * (SAMPLE_BITS/8);
  uint16_t blockAlign = NUM_CHANNELS * (SAMPLE_BITS/8);
  memcpy(h,   "RIFF", 4);
  h[4]=chunkSize; h[5]=chunkSize>>8; h[6]=chunkSize>>16; h[7]=chunkSize>>24;
  memcpy(h+8, "WAVEfmt ", 8);
  h[16]=16; h[17]=0; h[18]=0; h[19]=0;
  h[20]=1;  h[21]=0;
  h[22]=NUM_CHANNELS; h[23]=0;
  h[24]=sr; h[25]=sr>>8; h[26]=sr>>16; h[27]=sr>>24;
  h[28]=byteRate; h[29]=byteRate>>8; h[30]=byteRate>>16; h[31]=byteRate>>24;
  h[32]=blockAlign; h[33]=0;
  h[34]=SAMPLE_BITS; h[35]=0;
  memcpy(h+36, "data", 4);
  h[40]=dataSize; h[41]=dataSize>>8; h[42]=dataSize>>16; h[43]=dataSize>>24;
}

// --- Handshake HMAC ---
bool do_authentication(){
  uint8_t nonce[32];
  if (!recvExact(nonce, sizeof(nonce))) { Serial.println("AUTH: fallo NONCE"); return false; }

  uint8_t mac6[6];
  WiFi.macAddress(mac6);
  const size_t devLen = 6;

  uint8_t msg[32 + 6];
  memcpy(msg, nonce, 32);
  memcpy(msg + 32, mac6, 6);

  uint8_t keyBytes[64]; size_t keyLen=0;
  if (!hexToBytes(SHARED_KEY_HEX, keyBytes, sizeof(keyBytes), keyLen)) {
    Serial.println("AUTH: clave HEX inválida"); return false;
  }
  if (keyLen != 32) {
    Serial.printf("AUTH: tamaño clave=%u (esperado 32)\n", (unsigned)keyLen); return false;
  }

  uint8_t tag[32];
  if (!hmac_sha256(keyBytes, keyLen, msg, sizeof(msg), tag)) {
    Serial.println("AUTH: fallo HMAC"); return false;
  }

  Serial.printf("[auth.debug] key_len=%u head=%02X%02X%02X%02X\n",
                (unsigned)keyLen, keyBytes[0], keyBytes[1], keyBytes[2], keyBytes[3]);
  Serial.print("[auth.debug] nonce="); for(int i=0;i<32;i++){ Serial.printf("%02X", nonce[i]); } Serial.println();
  Serial.print("[auth.debug] dev_id="); for(int i=0;i<6;i++){ Serial.printf("%02X", mac6[i]); } Serial.println();
  Serial.print("[auth.debug] tag="); for(int i=0;i<32;i++){ Serial.printf("%02X", tag[i]); } Serial.println();

  uint8_t payload[44];
  memcpy(payload, "AUTH1", 5);
  payload[5] = (uint8_t)devLen;
  memcpy(payload + 6, mac6, devLen);
  memcpy(payload + 12, tag, 32);

  bool sent = sendAll(payload, 12) && sendAll(payload + 12, 32);
  Serial.printf("[auth.debug] sent AUTH1=%s\n", sent ? "yes" : "no");
  if (!sent) return false;

  uint8_t resp[2];
  if (!recvExact(resp, 2)) {
    Serial.println("[auth.debug] no response to AUTH1");
    return false;
  }
  Serial.printf("[auth.debug] resp=%c%c\n", resp[0], resp[1]);
  return (resp[0]=='O' && resp[1]=='K');
}

// ==================== DIAGNÓSTICO WIFI ====================
const char* wifiReasonToStr(uint8_t r){
  switch (r){
    case 2:  return "AUTH_EXPIRE";
    case 3:  return "AUTH_LEAVE";
    case WIFI_REASON_NO_AP_FOUND:       return "NO_AP_FOUND";
    case WIFI_REASON_AUTH_FAIL:         return "AUTH_FAIL";
    case WIFI_REASON_HANDSHAKE_TIMEOUT: return "HANDSHAKE_TIMEOUT";
    case WIFI_REASON_BEACON_TIMEOUT:    return "BEACON_TIMEOUT";
    default: return "OTHER";
  }
}

void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info){
  switch(event){
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.printf("WiFi: asociado a %s ch%d\n", WiFi.SSID().c_str(), WiFi.channel());
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      Serial.printf("WiFi: desconectado, reason=%u %s\n",
                    info.wifi_sta_disconnected.reason,
                    wifiReasonToStr(info.wifi_sta_disconnected.reason));
      break;
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.printf("WiFi: IP %s  GW %s  MASK %s  RSSI %d\n",
        WiFi.localIP().toString().c_str(),
        WiFi.gatewayIP().toString().c_str(),
        WiFi.subnetMask().toString().c_str(),
        WiFi.RSSI());
      break;
    default: break;
  }
}

void scanAndReport(const char* ssid){
  Serial.println("Escaneando redes 2.4 GHz...");
  int n = WiFi.scanNetworks(false, true);
  if (n <= 0){ Serial.println("No se encontraron redes"); return; }
  bool seen = false;
  for (int i=0;i<n;i++){
    String s = WiFi.SSID(i);
    int ch = WiFi.channel(i);
    wifi_auth_mode_t enc = (wifi_auth_mode_t)WiFi.encryptionType(i);
    Serial.printf("  %2d) %-32s  RSSI %4d  ch%-2d  enc %d\n",
                  i+1, s.c_str(), WiFi.RSSI(i), ch, (int)enc);
    if (s == ssid){ seen = true; Serial.println("     ^-- objetivo encontrado"); }
  }
  if (!seen) Serial.println("Tu SSID no aparece. Probable 5 GHz, canal fuera de rango, oculto o AP lejano.");
}

bool findTargetAp(const char* ssid){
  int n = WiFi.scanNetworks(false, true);
  for (int i=0;i<n;i++){
    if (WiFi.SSID(i) == ssid){
      const uint8_t* b = WiFi.BSSID(i);
      memcpy(g_bssid, b, 6);
      g_channel = WiFi.channel(i);
      g_have_ap = true;
      Serial.printf("AP objetivo: ch%d  BSSID %02X:%02X:%02X:%02X:%02X:%02X\n",
        g_channel, g_bssid[0],g_bssid[1],g_bssid[2],g_bssid[3],g_bssid[4],g_bssid[5]);
      return true;
    }
  }
  g_have_ap = false;
  return false;
}

void applyStaSecurityOnce(){
  wifi_config_t cfg;
  esp_wifi_get_config(WIFI_IF_STA, &cfg);
  cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;  // fuerza WPA2-PSK
  cfg.sta.pmf_cfg.capable    = true;                // PMF opcional
  cfg.sta.pmf_cfg.required   = false;
  esp_wifi_set_config(WIFI_IF_STA, &cfg);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
}
// =========================================================

// --- Conexión Wi-Fi con timeout y reintentos (sin set_config durante conexión) ---
bool connectWiFi(uint8_t maxRetries = WIFI_MAX_RETRIES) {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);

  WiFi.disconnect(true, true);
  delay(100);

  Serial.println("Conectando a Wi-Fi");
  for (uint8_t attempt = 1; attempt <= maxRetries; ++attempt) {
    if (g_have_ap && g_channel > 0) {
      WiFi.begin(SSID, PASS, g_channel, g_bssid, true);
    } else {
      WiFi.begin(SSID, PASS);
    }

    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - start) < WIFI_CONNECT_TIMEOUT_MS) {
      delay(5000); Serial.print("."); yield();
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("\nIP: %s (intento %u)\n", WiFi.localIP().toString().c_str(), attempt);
      return true;
    }

    Serial.printf("\nTiempo agotado esperando Wi-Fi (intento %u). Reintentando...\n", attempt);
    WiFi.disconnect(true, true);
    delay(WIFI_BACKOFF_BASE_MS * attempt);
  }

  Serial.println("No se pudo conectar a Wi-Fi tras varios intentos.");
  return false;
}

void setup(){
  Serial.begin(115200); delay(100);

  WiFi.onEvent(onWiFiEvent);
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  // País para canales 1–13
  wifi_country_t c = {"MX", 1, 13, WIFI_COUNTRY_POLICY_MANUAL};
  esp_wifi_set_country(&c);

  scanAndReport(SSID);
  findTargetAp(SSID);     // fija canal/BSSID si está visible
  applyStaSecurityOnce(); // configura WPA2 + PMF opcional una sola vez

  connectWiFi(); // si falla, loop() seguirá intentando

  i2s.setPinsPdmRx(/*clk*/42, /*data*/41);
  if (!i2s.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)){
    Serial.println("Error I2S"); while(1) delay(1000);
  }
  Serial.println("I2S listo");
}

void loop(){
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi desconectado. Reintentando...");
    // vuelve a buscar canal/BSSID por si el AP cambió
    findTargetAp(SSID);
    if (!connectWiFi()) { delay(1000); return; }
  }

  if (!audioClient.connected()){
    Serial.printf("Conectando a %s:%d\n", AUDIO_SERVER_IP, AUDIO_SERVER_PORT);
    while (WiFi.status() == WL_CONNECTED && !audioClient.connect(AUDIO_SERVER_IP, AUDIO_SERVER_PORT)) {
      delay(1000); Serial.print(".");
    }
    if (WiFi.status() != WL_CONNECTED) return;

    audioClient.setNoDelay(true);

    IPAddress rip = audioClient.remoteIP();
    Serial.printf("\nTCP conectado a %s:%u\n", rip.toString().c_str(), AUDIO_SERVER_PORT);

    Serial.println("Autenticando...");
    if (!do_authentication()){
      Serial.println("Autenticación fallida (no llegó NONCE o conexión cerrada).");
      audioClient.stop(); delay(1000); return;
    }
    Serial.println("Autenticación OK");

    uint8_t header[44];
    generateWavHeader(header, 0xFFFFFFFF, SAMPLE_RATE);
    if (!sendAll(header, sizeof(header))){
      Serial.println("Fallo enviando header WAV"); audioClient.stop(); delay(500); return;
    }
  }

  readSamples(buf, BUFFER_SIZE);

#if ENABLE_DSP
  for (size_t i=0;i<BUFFER_SIZE;++i){
    float x  = (float)buf[i];
    float hp = highPass(x);
    float lp = lowPass(hp);
    float g  = applyNoiseGate(lp);
    int32_t y = (int32_t)(g * VOLUME_GAIN);
    if (y > INT16_MAX) y = INT16_MAX;
    if (y < INT16_MIN) y = INT16_MIN;
    buf[i] = (int16_t)y;
  }
#endif

  if (!sendAll((uint8_t*)buf, sizeof(buf))){
    Serial.println("Fallo envío. Reintentando...");
    audioClient.stop();
  }
  yield();
}


