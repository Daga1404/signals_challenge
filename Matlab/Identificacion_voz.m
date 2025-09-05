%% SCRIPT 2: IDENTIFICACIÓN DE VOZ
% Este script carga todos los datos de voz previamente grabados (archivos
% datos_TUNOMBRE.mat), entrena un modelo simple basado en centroides,
% graba una nueva muestra de voz y predice quién habló.

clearvars; close all; clc;

%% Cargar todos los archivos de datos de entrenamiento
% Busca todos los archivos que sigan el patrón 'datos_*.mat'
file_list = dir('datos_*.mat');

if isempty(file_list)
    error('No se encontraron archivos de datos (ej. "datos_Gal.mat"). Asegúrate de que estén en la misma carpeta que este script.');
end

fprintf('Se encontraron %d archivos de datos de voz. Cargando...\n', length(file_list));

% Estructuras para almacenar los datos y características
trainingData = {};
personNames = {};
all_features = [];
all_labels = [];
idx = 0;

for p = 1:length(file_list)
    fprintf('Cargando archivo: %s\n', file_list(p).name);
    data = load(file_list(p).name); % Carga S, fs, personName
    
    personNames{p} = data.personName;
    Ntakes = size(data.S, 2);
    
    % Extraer características de cada toma de la persona
    for k = 1:Ntakes
        idx = idx + 1;
        signal = data.S(:, k);
        features = compute_features(signal, data.fs);
        
        all_features(idx, :) = features;
        all_labels(idx) = p; % La etiqueta es el índice de la persona
    end
end

%% Calcular los centroides de características para cada persona
% Un centroide es el vector de características promedio de una persona.
numPersons = length(personNames);
numFeatures = size(all_features, 2);
centroids = zeros(numPersons, numFeatures);

for p = 1:numPersons
    % Encuentra todas las características de la persona 'p' y calcula su promedio
    centroids(p, :) = mean(all_features(all_labels == p, :), 1);
end

fprintf('\n=== Entrenamiento completado. Se calcularon los centroides para: %s ===\n', strjoin(personNames, ', '));

%% PREDICCIÓN: Grabar una nueva voz y predecir quién es
fprintf('\n================================================\n');
fprintf('Ahora, graba una nueva muestra de voz para identificarla.\n');
fprintf('Presiona cualquier tecla cuando estés listo para empezar.\n');
pause;

% Parámetros para la grabación de prueba
Tblock = 3;
fs = 22050;
Nsamples = Tblock * fs;

% Conteo regresivo
fprintf('¡Prepárate! La grabación comenzará en:\n');
for countdown = 3:-1:1
    fprintf('%d...\n', countdown);
    pause(1);
end

RecObj = audiorecorder(fs, 16, 1);
fprintf('>> HABLA AHORA durante %.1f segundos...\n', Tblock);
recordblocking(RecObj, Tblock);
fprintf('>> Fin de la grabación.\n');

% Obtener y acondicionar la señal de prueba
s_raw = getaudiodata(RecObj);
if numel(s_raw) >= Nsamples
    s_test = s_raw(1:Nsamples);
else
    s_test = [s_raw; zeros(Nsamples - numel(s_raw), 1)];
end
mx = max(abs(s_test));
if mx > 0
    s_test = s_test ./ mx;
end

%% Clasificación (Vecino más cercano al centroide)
% 1. Extraer características de la nueva grabación
F_test = compute_features(s_test, fs);

% 2. Calcular la distancia euclidiana desde F_test a cada centroide
% La función vecnorm calcula la norma (distancia) de forma eficiente.
distances = vecnorm(centroids - F_test, 2, 2);

% 3. Encontrar el centroide con la distancia mínima
[min_dist, predicted_index] = min(distances);

% 4. Obtener el nombre de la persona correspondiente
predicted_person = personNames{predicted_index};

fprintf('\n================================================\n');
fprintf('      >>> La predicción es: %s\n', predicted_person);
fprintf('================================================\n');


%% ===== Funciones Auxiliares para Extracción de Características =====
% Estas funciones calculan descriptores numéricos de la señal de voz.

function f_vector = compute_features(x, fs)
    % 1. Energía Logarítmica: Mide la "potencia" de la señal.
    logE = 10 * log10(sum(x.^2) + eps);

    % 2. Tasa de Cruce por Cero (ZCR): Relacionado con la "agudeza" del sonido.
    zcr = zero_crossing_rate(x);

    % --- Características espectrales (basadas en la FFT) ---
    N_fft = 2^nextpow2(length(x));
    Y = fft(x, N_fft);
    mag = abs(Y(1:N_fft/2+1));
    f_axis = (0:N_fft/2)' * (fs / N_fft);
    
    % 3. Centroide Espectral: El "centro de masa" del espectro.
    centroid = sum(f_axis .* mag) / (sum(mag) + eps);
    
    % 4. Ancho de Banda Espectral: La "dispersión" del espectro.
    bw = sqrt(sum(((f_axis - centroid).^2) .* mag) / (sum(mag) + eps));

    % Devuelve un vector con todas las características calculadas.
    f_vector = [logE, zcr, centroid, bw];
end

function zcr = zero_crossing_rate(x)
    % Cuenta cuántas veces la señal cruza el eje cero.
    zcr = sum(abs(diff(sign(x)))) / (2 * length(x));
end
