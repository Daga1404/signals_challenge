%% SCRIPT 1: RECOPILACIÓN DE DATOS DE VOZ (EJECUTAR POR CADA PERSONA)
% Este script graba 5 muestras de audio de una persona, las procesa para
% obtener su espectro de frecuencia (FFT) y guarda las señales en un

% archivo .mat con el nombre de la persona.

clearvars; close all; clc;

%% Parámetros de Grabación
Tblock   = 3;       % 5 segundos por grabación
fs       = 22050;   % Frecuencia de muestreo (Hz)
Ntakes   = 3;       % 5 tomas por persona
Nsamples = Tblock*fs; % Número de muestras por toma

%% Solicitar nombre de la persona
% El nombre se usará para guardar el archivo de datos.
personName = input('Introduce tu nombre (sin espacios, ej. "Gal"): ', 's');
if isempty(personName)
    error('El nombre no puede estar vacío.');
end

fprintf('\nHola, %s. Vamos a grabar %d tomas de tu voz.\n', personName, Ntakes);

% Matriz para guardar las 5 señales de audio
S = zeros(Nsamples, Ntakes);

%% Bucle de Grabación
for k = 1:Ntakes
    fprintf('\n================================================\n');
    fprintf('Preparando para grabar la Toma %d/%d...\n', k, Ntakes);
    fprintf('Presiona cualquier tecla cuando estés listo para empezar.\n');
    pause;

    % Conteo regresivo para que el usuario se prepare
    fprintf('¡Prepárate! La grabación comenzará en:\n');
    for countdown = 3:-1:1
        fprintf('%d...\n', countdown);
        pause(1);
    end

    % Objeto para grabar audio
    RecObj = audiorecorder(fs, 16, 1);
    
    fprintf('>> HABLA AHORA durante %.1f segundos...\n', Tblock);
    recordblocking(RecObj, Tblock); % Graba durante Tblock segundos
    fprintf('>> Fin de la grabación de la Toma %d.\n', k);

    % Obtener los datos de audio como un vector
    s_raw = getaudiodata(RecObj);

    % --- Acondicionamiento de la señal ---
    % 1. Asegurar que la longitud sea exactamente Nsamples
    if numel(s_raw) >= Nsamples
        s = s_raw(1:Nsamples);
    else
        % Si la grabación es más corta, rellenar con ceros al final
        s = [s_raw; zeros(Nsamples - numel(s_raw), 1)];
    end

    % 2. Normalización de la amplitud a [-1, 1]
    % Esto hace que las grabaciones sean comparables en volumen.
    mx = max(abs(s));
    if mx > 0
        s = s ./ mx;
    end
    
    % Guardar la señal procesada en nuestra matriz
    S(:, k) = s;
end

fprintf('\n================================================\n');
fprintf('¡Excelente! Todas las grabaciones de %s han sido completadas.\n', personName);

%% Análisis de Frecuencia (Transformada de Fourier - FFT) y Gráfica
fprintf('Calculando y mostrando la Transformada de Fourier de las 5 tomas...\n');

figure('Name', ['Espectro de Frecuencia (FFT) - ' personName], 'Color', 'w', 'NumberTitle', 'off');
hold on; % Permite graficar múltiples líneas en la misma figura

% Preparar nombres para la leyenda de la gráfica
legend_entries = cell(1, Ntakes);

for k = 1:Ntakes
    % Obtener la señal de la toma actual
    x = S(:, k);
    
    % Calcular la FFT
    N_fft = 2^nextpow2(length(x)); % Siguiente potencia de 2 para eficiencia
    Y = fft(x, N_fft);
    P2 = abs(Y / N_fft); % Amplitud del espectro de 2 lados
    P1 = P2(1:N_fft/2+1); % Espectro de 1 lado
    P1(2:end-1) = 2 * P1(2:end-1);
    
    % Crear el eje de frecuencias
    f_axis = fs * (0:(N_fft/2)) / N_fft;
    
    % Graficar el espectro de la toma actual
    plot(f_axis, P1);
    legend_entries{k} = sprintf('Toma %d', k);
end

hold off;
grid on;
box on;
title(['Comparación del Espectro de Frecuencia para ' personName]);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
legend(legend_entries);
xlim([0, 5000]); % Limitar el eje X a las frecuencias más relevantes para la voz

%% Guardar los datos en un archivo .mat
filename = ['datos_' personName '.mat'];
save(filename, 'S', 'fs', 'personName');
fprintf('\nLas 5 grabaciones de voz han sido guardadas en el archivo: %s\n', filename);
