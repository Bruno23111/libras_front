// --- Configurações (Escopo Global) ---

window.videoAlfabeto = document.getElementById('webcam');
window.isAlfabetoTabActive = false; 
let isAlfabetoInitialized = false;  

let modelAlfabeto = null;
const MODEL_URL = './modelos_web/model.json';
const IMG_WIDTH = 128;
const IMG_HEIGHT = 128;
const CLASS_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "I", 8: "L", 9: "M",
    10: "N", 11: "O", 12: "P", 13: "Q", 14: "R",
    15: "S", 16: "T", 17: "U", 18: "V", 19: "W", 20: "Y"
};
let lastPredictions = [];
const SMOOTHING_WINDOW = 5;

// Elementos locais
const statusElement = document.getElementById('status');
const resultElement = document.getElementById('result');
const overlayCanvas = document.getElementById('overlay');
const overlayCtx = overlayCanvas.getContext('2d');
const alfabetoContent = document.getElementById('alfabeto-content');

// MUDANÇA: Variáveis para armazenar as coordenadas do quadrado (bounding box)
// para que a predição possa usá-las.
let boxX = 0, boxY = 0, boxSize = 0;

// CORREÇÃO: Criada função de inicialização para ser chamada pela aba
window.initAlfabeto = async function() {
    if (isAlfabetoInitialized) return; // Não inicializar duas vezes
    isAlfabetoInitialized = true;

    try {
        modelAlfabeto = await tf.loadLayersModel(MODEL_URL);
        // "Aquece" o modelo
        tf.tidy(() => {
            modelAlfabeto.predict(tf.zeros([1, IMG_WIDTH, IMG_HEIGHT, 3]));
        });
        statusElement.innerText = 'Modelo carregado. Iniciando webcam...';
        await startWebcam();
    } catch (err) {
        console.error("Erro ao carregar o modelo: ", err);
        statusElement.innerText = 'Erro ao carregar o modelo.';
    }
}

// Inicia a webcam
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }, // Pede alta resolução
            audio: false
        });
        window.videoAlfabeto.srcObject = stream;

        // MUDANÇA: Mostra o conteúdo IMEDIATAMENTE após a permissão da câmera.
        // Isso corrige o problema de "nada aparecer" se o onloadedmetadata atrasar.
        statusElement.classList.add("hidden");
        alfabetoContent.classList.remove("hidden");

        window.videoAlfabeto.onloadedmetadata = () => {
            window.videoAlfabeto.play();
            
            // Ajusta o tamanho do canvas para o tamanho do vídeo
            overlayCanvas.width = window.videoAlfabeto.videoWidth;
            overlayCanvas.height = window.videoAlfabeto.videoHeight;

            window.isAlfabetoTabActive = true; // Ativa o loop
            predictLoopAlfabeto();
        };
    } catch (err) {
        console.error("Erro ao acessar à webcam: ", err);
        statusElement.innerText = 'Erro ao acessar à webcam.';
    }
}

// Suavização de predições
function smoothPrediction(newLetter) {
    lastPredictions.push(newLetter);
    if (lastPredictions.length > SMOOTHING_WINDOW) lastPredictions.shift();
    const counts = {};
    lastPredictions.forEach(l => counts[l] = (counts[l] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

// Loop de predição
async function predictLoopAlfabeto() {
    // CORREÇÃO: Só executa se a aba estiver ativa e o vídeo pronto
    if (!window.isAlfabetoTabActive || !modelAlfabeto || window.videoAlfabeto.paused || window.videoAlfabeto.readyState < 2) {
        // Desenha o overlay mesmo que o vídeo esteja pausado
        if (window.isAlfabetoTabActive) drawOverlay(); 
        requestAnimationFrame(predictLoopAlfabeto);
        return;
    }

    const [letter, confidence] = tf.tidy(() => {
        const frame = tf.browser.fromPixels(window.videoAlfabeto);

        // MUDANÇA: Lógica de Corte (Crop)
        // Converte as coordenadas do quadrado (em pixels) para coordenadas normalizadas (0.0 a 1.0)
        // que o TensorFlow exige.
        const h = frame.shape[0];
        const w = frame.shape[1];
        
        const y1 = boxY / h;
        const x1 = boxX / w;
        const y2 = (boxY + boxSize) / h;
        const x2 = (boxX + boxSize) / w;

        // Corta o 'frame' usando as coordenadas do quadrado e redimensiona para o tamanho do modelo
        const cropped = tf.image.cropAndResize(
            frame.expandDims(0), // [1, h, w, 3]
            [[y1, x1, y2, x2]],  // Coordenadas normalizadas [y1, x1, y2, x2]
            [0],                 // box_ind
            [IMG_WIDTH, IMG_HEIGHT] // Tamanho final (128, 128)
        );
        
        // Normaliza os pixels (0-255 -> 0-1)
        const scaled = cropped.div(255.0);
        
        // Faz a predição USANDO a imagem cortada ('scaled')
        const prediction = modelAlfabeto.predict(scaled);
        
        const probabilities = prediction.dataSync();
        const predictedIndex = prediction.argMax(-1).dataSync()[0];
        const confidence = probabilities[predictedIndex];
        const letter = CLASS_MAP[predictedIndex];
        return [letter, confidence];
    });

    // Mostra resultado
    if (confidence > 0.7) {
        const smoothed = smoothPrediction(letter);
        resultElement.innerText = `${smoothed}`;
    } else {
        resultElement.innerText = "...";
    }

    drawOverlay(); // desenha o quadrado em cada frame
    requestAnimationFrame(predictLoopAlfabeto);
}

// CORREÇÃO: Função para desenhar o overlay (quadrado de referência)
function drawOverlay() {
    if (!overlayCanvas || !overlayCtx || !window.videoAlfabeto.videoWidth) {
        if (overlayCtx) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        return;
    }

    // Garante que o canvas tem o mesmo tamanho do vídeo
    overlayCanvas.width = window.videoAlfabeto.videoWidth;
    overlayCanvas.height = window.videoAlfabeto.videoHeight;

    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    // Limpa o canvas antes de desenhar
    overlayCtx.clearRect(0, 0, w, h);
    
    // MUDANÇA: Calcula e ATUALIZA as variáveis globais do quadrado
    boxSize = Math.min(w, h) * 0.7; 
    boxX = (w - boxSize) / 2;
    boxY = (h - boxSize) / 2;

    // Desenha o quadrado (linha tracejada)
    overlayCtx.strokeStyle = "#7C3AED"; // Violeta
    overlayCtx.lineWidth = 4;
    overlayCtx.setLineDash([10, 10]); 
    overlayCtx.strokeRect(boxX, boxY, boxSize, boxSize); // Usa as variáveis
    overlayCtx.setLineDash([]); 

    // Adiciona texto de instrução
    overlayCtx.font = "bold 24px 'Inter', sans-serif";
    overlayCtx.fillStyle = "#E0E0FF"; 
    overlayCtx.textAlign = "center";
    overlayCtx.textBaseline = "bottom"; 
    overlayCtx.fillText("Posicione sua mão aqui", w / 2, boxY - 10); // Usa boxY
}

// NÃO inicia sozinho. A função openTab() irá chamar window.initAlfabeto()

