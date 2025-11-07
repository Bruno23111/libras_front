import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";



// ===================================================================
// PARTE 1: LÓGICA DO ALFABETO (TensorFlow.js)
// ===================================================================

// --- Configurações (Alfabeto) ---
let videoAlfabeto;
let isAlfabetoTabActive = false;
let isAlfabetoInitialized = false;
let modelAlfabeto = null;
let streamAlfabeto = null; // Para guardar o stream da câmara
const MODEL_URL = './modelos_web/model.json';
const IMG_WIDTH_ALFABETO = 128;
const IMG_HEIGHT_ALFABETO = 128;
const CLASS_MAP_ALFABETO = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "I", 8: "L", 9: "M",
    10: "N", 11: "O", 12: "P", 13: "Q", 14: "R",
    15: "S", 16: "T", 17: "U", 18: "V", 19: "W", 20: "Y"
};
let lastPredictionsAlfabeto = [];
const SMOOTHING_WINDOW_ALFABETO = 5;


// Elementos locais (Alfabeto)
let statusElementAlfabeto;
let resultElementAlfabeto;
let overlayCanvasAlfabeto;
let overlayCtxAlfabeto;
let alfabetoContent;

let boxX = 0, boxY = 0, boxSize = 0;

// Suavização de predições (Alfabeto)
function smoothPredictionAlfabeto(newLetter) {
    lastPredictionsAlfabeto.push(newLetter);
    if (lastPredictionsAlfabeto.length > SMOOTHING_WINDOW_ALFABETO) lastPredictionsAlfabeto.shift();
    const counts = {};
    lastPredictionsAlfabeto.forEach(l => counts[l] = (counts[l] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

// Função de inicialização (Alfabeto)
async function initAlfabeto() {
    if (isAlfabetoInitialized) return;
    isAlfabetoInitialized = true;

    try {
        try {
            await tf.setBackend('webgl');
            await tf.ready();
            console.log("TensorFlow.js a usar backend: WebGL");
        } catch (e) {
            console.warn("WebGL não suportado, a usar backend 'cpu'. O desempenho pode ser mais lento.");
            await tf.setBackend('cpu');
            await tf.ready();
        }

        modelAlfabeto = await tf.loadLayersModel(MODEL_URL);
        tf.tidy(() => {
            modelAlfabeto.predict(tf.zeros([1, IMG_WIDTH_ALFABETO, IMG_HEIGHT_ALFABETO, 3]));
        });
        statusElementAlfabeto.innerText = 'Modelo carregado. Iniciando webcam...';
    } catch (err) {
        console.error("Erro ao carregar o modelo (Alfabeto): ", err);
        statusElementAlfabeto.innerText = 'Erro ao carregar o modelo.';
    }
}

// Inicia a webcam (Alfabeto)
async function startWebcamAlfabeto() {
    try {
        videoAlfabeto.onplaying = () => {
            overlayCanvasAlfabeto.width = videoAlfabeto.videoWidth;
            overlayCanvasAlfabeto.height = videoAlfabeto.videoHeight;
            statusElementAlfabeto.classList.add("hidden");
            alfabetoContent.classList.remove("hidden");
            isAlfabetoTabActive = true;
            predictLoopAlfabeto();
        };

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: false
        });
        streamAlfabeto = stream; // Guarda o stream
        videoAlfabeto.srcObject = streamAlfabeto;
        await videoAlfabeto.play();

    } catch (err) {
        console.error("Erro ao acessar à webcam (Alfabeto): ", err);
        statusElementAlfabeto.innerText = 'Erro ao acessar à webcam.';
    }
}


// Loop de predição (Alfabeto)
async function predictLoopAlfabeto() {
    if (!isAlfabetoTabActive || !modelAlfabeto || !videoAlfabeto || videoAlfabeto.paused || videoAlfabeto.readyState < 2) {
        if (isAlfabetoTabActive) drawOverlayAlfabeto();
        requestAnimationFrame(predictLoopAlfabeto);
        return;
    }

    const [letter, confidence] = tf.tidy(() => {
        const frame = tf.browser.fromPixels(videoAlfabeto);
        const h = frame.shape[0];
        const w = frame.shape[1];

        const y1 = boxY / h;
        const x1 = boxX / w;
        const y2 = (boxY + boxSize) / h;
        const x2 = (boxX + boxSize) / w;

        const cropped = tf.image.cropAndResize(
            frame.expandDims(0),
            [[y1, x1, y2, x2]],
            [0],
            [IMG_WIDTH_ALFABETO, IMG_HEIGHT_ALFABETO]
        );

        const scaled = cropped.div(255.0);
        const prediction = modelAlfabeto.predict(scaled);

        const probabilities = prediction.dataSync();
        const predictedIndex = prediction.argMax(-1).dataSync()[0];
        const confidence = probabilities[predictedIndex];
        const letter = CLASS_MAP_ALFABETO[predictedIndex];
        return [letter, confidence];
    });

    if (confidence > 0.7) {
        const smoothed = smoothPredictionAlfabeto(letter);
        if (resultElementAlfabeto) resultElementAlfabeto.innerText = `${smoothed}`;
    } else {
        if (resultElementAlfabeto) resultElementAlfabeto.innerText = "...";
    }

    drawOverlayAlfabeto();
    requestAnimationFrame(predictLoopAlfabeto);
}

// Desenha o quadrado (Alfabeto)
function drawOverlayAlfabeto() {
    if (!overlayCanvasAlfabeto || !overlayCtxAlfabeto || !videoAlfabeto || !videoAlfabeto.videoWidth) {
        if (overlayCtxAlfabeto) overlayCtxAlfabeto.clearRect(0, 0, overlayCanvasAlfabeto.width, overlayCanvasAlfabeto.height);
        return;
    }

    overlayCanvasAlfabeto.width = videoAlfabeto.videoWidth;
    overlayCanvasAlfabeto.height = videoAlfabeto.videoHeight;
    const w = overlayCanvasAlfabeto.width;
    const h = overlayCanvasAlfabeto.height;

    overlayCtxAlfabeto.clearRect(0, 0, w, h);

    boxSize = Math.min(w, h) * 0.5;
    boxX = (w - boxSize) / 2;
    boxY = (h - boxSize) / 2;

    overlayCtxAlfabeto.strokeStyle = "#7C3AED";
    overlayCtxAlfabeto.lineWidth = 4;
    overlayCtxAlfabeto.setLineDash([10, 10]);
    overlayCtxAlfabeto.strokeRect(boxX, boxY, boxSize, boxSize);
    overlayCtxAlfabeto.setLineDash([]);

    overlayCtxAlfabeto.font = "bold 24px 'Inter', sans-serif";
    overlayCtxAlfabeto.fillStyle = "#E0E0FF";
    overlayCtxAlfabeto.textAlign = "center";
    overlayCtxAlfabeto.textBaseline = "bottom";
    overlayCtxAlfabeto.fillText("Posicione sua mão aqui", w / 2, boxY - 10);
}


// ===================================================================
// PARTE 2: LÓGICA DE PALAVRAS (MediaPipe)
// ===================================================================

// --- Configurações (MediaPipe para Palavras) ---
let handLandmarker;
let runningMode = "VIDEO";
let video_palavras, canvas_palavras, ctx_palavras, drawingUtils;
let lastVideoTime = -1;
let results = null;
let isPalavrasInitialized = false;
let streamPalavras = null; // Para guardar o stream da câmara

// Inicialização do MediaPipe (Palavras)
async function initPalavras() {
    if (isPalavrasInitialized) return;
    const statusElem = document.getElementById("status_palavras");
    try {
        statusElem && (statusElem.innerText = "Carregando MediaPipe Vision...");

        statusElem && (statusElem.innerText = "Iniciando FilesetResolver...");

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        statusElem && (statusElem.innerText = "Carregando modelo HandLandmarker...");

        try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode,
                numHands: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7,
            });
        } catch (errGpu) {
            console.warn("Falha ao criar HandLandmarker com delegate GPU, tentando CPU...", errGpu);
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
                    delegate: "CPU"
                },
                runningMode,
                numHands: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7,
            });
        }

        isPalavrasInitialized = true;
        console.log("HandLandmarker (Palavras) inicializado com sucesso.");
        statusElem && (statusElem.classList.add('hidden'));

    } catch (err) {
        console.error("Erro em initPalavras():", err);
        isPalavrasInitialized = false;
        const statusElem = document.getElementById("status_palavras");
        if (statusElem) {
            statusElem.classList.remove('hidden');
            statusElem.innerText = "Erro ao carregar MediaPipe (ver console). Verifique rede/CDN ou tente recarregar a página.";
        }
        throw err;
    }
}

// Inicia a webcam (Palavras)
async function setupCameraPalavras() {
    if (!video_palavras) {
        console.error("Elemento de vídeo 'webcam_palavras' não encontrado.");
        return;
    }
    const statusElem = document.getElementById("status_palavras");
    const contentElem = document.getElementById("palavras-content");

    try {
        video_palavras.onplaying = () => {
            canvas_palavras.width = video_palavras.videoWidth;
            canvas_palavras.height = video_palavras.videoHeight;

            statusElem.classList.add("hidden");
            contentElem.classList.remove("hidden");

            predictLoopPalavras();
        };

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: false
        });
        streamPalavras = stream; // Guarda o stream para parar depois
        video_palavras.srcObject = streamPalavras;
        await video_palavras.play();

    } catch (err) {
        console.error("Erro ao acessar à webcam (Palavras): ", err);
        statusElem.innerText = 'Erro ao acessar à webcam.';
    }
}

// Loop principal (Palavras)
async function predictLoopPalavras() {
    if (!handLandmarker || !video_palavras || video_palavras.paused || video_palavras.readyState < 2) {
        requestAnimationFrame(predictLoopPalavras);
        return;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video_palavras.currentTime) {
        lastVideoTime = video_palavras.currentTime;
        results = handLandmarker.detectForVideo(video_palavras, startTimeMs);
    }

    drawResultsPalavras(results);
    requestAnimationFrame(predictLoopPalavras);
}

// Desenha a saída no canvas (Palavras)
function drawResultsPalavras(results) {
    if (!results || !ctx_palavras || !drawingUtils) return;

    ctx_palavras.save();
    ctx_palavras.clearRect(0, 0, canvas_palavras.width, canvas_palavras.height);
    ctx_palavras.drawImage(video_palavras, 0, 0, canvas_palavras.width, canvas_palavras.height);

    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(
                landmarks,
                HandLandmarker.HAND_CONNECTIONS,
                { color: "#FFFFFF", lineWidth: 3 }
            );
            drawingUtils.drawLandmarks(landmarks, {
                color: "#7C3AED",
                fillColor: "#C4B5FD",
                lineWidth: 1,
                radius: 4,
            });
            drawBoundingBoxPalavras(landmarks);
            recognizeGestureWord(landmarks);
        }
    }
    ctx_palavras.restore();
}

// Desenha o quadrado (Palavras)
function drawBoundingBoxPalavras(landmarks) {
    const width = canvas_palavras.width;
    const height = canvas_palavras.height;
    let minX = width, minY = height, maxX = 0, maxY = 0;

    for (const point of landmarks) {
        const x = point.x * width;
        const y = point.y * height;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    const padding = 20;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(width, maxX + padding);
    maxY = Math.min(height, maxY + padding);
    const w = maxX - minX;
    const h = maxY - minY;

    ctx_palavras.strokeStyle = "#00FF00";
    ctx_palavras.lineWidth = 4;
    ctx_palavras.strokeRect(minX, minY, w, h);
}

// Reconhece o gesto/palavra
function recognizeGestureWord(landmarks) {
    const resultadoElem = document.getElementById("result_palavras");
    if (!resultadoElem) return;

    const getDistance = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

    const thumbTip = landmarks[4], indexTip = landmarks[8],
        middleTip = landmarks[12], ringTip = landmarks[16],
        pinkyTip = landmarks[20], wrist = landmarks[0],
        indexPip = landmarks[6]; 

    const isThumbUp = thumbTip.y < landmarks[2].y;
    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleUp = middleTip.y < landmarks[10].y;
    const isRingUp = ringTip.y < landmarks[14].y;
    const isPinkyUp = pinkyTip.y < landmarks[18].y;

    const countFingersUp = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

    // "Pensar"
    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && indexTip.y < 0.25) {
        return resultadoElem.innerText = "Pensar";
    }

    // "Aqui"
    const isIndexPointingDown = indexTip.y > indexPip.y + 0.05;
    const isIndexVertical = Math.abs(indexTip.x - indexPip.x) < 0.08;
    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && isIndexPointingDown && isIndexVertical) {
         return resultadoElem.innerText = "Aqui";
    }

    if (isThumbUp && isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = "Eu te amo";

    if (isThumbUp && !isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = "Telefone / Agora";

    if (!isThumbUp && isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = "Rock";
    if (isMiddleUp && isRingUp && isPinkyUp && getDistance(thumbTip, indexTip) < 0.07)
        return resultadoElem.innerText = "OK";
    if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp)
        return resultadoElem.innerText = "Paz";
    if (isThumbUp && countFingersUp === 0)
        return resultadoElem.innerText = "Legal / Bom";
    if (thumbTip.y > landmarks[2].y && countFingersUp === 0)
        return resultadoElem.innerText = "Ruim";
    if (countFingersUp === 4 && isThumbUp)
        return resultadoElem.innerText = "Oi";

    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp)
        return resultadoElem.innerText = (indexTip.x > wrist.x) ? "Eu" : "Você / Tu"; 

    if (countFingersUp === 0 && getDistance(thumbTip, indexTip) < 0.06)
        return resultadoElem.innerText = "Comer";
    if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && getDistance(thumbTip, indexTip) > 0.1)
        return resultadoElem.innerText = "Beber";
    if (isIndexUp && isMiddleUp && getDistance(thumbTip, indexTip) < 0.05)
        return resultadoElem.innerText = "Dinheiro";

    resultadoElem.innerText = "...";
}

// ===================================================================
// PARTE 3: CONTROLO DE ABAS
// ===================================================================

function stopAllStreams() {
    if (streamAlfabeto) {
        streamAlfabeto.getTracks().forEach(track => track.stop());
        streamAlfabeto = null;
    }
    if (streamPalavras) {
        streamPalavras.getTracks().forEach(track => track.stop());
        streamPalavras = null;
    }

    if (videoAlfabeto) {
        videoAlfabeto.pause();
        videoAlfabeto.srcObject = null;
        videoAlfabeto.onplaying = null;
    }
    if (video_palavras) {
        video_palavras.pause();
        video_palavras.srcObject = null;
        video_palavras.onplaying = null;
    }
    isAlfabetoTabActive = false;
}

window.openTab = async function (tabId) {
    document.querySelectorAll(".tab-content").forEach((el) => {
        el.style.display = "none";
        el.classList.remove("active");
    });
    document.querySelectorAll(".tab-button").forEach((el) => el.classList.remove("active"));

    document.getElementById(tabId).style.display = "block";
    document.getElementById(tabId).classList.add("active");

    const buttonId = `btn-${tabId.replace('introducao', 'intro')}`;
    document.getElementById(buttonId)?.classList.add("active");

    stopAllStreams();

    if (tabId === "palavras") {
        if (!isPalavrasInitialized) {
            await initPalavras();
        }
        await setupCameraPalavras();

    } else if (tabId === "alfabeto") {
        if (!isAlfabetoInitialized) {
            await initAlfabeto();
        }
        await startWebcamAlfabeto();
    }
};


// ===================================================================
// PARTE 4: LÓGICA DO ASSISTENTE IA GENERATIVA (VERSÃO SEGURA PARA VERCEL)
// ===================================================================

let chatMessages, chatInput, chatSend, chatHistory;

// Adiciona uma mensagem à janela de chat
function addMessageToChat(sender, message) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender === "user" ? "user-message" : "ai-message");

    let htmlMessage = message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');

    messageElement.innerHTML = htmlMessage;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Mostra o indicador de "a escrever..."
function showLoadingIndicator() {
    const loadingElement = document.createElement("div");
    loadingElement.id = "chat-loading";
    loadingElement.classList.add("chat-message");
    loadingElement.innerHTML = '<div class="dot-flashing"></div>';
    chatMessages.appendChild(loadingElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove o indicador de "a escrever..."
function removeLoadingIndicator() {
    const loadingElement = document.getElementById("chat-loading");
    if (loadingElement) {
        chatMessages.removeChild(loadingElement);
    }
}

// Lida com o envio da mensagem
async function handleChatSubmit() {
    const prompt = chatInput.value.trim();
    if (!prompt) return;

    addMessageToChat("user", prompt);
    chatInput.value = "";
    showLoadingIndicator();

    // Atualiza o histórico para enviar à API
    chatHistory.push({ role: "user", parts: [{ text: prompt }] });

    try {
        // MODIFICADO: Chama a NOSSA API em /api/chat
        const aiResponse = await callGeminiAPI(chatHistory); // Envia o histórico
        removeLoadingIndicator();
        addMessageToChat("ai", aiResponse);
        // Adiciona a resposta da IA ao histórico
        chatHistory.push({ role: "model", parts: [{ text: aiResponse }] });

    } catch (error) {
        console.error("Erro ao chamar a API local:", error); // Mensagem de erro atualizada
        removeLoadingIndicator();
        // Usa a mensagem de erro tratada da callGeminiAPI
        addMessageToChat("ai", `Desculpe, ocorreu um erro ao conectar-me à IA. (Erro: ${error.message})`);
    }
}

// Chama a API do Google Gemini (AGORA CHAMA O NOSSO BACKEND)
async function callGeminiAPI(history) { // Modificado: recebe o histórico
    // A URL AGORA APONTA PARA A NOSSA FUNÇÃO DE BACKEND NA VERCEL
    const API_URL = "/api/chat"; // Caminho relativo para a Serverless Function

    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Envia apenas o histórico
        body: JSON.stringify({ history: history })
    });

    if (!response.ok) {
        let errorText = "Erro desconhecido do servidor";
        try {
            // Tenta ler a resposta de erro como JSON (o que o nosso api/chat.js envia)
            const errorData = await response.json();
            errorText = errorData.error || JSON.stringify(errorData);
        } catch (jsonError) {
            // Se falhar (como agora), lê como texto puro (o que a Vercel envia)
            errorText = await response.text();
        }
        // Lança um erro para ser apanhado pelo bloco catch em handleChatSubmit
        throw new Error(errorText);
    }

    const data = await response.json();
    return data.text; // O backend agora nos envia o texto diretamente
}


// ===================================================================
// PONTO DE ENTRADA (DOMContentLoaded)
// ===================================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- Define elementos do Alfabeto (Parte 1) ---
    videoAlfabeto = document.getElementById('webcam');
    statusElementAlfabeto = document.getElementById('status');
    resultElementAlfabeto = document.getElementById('result');
    overlayCanvasAlfabeto = document.getElementById('overlay');
    overlayCtxAlfabeto = overlayCanvasAlfabeto.getContext('2d');
    alfabetoContent = document.getElementById('alfabeto-content');

    // --- Define elementos do MediaPipe (Parte 2 - Palavras) ---
    video_palavras = document.getElementById('webcam_palavras');
    canvas_palavras = document.getElementById('output_canvas_palavras');
    ctx_palavras = canvas_palavras?.getContext('2d');
    drawingUtils = new DrawingUtils(ctx_palavras);

    // --- Define elementos do Assistente IA (Parte 4) ---
    chatMessages = document.getElementById('chat-messages');
    chatInput = document.getElementById('chat-input');
    chatSend = document.getElementById('chat-send');
    chatHistory = [];

    // Botões de navegação
    document.getElementById('btn-intro')?.addEventListener('click', () => window.openTab('introducao'));
    document.getElementById('btn-alfabeto')?.addEventListener('click', () => window.openTab('alfabeto'));
    document.getElementById('btn-palavras')?.addEventListener('click', () => window.openTab('palavras'));
    document.getElementById('btn-assistente')?.addEventListener('click', () => window.openTab('assistente'));

    chatSend?.addEventListener('click', handleChatSubmit);
    chatInput?.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') handleChatSubmit();
    });

    window.openTab('introducao');
});