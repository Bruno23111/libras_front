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

// CORREÇÃO: Movida para o topo da PARTE 1
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
        // CORREÇÃO: Tenta usar 'webgl' e, se falhar, usa 'cpu'
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
        // 'videoAlfabeto' é definido no DOMContentLoaded
        
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
        await videoAlfabeto.play(); // Tenta dar play

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
        // CORREÇÃO: Chamada agora é válida
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

// CORREÇÃO: Esta é a nova função que ESPERA pelo bundle
function waitForVisionBundle() {
    // Retorna uma Promessa que só resolve quando o 'vision' estiver pronto
    return new Promise((resolve) => {
        function check() {
            if (window.tasks && window.tasks.vision) {
                console.log("MediaPipe Vision bundle CARREGADO.");
                resolve();
            } else {
                console.log("A aguardar pelo MediaPipe Vision bundle...");
                setTimeout(check, 100); // Tenta novamente em 100ms
            }
        }
        check();
    });
}

// Inicialização do MediaPipe (Palavras)
async function initPalavras() {
    if (isPalavrasInitialized) return;
    
    // CORREÇÃO: Agora 'initPalavras' espera (await) pelo bundle
    await waitForVisionBundle();
    
    // O bundle está carregado, podemos desestruturar com segurança
    const { HandLandmarker, FilesetResolver, DrawingUtils } = window.tasks.vision;

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

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

    isPalavrasInitialized = true;
    console.log("HandLandmarker (Palavras) inicializado.");
}

// Configura câmera (Palavras)
async function setupCameraPalavras() {
    video_palavras = document.getElementById("webcam_palavras");
    canvas_palavras = document.getElementById("output_canvas_palavras");
    ctx_palavras = canvas_palavras.getContext("2d");
    
    // CORREÇÃO: Esta verificação agora é segura, pois 'initPalavras' esperou
    if (window.tasks && window.tasks.vision) {
        const { DrawingUtils } = window.tasks.vision;
        drawingUtils = new DrawingUtils(ctx_palavras);
    } else {
        // Este erro não deve mais acontecer
        console.error("Vision bundle não carregado, DrawingUtils indisponível.");
        return; 
    }

    try {
        video_palavras.onplaying = () => {
            canvas_palavras.width = video_palavras.videoWidth;
            canvas_palavras.height = video_palavras.videoHeight;

            document.getElementById("status_palavras").classList.add("hidden");
            document.getElementById("palavras-content").classList.remove("hidden");

            predictLoopPalavras();
        };

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 1280, height: 720 },
        });
        streamPalavras = stream; // Guarda o stream
        video_palavras.srcObject = streamPalavras;
        await video_palavras.play(); // Tenta dar play

    } catch (err) {
        console.error("Erro ao acessar câmera (Palavras):", err);
        document.getElementById("status_palavras").innerText = "Erro ao acessar câmera.";
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

// Reconhecimento de gestos (Palavras)
function recognizeGestureWord(landmarks) {
    const resultadoElem = document.getElementById("result_palavras");
    if (!resultadoElem) return;

    const getDistance = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

    const thumbTip = landmarks[4], indexTip = landmarks[8],
        middleTip = landmarks[12], ringTip = landmarks[16],
        pinkyTip = landmarks[20], wrist = landmarks[0];

    const isThumbUp = thumbTip.y < landmarks[2].y;
    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleUp = middleTip.y < landmarks[10].y;
    const isRingUp = ringTip.y < landmarks[14].y;
    const isPinkyUp = pinkyTip.y < landmarks[18].y;

    const countFingersUp = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

    if (isThumbUp && isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = "Eu te amo";
    if (isThumbUp && !isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = "Telefone";
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
        return resultadoElem.innerText = (indexTip.x > wrist.x) ? "Eu" : "Você";
    if (countFingersUp === 0 && getDistance(thumbTip, indexTip) < 0.06)
        return resultadoElem.innerText = "Comer";
    if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && getDistance(thumbTip, indexTip) > 0.1)
        return resultadoElem.innerText = "Beber";
    if (isIndexUp && isMiddleUp && getDistance(thumbTip, indexTip) < 0.05)
        return resultadoElem.innerText = "Dinheiro";

    resultadoElem.innerText = "...";
}


// ===================================================================
// PARTE 3: CONTROLO DE ABAS (CORRIGIDO)
// ===================================================================

// CORREÇÃO: Nova função para PARAR todos os streams (desligar a câmara)
function stopAllStreams() {
    if (streamAlfabeto) {
        streamAlfabeto.getTracks().forEach(track => track.stop());
        streamAlfabeto = null;
    }
    if (streamPalavras) {
        streamPalavras.getTracks().forEach(track => track.stop());
        streamPalavras = null;
    }
    
    // Liberta os elementos de vídeo
    if (videoAlfabeto) {
        videoAlfabeto.pause();
        videoAlfabeto.srcObject = null;
        videoAlfabeto.onplaying = null; // Remove o listener
    }
    if (video_palavras) {
        video_palavras.pause();
        video_palavras.srcObject = null;
        video_palavras.onplaying = null; // Remove o listener
    }
    isAlfabetoTabActive = false;
}

// Define openTab no 'window' para ser acessível globalmente
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

    // --- MUDANÇA: Gerenciamento das Câmeras CORRIGIDO ---

    // 1. Desliga SEMPRE todas as câmaras ativas antes de mudar de aba
    stopAllStreams();

    // 2. Liga a câmara necessária para a aba clicada
    if (tabId === "palavras") {
        if (!isPalavrasInitialized) {
            await initPalavras(); // Carrega o modelo (só 1 vez)
        }
        await setupCameraPalavras(); // Inicia a câmara

    } else if (tabId === "alfabeto") {
        if (!isAlfabetoInitialized) { 
            await initAlfabeto(); // Carrega o modelo (só 1 vez)
        }
        await startWebcamAlfabeto(); // Inicia a câmara
    }
    // Se for "Introdução" ou "Assistente", nenhuma câmara é ligada.
};


// ===================================================================
// PARTE 4: LÓGICA DO ASSISTENTE IA GENERATIVA
// ===================================================================

// --- ATENÇÃO: COLOQUE A SUA CHAVE DA API AQUI ---
const API_KEY = "COLE_SUA_CHAVE_DA_API_AQUI"; 
const GEN_MODEL = "gemini-2.5-flash-preview-09-2025"; 

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

    if (API_KEY === "COLE_SUA_CHAVE_DA_API_AQUI") {
        addMessageToChat("ai", "Erro: A chave da API (API_KEY) não foi definida no `script.js`. Por favor, obtenha uma chave no Google AI Studio e cole-a no ficheiro.");
        return;
    }

    addMessageToChat("user", prompt);
    chatInput.value = ""; 
    showLoadingIndicator(); 

    // Atualiza o histórico para enviar à API
    chatHistory.push({ role: "user", parts: [{ text: prompt }] });

    try {
        const aiResponse = await callGeminiAPI(prompt);
        removeLoadingIndicator(); 
        addMessageToChat("ai", aiResponse); 

        // Adiciona a resposta da IA ao histórico
        chatHistory.push({ role: "model", parts: [{ text: aiResponse }] });

    } catch (error) {
        console.error("Erro ao chamar a API Gemini:", error);
        removeLoadingIndicator();
        addMessageToChat("ai", "Desculpe, ocorreu um erro ao conectar-me à IA. Por favor, tente novamente. (Verifique a consola para mais detalhes)");
    }
}

// Chama a API do Google Gemini
async function callGeminiAPI(prompt) {
    const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEN_MODEL}:generateContent?key=${API_KEY}`;

    const systemInstruction = {
        role: "system",
        parts: [{ text: "Você é o 'Libras.IO', um assistente de IA amigável, especialista e entusiasta da Língua Brasileira de Sinais (Libras). Sua missão é ajudar estudantes a aprender, tirando dúvidas sobre gramática, história, cultura Surda e os sinais. Seja didático, encorajador e use formatação Markdown (como **negrito**) para destacar termos importantes. Responda em português do Brasil." }]
    };

    const requestBody = {
        systemInstruction: systemInstruction,
        contents: [...chatHistory],
        safetySettings: [
            { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
            { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
            { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
            { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
        ],
        generationConfig: {
            temperature: 0.7,
            topK: 1,
            topP: 1,
        }
    };

    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error ${response.status}: ${errorData.error.message}`);
    }

    const data = await response.json();
    
    if (data.candidates && data.candidates.length > 0) {
        // Verifica se a resposta não foi bloqueada
        if (data.candidates[0].content && data.candidates[0].content.parts) {
            const text = data.candidates[0].content.parts[0].text;
            return text;
        } else if (data.candidates[0].finishReason === 'SAFETY') {
            return "Desculpe, não posso responder a essa pergunta pois ela viola as minhas diretrizes de segurança.";
        }
    }
    
    return "Não consegui gerar uma resposta. Tente novamente.";
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
    
    // --- Define elementos do Assistente IA (Parte 4) ---
    // CORREÇÃO: Adiciona '?' para evitar erros se os elementos não existirem
    // (O seu index.html mais recente não tem a aba 'assistente')
    chatMessages = document.getElementById('chat-messages');
    chatInput = document.getElementById('chat-input');
    chatSend = document.getElementById('chat-send');
    chatHistory = []; // Inicia o histórico do chat

    // --- Adiciona os cliques aos botões (Parte 3 e 4) ---
    document.getElementById('btn-intro')?.addEventListener('click', () => window.openTab('introducao'));
    document.getElementById('btn-alfabeto')?.addEventListener('click', () => window.openTab('alfabeto'));
    document.getElementById('btn-palavras')?.addEventListener('click', () => window.openTab('palavras'));
    document.getElementById('btn-assistente')?.addEventListener('click', () => window.openTab('assistente')); 

    // Adiciona listener para o botão de Enviar do chat
    chatSend?.addEventListener('click', handleChatSubmit);
    // Adiciona listener para a tecla "Enter" no input do chat
    chatInput?.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleChatSubmit();
        }
    });

    // Abre a aba "Introdução" por padrão
    window.openTab('introducao');
});

