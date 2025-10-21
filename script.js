import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let handLandmarker;
let runningMode = "VIDEO";
let video_alfabeto, video_palavras;
let canvas_alfabeto, canvas_palavras;
let ctx_alfabeto, ctx_palavras;
let lastVideoTime = -1;
let currentTab = 'introducao';

// Suavização dos pontos da mão
let previousLandmarks = [];
function smoothLandmarks(newLandmarks) {
    const SMOOTHING_FACTOR = 0.6; // quanto maior, mais lento e estável
    if (previousLandmarks.length === 0) {
        previousLandmarks = newLandmarks.map(p => ({ ...p }));
        return newLandmarks;
    }
    return newLandmarks.map((p, i) => ({
        x: previousLandmarks[i].x * SMOOTHING_FACTOR + p.x * (1 - SMOOTHING_FACTOR),
        y: previousLandmarks[i].y * SMOOTHING_FACTOR + p.y * (1 - SMOOTHING_FACTOR),
        z: previousLandmarks[i].z * SMOOTHING_FACTOR + p.z * (1 - SMOOTHING_FACTOR)
    }));
}

// Suavização da letra reconhecida
let lastDetected = '?';
let stableCount = 0;
function updateResult(elem, newLetter) {
    if (newLetter === lastDetected) {
        stableCount++;
        if (stableCount > 5) { // precisa aparecer igual por 5 frames
            elem.innerText = newLetter;
        }
    } else {
        lastDetected = newLetter;
        stableCount = 0;
    }
}

// Criação do modelo
async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
}

// Configura a câmera
function setupCamera(tabId) {
    const video = document.getElementById(`webcam_${tabId}`);
    const canvas = document.getElementById(`output_canvas_${tabId}`);
    const ctx = canvas.getContext("2d");

    if (tabId === 'alfabeto') {
        video_alfabeto = video;
        canvas_alfabeto = canvas;
        ctx_alfabeto = ctx;
    } else {
        video_palavras = video;
        canvas_palavras = canvas;
        ctx_palavras = ctx;
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                document.getElementById(`${tabId}-loader`).style.display = 'none';
                document.getElementById(`${tabId}-content`).classList.remove('hidden');
                if (tabId === 'alfabeto') predictAlphabet();
                if (tabId === 'palavras') predictWord();
            });
        }).catch(error => {
            console.error("Erro ao acessar a câmera: ", error);
            document.getElementById(`${tabId}-loader`).innerText = 'Erro ao acessar a câmera.';
        });
    } else {
        document.getElementById(`${tabId}-loader`).innerText = 'Seu navegador não suporta acesso à câmera.';
    }
}

// Predição do alfabeto
async function predictAlphabet() {
    if (currentTab !== 'alfabeto' || !video_alfabeto || !video_alfabeto.srcObject) {
        window.requestAnimationFrame(predictAlphabet);
        return;
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== video_alfabeto.currentTime) {
        lastVideoTime = video_alfabeto.currentTime;
        const results = handLandmarker.detectForVideo(video_alfabeto, startTimeMs);

        ctx_alfabeto.save();
        ctx_alfabeto.clearRect(0, 0, canvas_alfabeto.width, canvas_alfabeto.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const smoothed = smoothLandmarks(results.landmarks[0]);
            drawLandmarks(ctx_alfabeto, smoothed);
            recognizeGestureAlphabet(smoothed);
        } else {
            document.getElementById('resultado_alfabeto').innerText = '...';
        }

        ctx_alfabeto.restore();
    }

    window.requestAnimationFrame(predictAlphabet);
}

// Predição das palavras
async function predictWord() {
    if (currentTab !== 'palavras' || !video_palavras || !video_palavras.srcObject) {
        window.requestAnimationFrame(predictWord);
        return;
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== video_palavras.currentTime) {
        lastVideoTime = video_palavras.currentTime;
        const results = handLandmarker.detectForVideo(video_palavras, startTimeMs);

        ctx_palavras.save();
        ctx_palavras.clearRect(0, 0, canvas_palavras.width, canvas_palavras.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const smoothed = smoothLandmarks(results.landmarks[0]);
            drawLandmarks(ctx_palavras, smoothed);
            recognizeGestureWord(smoothed);
        } else {
            document.getElementById('resultado_palavras').innerText = '...';
        }

        ctx_palavras.restore();
    }

    window.requestAnimationFrame(predictWord);
}

// Desenho dos pontos
function drawLandmarks(ctx, landmarks) {
    const connect = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]];
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#7C3AED';
    ctx.fillStyle = '#A78BFA';

    for (const connection of connect) {
        const [a, b] = connection;
        const p1 = landmarks[a], p2 = landmarks[b];
        ctx.beginPath();
        ctx.moveTo(p1.x * ctx.canvas.width, p1.y * ctx.canvas.height);
        ctx.lineTo(p2.x * ctx.canvas.width, p2.y * ctx.canvas.height);
        ctx.stroke();
    }

    for (const point of landmarks) {
        ctx.beginPath();
        ctx.arc(point.x * ctx.canvas.width, point.y * ctx.canvas.height, 5, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Reconhecimento de gestos do alfabeto (melhorado)
function recognizeGestureAlphabet(landmarks) {
    const resultadoElem = document.getElementById('resultado_alfabeto');
    const getDistance = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
    const normDist = (d) => d / getDistance(landmarks[0], landmarks[9]); // normaliza pelo tamanho da mão

    const thumbTip = landmarks[4], indexTip = landmarks[8], middleTip = landmarks[12],
          ringTip = landmarks[16], pinkyTip = landmarks[20],
          indexMcp = landmarks[5], middleMcp = landmarks[9],
          ringMcp = landmarks[13], wrist = landmarks[0];

    const isThumbUp = thumbTip.y < landmarks[3].y;
    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleUp = middleTip.y < landmarks[10].y;
    const isRingUp = ringTip.y < landmarks[14].y;
    const isPinkyUp = pinkyTip.y < landmarks[18].y;
    const countFingersUp = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

    // Exemplos com distâncias normalizadas
    if (isThumbUp && isPinkyUp && !isIndexUp && !isMiddleUp && !isRingUp)
        return updateResult(resultadoElem, 'Y');
    if (isPinkyUp && !isIndexUp && !isMiddleUp && !isRingUp)
        return updateResult(resultadoElem, 'I');
    if (isMiddleUp && isRingUp && isPinkyUp && normDist(getDistance(thumbTip, indexTip)) < 0.3)
        return updateResult(resultadoElem, 'F');
    if (isIndexUp && isMiddleUp && isRingUp && !isPinkyUp)
        return updateResult(resultadoElem, 'W');
    if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp)
        return updateResult(resultadoElem, 'V');
    if (isThumbUp && isIndexUp && countFingersUp === 1)
        return updateResult(resultadoElem, 'L');
    if (countFingersUp === 4)
        return updateResult(resultadoElem, 'B');

    // Todos abaixados
    if (countFingersUp === 0) {
        if (thumbTip.y > indexMcp.y) return updateResult(resultadoElem, 'A');
        if (normDist(getDistance(thumbTip, indexTip)) < 0.3) return updateResult(resultadoElem, 'O');
    }

    return updateResult(resultadoElem, '?');
}

// Reconhecimento de gestos de palavras (mesmo do seu código)
function recognizeGestureWord(landmarks) {
    const resultadoElem = document.getElementById('resultado_palavras');
    const getDistance = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
    const thumbTip = landmarks[4], indexTip = landmarks[8],
          middleTip = landmarks[12], ringTip = landmarks[16], pinkyTip = landmarks[20];
    const wrist = landmarks[0];

    const isThumbUp = thumbTip.y < landmarks[2].y;
    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleUp = middleTip.y < landmarks[10].y;
    const isRingUp = ringTip.y < landmarks[14].y;
    const isPinkyUp = pinkyTip.y < landmarks[18].y;
    const countFingersUp = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

    if (isThumbUp && isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = 'Eu te amo';
    if (isThumbUp && !isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = 'Telefone';
    if (!isThumbUp && isIndexUp && !isMiddleUp && !isRingUp && isPinkyUp)
        return resultadoElem.innerText = 'Rock';
    if (isMiddleUp && isRingUp && isPinkyUp && getDistance(thumbTip, indexTip) < 0.07)
        return resultadoElem.innerText = 'OK';
    if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp)
        return resultadoElem.innerText = 'Paz';
    if (isThumbUp && countFingersUp === 0)
        return resultadoElem.innerText = 'Legal / Bom';
    if (thumbTip.y > landmarks[2].y && countFingersUp === 0)
        return resultadoElem.innerText = 'Ruim';
    if (countFingersUp === 4 && isThumbUp)
        return resultadoElem.innerText = 'Oi / Parar';
    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp)
        return resultadoElem.innerText = (indexTip.x < wrist.x) ? 'Você' : 'Eu';
    if (countFingersUp === 0 && getDistance(thumbTip, indexTip) < 0.06)
        return resultadoElem.innerText = 'Comer';
    if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && getDistance(thumbTip, indexTip) > 0.1)
        return resultadoElem.innerText = 'Beber';
    if (isIndexUp && isMiddleUp && getDistance(thumbTip, indexTip) < 0.05)
        return resultadoElem.innerText = 'Dinheiro';

    resultadoElem.innerText = '?';
}

// Controle de abas
window.openTab = function (tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`button[onclick="openTab('${tabId}')"]`).classList.add('active');

    const manageStream = (video, enable) => {
        if (video && video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.enabled = enable);
            enable ? video.play() : video.pause();
        }
    };
    manageStream(video_alfabeto, tabId === 'alfabeto');
    manageStream(video_palavras, tabId === 'palavras');

    if (tabId === 'alfabeto' && !video_alfabeto) setupCamera('alfabeto');
    if (tabId === 'palavras' && !video_palavras) setupCamera('palavras');
};

// Inicialização
async function runDemo() {
    await createHandLandmarker();
    openTab('introducao');
}
runDemo();
