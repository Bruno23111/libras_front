import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let handLandmarker;
let runningMode = "VIDEO";
let video_palavras, canvas_palavras, ctx_palavras;
let lastVideoTime = -1;
let results = null;

// Inicialização
async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    },
    runningMode,
    numHands: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  setupCamera();
}

// Configura câmera
async function setupCamera() {
  video_palavras = document.getElementById("webcam_palavras");
  canvas_palavras = document.getElementById("output_canvas_palavras");
  ctx_palavras = canvas_palavras.getContext("2d");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 1280, height: 720 },
  });

  video_palavras.srcObject = stream;

  video_palavras.addEventListener("loadeddata", () => {
    document.getElementById("palavras-loader").classList.add("hidden");
    document.getElementById("palavras-content").classList.remove("hidden");

    // Ajusta tamanho do canvas ao vídeo
    canvas_palavras.width = video_palavras.videoWidth;
    canvas_palavras.height = video_palavras.videoHeight;

    predictLoop();
  });
}

// Loop principal
async function predictLoop() {
  if (!handLandmarker) {
    requestAnimationFrame(predictLoop);
    return;
  }

  let startTimeMs = performance.now();

  if (lastVideoTime !== video_palavras.currentTime) {
    lastVideoTime = video_palavras.currentTime;
    results = await handLandmarker.detectForVideo(video_palavras, startTimeMs);
  }

  drawResults(results);
  requestAnimationFrame(predictLoop);
}

// Desenha a saída no canvas
function drawResults(results) {
  if (!results || !ctx_palavras) return;

  ctx_palavras.save();

  // Espelha horizontalmente (modo selfie)
  ctx_palavras.translate(canvas_palavras.width, 0);
  ctx_palavras.scale(-1, 1);

  ctx_palavras.drawImage(video_palavras, 0, 0, canvas_palavras.width, canvas_palavras.height);

  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawLandmarksFixed(ctx_palavras, landmarks);
      recognizeGestureWord(landmarks); // 🔹 Reconhecimento de palavras
    }
  }

  ctx_palavras.restore();
}

// Corrige a inversão dos eixos para os landmarks
function drawLandmarksFixed(ctx, landmarks) {
  ctx.fillStyle = "#8B5CF6";
  ctx.strokeStyle = "#C084FC";
  ctx.lineWidth = 2;

  const height = canvas_palavras.height;
  const width = canvas_palavras.width;

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ];

  for (const [a, b] of connections) {
    const p1 = landmarks[a];
    const p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }

  for (const landmark of landmarks) {
    const x = landmark.x * width;
    const y = landmark.y * height;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// 🔹 Reconhecimento de gestos → palavras
function recognizeGestureWord(landmarks) {
  const resultadoElem = document.getElementById("resultado_palavras");
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

  // 🔸 Regras de gestos
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
    return resultadoElem.innerText = (indexTip.x < wrist.x) ? "Você" : "Eu";
  if (countFingersUp === 0 && getDistance(thumbTip, indexTip) < 0.06)
    return resultadoElem.innerText = "Comer";
  if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && getDistance(thumbTip, indexTip) > 0.1)
    return resultadoElem.innerText = "Beber";
  if (isIndexUp && isMiddleUp && getDistance(thumbTip, indexTip) < 0.05)
    return resultadoElem.innerText = "Dinheiro";

  resultadoElem.innerText = "...";
}

// Troca de abas
window.openTab = function (tabId) {
  document.querySelectorAll(".tab-content").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".tab-button").forEach((el) => el.classList.remove("active"));

  document.getElementById(tabId).classList.add("active");
  document.querySelector(`[onclick="openTab('${tabId}')"]`).classList.add("active");

  if (tabId === "palavras" && !video_palavras) init();
};
