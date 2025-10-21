// --- Configurações Essenciais ---
const MODEL_PATH = './modelos_web/model.json';
const IMG_WIDTH = 128;
const IMG_HEIGHT = 128;

const CLASS_MAP = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'I', // Pula H
    8: 'L', // Pula J, K
    9: 'M',
    10: 'N',
    11: 'O',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'U',
    18: 'V',
    19: 'W',
    20: 'Y' // Pula X, Z
};
// ---------------------------------

// --- Variáveis Globais para a IA do Alfabeto ---
let modelAlfabeto; // Renomeado para não conflitar
let videoAlfabeto;
let resultAlfabeto;
let loaderAlfabeto;
let contentAlfabeto;
let isAlfabetoTabActive = false; // Controle para saber se a aba está visível

/**
 * Carrega o modelo TensorFlow.js
 */
async function loadModel() {
    console.log("Carregando modelo TF.js (Alfabeto)...");
    try {
        modelAlfabeto = await tf.loadLayersModel(MODEL_PATH);
        // Aquece o modelo
        modelAlfabeto.predict(tf.zeros([1, IMG_WIDTH, IMG_HEIGHT, 3])).dispose();
        console.log("Modelo TF.js (Alfabeto) carregado com sucesso!");
        return true; // Retorna sucesso
    } catch (err) {
        console.error("Erro ao carregar o modelo TF.js:", err);
        return false; // Retorna falha
    }
}

/**
 * Inicia a webcam do Alfabeto
 */
async function setupWebcamAlfabeto() {
    console.log("Iniciando webcam (Alfabeto)...");
    videoAlfabeto = document.getElementById("webcam_alfabeto");
    if (!videoAlfabeto) {
        console.error("Elemento de vídeo #webcam_alfabeto não encontrado!");
        return false;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: IMG_WIDTH,
                height: IMG_HEIGHT,
                facingMode: 'user' 
            }
        });
        videoAlfabeto.srcObject = stream;
        videoAlfabeto.width = IMG_WIDTH;
        videoAlfabeto.height = IMG_HEIGHT;
        
        await new Promise(resolve => videoAlfabeto.onloadedmetadata = resolve);
        console.log("Webcam (Alfabeto) iniciada.");
        return true; // Retorna sucesso

    } catch (err) {
        console.error("Erro ao iniciar webcam (Alfabeto):", err);
        return false; // Retorna falha
    }
}

/**
 * Processa a imagem da webcam para o formato que o modelo espera
 */
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageElement);
        const resized = tf.image.resizeBilinear(tensor, [IMG_WIDTH, IMG_HEIGHT]);
        const normalized = resized.toFloat().div(tf.scalar(255.0));
        const batched = normalized.expandDims(0);
        return batched;
    });
}

/**
 * Loop de Previsão em Tempo Real (Alfabeto)
 */
async function predictLoopAlfabeto() {
    // Só roda a previsão se o modelo estiver pronto e a aba "alfabeto" estiver ativa
    if (modelAlfabeto && videoAlfabeto && resultAlfabeto && isAlfabetoTabActive) {
        
        // 1. Processa o frame atual
        const tensor = preprocessImage(videoAlfabeto);

        // 2. Faz a previsão
        const predictionTensor = modelAlfabeto.predict(tensor);

        // 3. Obtém os dados
        const predictionData = await predictionTensor.data();
        const predictionIndex = tf.argMax(predictionTensor, 1).dataSync()[0];
        const confidence = predictionData[predictionIndex];
        const predictedLetter = CLASS_MAP[predictionIndex];

        // 7. Exibe o resultado (com limiar de 85% de confiança)
        if (confidence > 0.85) { 
             resultAlfabeto.innerText = `${predictedLetter} (${(confidence * 100).toFixed(0)}%)`;
        } else {
             resultAlfabeto.innerText = "...";
        }
        
        // 8. Limpa os tensores da memória
        tensor.dispose();
        predictionTensor.dispose();
    }

    // Chama esta função novamente no próximo quadro
    requestAnimationFrame(predictLoopAlfabeto);
}

/**
 * Função principal de inicialização da IA do Alfabeto
 */
async function initAlfabetoIA() {
    // Pega as referências dos elementos do HTML
    loaderAlfabeto = document.getElementById("alfabeto-loader");
    contentAlfabeto = document.getElementById("alfabeto-content");
    resultAlfabeto = document.getElementById("resultado_alfabeto");
    
    if (!loaderAlfabeto || !contentAlfabeto || !resultAlfabeto) {
        console.log("Elementos da IA (Alfabeto) não encontrados. Script não será iniciado.");
        return;
    }

    // Espera o modelo carregar E a webcam iniciar
    const [modelReady, webcamReady] = await Promise.all([
        loadModel(),
        setupWebcamAlfabeto()
    ]);

    // Se ambos estiverem prontos, esconde o loader
    if (modelReady && webcamReady) {
        loaderAlfabeto.classList.add("hidden");
        contentAlfabeto.classList.remove("hidden");
        
        // Inicia o loop de previsão
        predictLoopAlfabeto();
    } else {
        loaderAlfabeto.innerHTML = '<p class="text-red-500 font-semibold">Falha ao carregar a IA. Verifique as permissões da câmera e recarregue a página.</p>';
    }
}

// Roda a função 'main' quando a página terminar de carregar
document.addEventListener("DOMContentLoaded", () => {
    
    // Inicia a IA do Alfabeto
    initAlfabetoIA();

    // Adiciona um listener para o botão da aba "alfabeto" para controlar o loop
    const alfabetoButton = document.querySelector("button[onclick=\"openTab('alfabeto')\"]");
    const palavrasButton = document.querySelector("button[onclick=\"openTab('palavras')\"]");
    const introButton = document.querySelector("button[onclick=\"openTab('introducao')\"]");

    if (alfabetoButton) {
        alfabetoButton.addEventListener('click', () => {
            isAlfabetoTabActive = true;
            if(videoAlfabeto && videoAlfabeto.srcObject) {
                 videoAlfabeto.play();
            }
        });
    }
    
    // Pausa o vídeo do alfabeto se sairmos da aba
    const stopAlfabeto = () => {
        isAlfabetoTabActive = false;
        if(videoAlfabeto && videoAlfabeto.srcObject) {
            videoAlfabeto.pause();
        }
    };

    if (palavrasButton) palavrasButton.addEventListener('click', stopAlfabeto);
    if (introButton) introButton.addEventListener('click', stopAlfabeto);
});