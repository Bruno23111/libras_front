
-----

# Libras.IO - Tradutor de Libras Interativo

Um projeto web que combina Visão Computacional (TensorFlow.js e MediaPipe) e IA Generativa (Google Gemini) para criar uma ferramenta completa de aprendizado e tradução da Língua Brasileira de Sinais (Libras).

-----

## 📚 Sobre o Projeto

O **Libras.IO** foi criado como uma ponte de comunicação acessível, utilizando o poder de três diferentes tipos de Inteligência Artificial diretamente no navegador. O objetivo é fornecer uma ferramenta de aprendizado interativa onde os usuários podem praticar o alfabeto, aprender palavras e tirar dúvidas sobre a cultura Surda em um só lugar.

A aplicação é dividida em quatro abas principais:

1.  **Introdução:** Apresenta o projeto e seus objetivos.
2.  **Alfabeto:** Utiliza TensorFlow.js para classificar sinais estáticos do alfabeto de Libras em tempo real.
3.  **Palavras:** Utiliza MediaPipe HandLandmarker para reconhecer gestos e palavras simples.
4.  **Assistente IA:** Um chatbot com Google Gemini para responder perguntas sobre gramática, história e cultura Surda.

## ✨ Funcionalidades Principais

  * **Tradução do Alfabeto:** Classificação de imagem em tempo real (20 letras do alfabeto) usando uma webcam e TensorFlow.js.
  * **Tradução de Palavras:** Reconhecimento de gestos e palavras simples (como "Oi", "Comer", "Pensar") usando MediaPipe.
  * **Assistente Generativo:** Um chatbot inteligente para tirar dúvidas contextuais sobre Libras, alimentado pela API Gemini.
  * **Interface Reativa:** Uma interface de usuário limpa, responsiva (mobile-first) e moderna construída com Tailwind CSS.
  * **Acessibilidade:** Integração com o widget **VLibras** para tradução de texto-para-Libras (Avatar 3D) em toda a página.

## 🛠️ Tecnologias Utilizadas

  * **Frontend:** HTML5, CSS3, JavaScript (ES6 Modules)
  * **Estilização:** [Tailwind CSS](https://tailwindcss.com/) (via CDN)
  * **Fontes:** [Google Fonts](https://fonts.google.com/) (Inter)
  * **IA (Alfabeto):** [TensorFlow.js](https://www.tensorflow.org/js) (`tf.loadLayersModel`)
  * **IA (Palavras):** [MediaPipe](https://developers.google.com/mediapipe) (`HandLandmarker`)
  * **IA (Assistente):** [Google Gemini API](https://ai.google.dev/) (modelo `gemini-2.5-flash-preview-09-2025`)
  * **Acessibilidade:** [VLibras Widget](https://www.google.com/search?q=https://www.gov.br/vlibras/)

-----

## 🧠 Arquitetura e Funcionamento Técnico

O projeto é modularizado em `script.js` e se baseia em três pilares de IA independentes que são ativados conforme a navegação do usuário.

### 1\. Aba "Alfabeto" (TensorFlow.js - Classificação)

Esta aba usa um modelo de classificação de imagem treinado (presumivelmente em Keras/Python e convertido para web).

  * **Modelo:** Carregado a partir de `./modelos_web/model.json`.
  * **Tecnologia:** `tf.loadLayersModel` do TensorFlow.js.
  * **Fluxo de Execução:**
    1.  Ao abrir a aba, `initAlfabeto()` carrega o modelo e `startWebcamAlfabeto()` ativa a câmera.
    2.  A função `drawOverlayAlfabeto()` desenha uma caixa-guia tracejada (bounding box) no canvas sobre o vídeo.
    3.  O loop `predictLoopAlfabeto()` é executado a cada frame:
    4.  O frame de vídeo é capturado com `tf.browser.fromPixels`.
    5.  A imagem é "cortada" (`tf.image.cropAndResize`) para a região da caixa-guia.
    6.  A imagem cortada é redimensionada para 128x128 pixels e normalizada (dividida por 255.0).
    7.  A previsão (`modelAlfabeto.predict()`) é executada.
  * **Otimização (Suavização):** A função `smoothPredictionAlfabeto()` armazena as últimas 5 predições. Ela retorna apenas a letra que mais apareceu nessa "janela", evitando que o resultado "pisque" e tornando a UI mais estável.

### 2\. Aba "Palavras" (MediaPipe - Deteção de Gestos)

Esta aba usa o modelo `HandLandmarker` pré-treinado do Google para detecção de pontos-chave da mão (landmarks). A lógica de reconhecimento de gestos é customizada.

  * **Modelo:** `hand_landmarker.task` (carregado da CDN do Google/MediaPipe).
  * **Tecnologia:** `HandLandmarker` e `DrawingUtils` do `@mediapipe/tasks-vision`.
  * **Fluxo de Execução:**
    1.  `initPalavras()` carrega o `FilesetResolver` e cria o `HandLandmarker` (tentando usar `GPU` com fallback para `CPU`).
    2.  O loop `predictLoopPalavras()` detecta as mãos no vídeo (`handLandmarker.detectForVideo()`).
    3.  `DrawingUtils` é usado para desenhar o "esqueleto" da mão no canvas (`output_canvas_palavras`).
    4.  Os 21 *landmarks* (pontos-chave) da mão detectada são passados para a função `recognizeGestureWord()`.
    5.  Esta função customizada usa lógica baseada em posições relativas (ex: `isThumbUp`, `isIndexUp`) e distância euclidiana entre os pontos (ex: `getDistance(thumbTip, indexTip)`) para classificar o gesto em uma palavra (ex: "Legal / Bom", "Comer", "Pensar").

### 3\. Aba "Assistente IA" (Google Gemini - IA Generativa)

Esta aba fornece um chatbot para responder perguntas usando um modelo de linguagem grande (LLM).

  * **Modelo:** `gemini-2.5-flash-preview-09-2025` (via API).
  * **Tecnologia:** Chamada `fetch` direta para a API `generativelanguage.googleapis.com`.
  * **Fluxo de Execução:**
    1.  Quando o usuário envia uma mensagem (`handleChatSubmit`), ela é adicionada a um array local `chatHistory`.
    2.  Uma chamada `POST` é feita para a API.
    3.  **Contexto (System Prompt):** A requisição inclui uma `systemInstruction` que define a persona da IA: *"Você é o 'Libras.IO', um assistente de IA amigável, especialista e entusiasta da Língua Brasileira de Sinais (Libras)..."*
    4.  **Memória:** O array `chatHistory` completo é enviado no corpo da requisição (`contents`), permitindo que a IA mantenha o contexto da conversa.
    5.  A resposta de texto da IA é recebida, formatada (Markdown para HTML) e exibida no chat.

### 4\. Gerenciamento de Câmera (Ponto Crítico)

Para evitar conflitos de hardware (duas abas tentando usar a câmera ao mesmo tempo), foi implementada uma lógica de gerenciamento no `script.js`:

  * A função `openTab(tabId)` primeiro chama `stopAllStreams()`.
  * `stopAllStreams()` desliga **todas** as trilhas de vídeo (`track.stop()`) de ambos os streams (Alfabeto e Palavras) e remove a fonte (`srcObject`) dos elementos de vídeo.
  * Somente **depois** de desligar tudo, a função liga a câmera específica necessária para a aba que foi clicada (`startWebcamAlfabeto()` ou `setupCameraPalavras()`).

-----

## 🚀 Como Executar o Projeto Localmente

Devido às políticas de segurança do navegador (CORS) para carregar modelos (`.json`) e o uso de Módulos JS (`import`), você **não pode** simplesmente abrir o `index.html` a partir do arquivo. Você precisa servi-lo a partir de um servidor web local.

### Pré-requisitos

1.  Um servidor web local. O mais simples é usar a extensão **Live Server** no VS Code, ou o módulo `http.server` do Python.
2.  A pasta `modelos_web/` contendo os arquivos `model.json` e `weights.bin` do seu modelo de Alfabeto.
3.  Uma **Chave de API** do [Google AI Studio](https://aistudio.google.com/app/apikey) para o Gemini.

### Passos

1.  **Clone o repositório:**

    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DA_PASTA]
    ```

2.  **Adicione seu modelo:**
    Certifique-se de que sua pasta `modelos_web/` (com os arquivos do modelo TF.js) esteja na raiz do projeto.

3.  **Adicione a Chave da API:**
    Abra o arquivo `script.js` e localize a **Linha 461** (aproximadamente). Substitua o valor da constante `API_KEY` pela sua chave:

    ```javascript
    // Linha 461 em script.js
    const API_KEY = "SUA_CHAVE_DA_API_DO_GEMINI_VEM_AQUI";
    ```

4.  **Inicie o servidor local:**
    Se você tiver o Python instalado, o método mais fácil é:

    ```bash
    # Para Python 3.x
    python -m http.server
    ```

    Alternativamente, use o **Live Server** do VS Code clicando em "Go Live".

5.  **Acesse o projeto:**
    Abra seu navegador e acesse `http://localhost:8000` (ou a porta que seu servidor indicar).

-----

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
