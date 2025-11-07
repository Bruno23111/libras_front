
export default async function handler(request, response) {
    // Apenas permite pedidos POST
    if (request.method !== 'POST') {
        response.status(405).json({ error: 'Method Not Allowed' });
        return;
    }

    // 1. Obter a chave da API (segura) das Vercel Environment Variables
    // O nome 'GEMINI_API_KEY' é o que você vai configurar no painel da Vercel.
    const API_KEY = process.env.GEMINI_API_KEY; // <-- ERRO CORRIGIDO (removido o 'a')
    
    // 2. Obter o histórico do chat do frontend (enviado no corpo do pedido)
    const { history } = request.body;

    // Verificações de segurança
    if (!API_KEY) {
        // Não expõe a chave, apenas informa que não está configurada
        response.status(500).json({ error: 'A chave da API não está configurada no servidor.' });
        return;
    }

    if (!history) {
        response.status(400).json({ error: 'Histórico do chat em falta.' });
        return;
    }

    // 3. Configurações da API (movidas do script.js para o backend)
    const GEN_MODEL = "gemini-2.5-flash-preview-09-2025";
    const GOOGLE_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEN_MODEL}:generateContent?key=${API_KEY}`;

    const systemInstruction = {
        role: "system",
        parts: [{ text: "Você é o 'Libras.IO', um assistente de IA amigável, especialista e entusiasta da Língua Brasileira de Sinais (Libras). Sua missão é ajudar estudantes a aprender, tirando dúvidas sobre gramática, história, cultura Surda e os sinais. Seja didático, encorajador e use formatação Markdown (como **negrito**) para destacar termos importantes. Responda em português do Brasil." }]
    };

    const requestBody = {
        systemInstruction: systemInstruction,
        contents: history, // Usa o histórico enviado pelo frontend
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

    // 4. Chamar a API do Google (do lado do servidor seguro)
    try {
        const geminiResponse = await fetch(GOOGLE_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
        });

        if (!geminiResponse.ok) {
            // Se o Google der erro, repassa o erro
            const errorData = await geminiResponse.json();
            console.error("Google API Error:", errorData);
            response.status(geminiResponse.status).json({ error: errorData.error.message });
            return;
        }

        const data = await geminiResponse.json();

        // 5. Enviar a resposta de volta ao frontend
        if (data.candidates && data.candidates.length > 0) {
            if (data.candidates[0].content && data.candidates[0].content.parts) {
                const text = data.candidates[0].content.parts[0].text;
                response.status(200).json({ text: text });
            } else if (data.candidates[0].finishReason === 'SAFETY') {
                response.status(200).json({ text: "Desculpe, não posso responder a essa pergunta pois ela viola as minhas diretrizes de segurança." });
            }
        } else {
             response.status(500).json({ error: "Não foi obtida uma resposta válida da API do Google." });
        }

    } catch (error) {
        // Erro de rede ou ao chamar o fetch
        console.error("Internal Server Error:", error);
        response.status(500).json({ error: 'Falha ao comunicar com a API do Google.' });
    }
}