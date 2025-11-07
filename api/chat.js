// --- /api/chat.js ---
// IMPORTANTE: Este ficheiro deve estar na pasta /api na raiz do projeto.

export default async function handler(request, response) {
    if (request.method !== "POST") {
        response.status(405).json({ error: "Method Not Allowed" });
        return;
    }

    const API_KEY = process.env.GEMINI_API_KEY;
    console.log("Valor de GEMINI_API_KEY:", API_KEY ? "Encontrada" : "NÃO ENCONTRADA");

    const { history } = request.body;

    if (!API_KEY) {
        response.status(500).json({ error: "A chave GEMINI_API_KEY não está configurada na Vercel." });
        return;
    }

    if (!history) {
        response.status(400).json({ error: "Histórico do chat não enviado." });
        return;
    }

    const GEN_MODEL = "gemini-pro"; // Modelo estável e suportado
    const GOOGLE_API_URL = `https://generativelanguage.googleapis.com/v1/models/${GEN_MODEL}:generateContent?key=${API_KEY}`;

    const systemInstruction = {
        role: "system",
        parts: [{
            text: "Você é o 'Libras.IO', um assistente de IA amigável e especialista em Língua Brasileira de Sinais (Libras). Ajude estudantes de forma clara, didática e empática, usando **Markdown** para destacar conceitos importantes. Responda sempre em português do Brasil."
        }]
    };

    const requestBody = {
        contents: [systemInstruction, ...history],
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
        },
    };

    async function callGeminiWithRetry(maxRetries = 3) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const geminiResponse = await fetch(GOOGLE_API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody),
                });

                if (!geminiResponse.ok) {
                    const errorData = await geminiResponse.json().catch(() => ({}));
                    console.error(`Erro Google API (tentativa ${attempt}):`, errorData);
                    if (attempt === maxRetries) throw new Error(errorData.error?.message || "Erro desconhecido");
                    await new Promise(res => setTimeout(res, 2000)); // espera 2s e tenta de novo
                    continue;
                }

                const data = await geminiResponse.json();
                if (data.candidates?.length > 0 && data.candidates[0].content?.parts) {
                    return data.candidates[0].content.parts[0].text;
                }

                throw new Error("Resposta inválida do modelo");
            } catch (err) {
                console.warn(`Erro na tentativa ${attempt}:`, err.message);
                if (attempt === maxRetries) throw err;
                await new Promise(res => setTimeout(res, 2000));
            }
        }
    }

    try {
        const text = await callGeminiWithRetry();
        response.status(200).json({ text });
    } catch (error) {
        console.error("Erro final após múltiplas tentativas:", error);
        response.status(500).json({ error: "Falha ao comunicar com o Gemini. Tente novamente." });
    }
}
