<a id="frozen-gpt-configuration"></a>

# Configuração congelada do GPT

O pacote armazena `config.json`, `prompt.txt` e `empirical_prompt.txt`, com
checksums em `manifest.json`. O snapshot remoto configurado é
`gpt-5.4-mini-2026-03-17`, pela integração de API.

Pesos GPT remotos não estão disponíveis localmente; o pacote não realiza
inferência offline. Não contém chaves ou credenciais. O pipeline existente
aceita `--config models/gpt/config.json`.

Os prompts são cópias exatas dos modelos-fonte identificados no manifesto;
a aplicação ainda usa sua implementação de prompts. Banco de demonstrações
e recuperação continuam necessários quando há exemplos de treino configurados.
Mudanças em prompt, banco ou parâmetros exigem nova identidade de execução.
