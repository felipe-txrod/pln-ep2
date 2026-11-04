# Verificação da implementação — 03/11/2026

Ambiente isolado `EP2/.venv`, Python 3.14.8, OpenAI SDK 2.54.0; dependências
registradas em `requirements-lock.txt`. `pip check`: nenhuma incompatibilidade.
Compilação dos módulos: sem erros.

`python -m pytest -q`: **58 testes aprovados** após a extensão para Codex e a cópia de comparação de teste1. O transporte HTTP simulado usa o
SDK oficial real e verifica o payload de `/v1/responses`, parsing, accounting de
usage, retries transitórios, autenticação/quota sem retry e limite de tentativas.
Também se verificam ausência de gabarito nos requests, partições agrupadas,
seleção balanceada, recusas e JSON inválido, retomada de execução interrompida,
cache de duplicatas, grade completa com seleção e empate, McNemar bilateral e
direção do ganho, alinhamento dos 900 textos da entrega EP1 e exportação das 897
linhas em arquivo temporário. Na exportação, apenas o XML da primeira planilha
mudou no ZIP; as demais entradas ficaram idênticas, inclusive os estilos.

Execuções locais com dados reais, **sem chamadas de API**:

| Verificação | Resultado |
|---|---|
| `audit` nas quatro bases | concluída, 20.092 exemplos de treino; teste1=900; teste2=897 |
| `develop --limit 150 --dry-run` | quatro candidatos preparados, 146 requests distintos por candidato |
| Banco interno da partição | 16.010 linhas; 11.661 exemplos elegíveis para recuperação |
| `predict --dry-run` de teste2 | 897 linhas; 862 requests distintos; zero alvos truncados |
| Banco final | 20.092 linhas; 14.597 exemplos elegíveis para recuperação |
| Estado das execuções | `dry_run`; nenhum arquivo de previsões fictícias publicado |

Os planos ficam em `results/development-preview/` e `results/test2-preview/`;
a inspeção das bases fica em `results/data-audit.json`. São saídas locais
ignoradas pelo Git. Os previews originais correspondem ao código anterior à
extensão Codex; novos experimentos devem usar diretórios novos. Textos de treino,
teste, planilhas e artefatos do EP1 não foram alterados.

**Esta etapa não mediu acurácia GPT.**
Respostas simuladas dos testes são fixtures de software e não evidência de
qualidade do classificador. A acurácia 48,0% do EP1 no teste1 foi recalculada a
partir da entrega existente, exclusivamente como referência histórica. Ainda
precisam ser executados o estudo autenticado, teste2 e a comparação final para
afirmar qualquer melhora. O relatório acadêmico, a apresentação PDF e o ZIP da
entrega dependem desses resultados e dos dados do grupo.

## Verificação da extensão Codex

- Codex CLI: **0.165.0**, com backend de autenticação ChatGPT.
- Chamada real de integração: uma frase sintética, zero-shot, `gpt-6.1-sol`,
  reasoning `low`, sem `OPENAI_API_KEY`, sem acesso às bases oficiais pelo modelo.
  Retorno validado: `{"label":"c5"}`. Usage informado: 5.724 tokens de entrada,
  16 tokens de saída, nenhum token de entrada em cache remoto. Não é uma medida
  de acurácia. O fluxo de invocação foi mantido após ajustes locais de validação
  e metadados; as execuções de software seguintes usam subprocesso simulado.
- `predict --backend codex --dry-run`: plano completo das **897** linhas de
  teste2 em `results/test2-codex-preview/`, com **862** requests distintos,
  zero alvos truncados e nenhuma classificação remota iniciada. O manifesto
  registra a configuração final, o runtime e hashes do adaptador.
- Novos testes usam executável real de subprocesso com implementação simulada,
  sem rede: stdin sem interpolação shell, diretório temporário, flags de
  isolamento, ausência de import do SDK/chave, orçamento, versão ausente/antiga,
  timeout com encerramento do processo, saída inválida, eventos de ferramenta,
  falha, ausência de conclusão/usage, cache/retomada e separação API/Codex.
- O SDK OpenAI passou a ser dependência opcional `[api]`; a instalação editável
  foi atualizada. O lock completo continua incluindo API e testes.

A classificação integral de treino/desenvolvimento/teste2/teste1 via Codex não
foi iniciada. Esse processamento pode ser feito pelos comandos documentados,
observando os limites da conta. A integração validada não demonstra melhora
sobre o NorBERTo.

O fluxo isolado de teste1 também foi verificado: classificação simulada das 900
linhas, exportação de uma cópia XLSX, avaliação contra o gabarito e rejeição de
teste1 como entrega oficial. O teste confere preservação da entrada e recusa
exportar a partir do gabarito. Nenhuma classificação remota das bases oficiais
foi executada nessa verificação.

## Atualização de 04/11/2026 — análise e lotes experimentais

Suíte atual: **69 testes aprovados**. Novos testes verificam JSON por ID,
rejeição de lotes incompletos/duplicados/inválidos, preservação da ordem de
linhas, deduplicação, teto de bytes e retomada somente dos lotes pendentes.

O resultado completo de teste1 foi analisado: GPT 382/900 contra NorBERTo
432/900. Os arquivos originais e seus dados foram preservados. Foi executado
um piloto com dez processos Codex e seis alvos retirados
do treino, em três variantes. Os lotes de três reduziram o consumo observado
de entrada, mas o piloto não demonstrou ganho de acurácia. Nenhuma configuração
nova foi promovida. A análise, os custos e as limitações estão em
`analise-primeiro-resultado-gpt.md`; o padrão continua individual e com o prompt
original. A nova funcionalidade altera fingerprints, exigindo diretórios novos
para experimentos seguintes.

## Recuperação híbrida — 04/11/2026

- Suíte final: `82 passed in 21.75s`; `git diff --check` sem erros e
  `pip check` sem conflitos. Os testes remotos usam transportes/executáveis
  simulados e não consomem chamadas reais.
- Embeddings reais E5/ONNX em CPU: smoke test de dimensão e similaridade,
  cache persistente com 11.948 textos únicos da preparação inicial. Pesos
  e tokenizador fixados por revisão e hashes; dependências opcionais em
  `requirements-rag-lock.txt`.
- Diagnóstico em 300 registros do treino: vizinho mais próximo 128/300 lexical
  e 136/300 híbrido. São proxies locais, não classificações GPT.
- Piloto real pareado em 12 registros do treino: oito processos Codex no total,
  2/12 acertos lexical e 3/12 híbrido. Total informado pelo CLI: 68.594 tokens
  de entrada e 746 de saída. Nenhum gabarito de teste1/teste2 foi utilizado.
- Artefatos em `results/retrieval-hibrida-v2-20261104/`; detalhes e limitações
  em [recuperacao-hibrida.md](recuperacao-hibrida.md). O piloto não selecionou
  automaticamente um método nem alterou os padrões anteriores.
- A primeira prévia local foi interrompida para eliminar normalizações
  repetidas; não fez chamadas GPT. Seus artefatos foram preservados em
  `results/retrieval-hibrida-20261104/` e a execução concluída usa outro destino.

## Comparação em lotes nos 300 registros — 04/11/2026

- Comparação com teto de 200 processos e
  três textos por chamada. O orçamento padrão continua oito; testes confirmam
  a aceitação do orçamento maior, a rejeição de valores inválidos e a
  preservação dos padrões.
- Suíte atual: `83 passed in 25.19s`. Conferência final de diferenças sem erros.
- Dois braços concluídos, com 300 previsões ordenadas e alinhadas por hash em
  cada um; mesmo código e ambiente, demonstrações apenas do banco de treino.
- 192 processos reais (96 por braço), sem falhas; 1.572.211 tokens de entrada,
  468.864 informados como cache remoto e 20.283 de saída. Não houve truncamento
  de alvos no prompt. Embeddings locais reutilizados.
- Acurácia lexical 121/300, híbrida 127/300; F1 macro 39,76% e 41,53%.
  Acertos exclusivos: lexical 16 e híbrido 22. Recall c1 igual: 26/96.
- Artefatos: `results/retrieval-hibrida-300-20261104/`, incluindo
  `paired-analysis.json`; agregado versionado em `resultados-validacao-300.json`.
  Nenhum teste1/teste2 utilizado. Não houve seleção automática nem expansão
  adicional de chamadas após a conclusão.
