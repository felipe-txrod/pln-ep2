# Contexto persistente — EP2

Atualizado em 04/11/2026, fuso America/Sao_Paulo. Estado do código/resultados
antes destes arquivos: commit `2aa99fc`. Confira `git log` e os manifests para
mudanças posteriores; este documento é um resumo, não uma autorização permanente.

## Projeto e protocolo

Classificar a nota subjetiva de clareza de respostas públicas do e-SIC usando
GPT pelo terminal, tentando melhorar o NorBERTo do EP1. Entrada: `resp_text`.
Classes: `c1` = nota 1, `c234` = notas 2–4, `c5` = nota 5. Pedido original,
anexos e avaliador não estão disponíveis; não inventar contexto.

- `data/train_ep1.xlsx`: 20.092 registros; `train_ep2.xlsx` tem os mesmos dados.
- `data/test1.xlsx`: 900 registros sem rótulos; gabarito em
  `data/test1-rotulado-para-teste-estatistico.xlsx`. Uso oficial: McNemar com EP1.
- `data/test2.xlsx`: 897 registros; entregar XLSX rotulado com mesmas linhas,
  colunas e ordem. Primeiro congelar o método e predizer teste2, depois teste1.
- Desenvolvimento: primeiro fold de `StratifiedGroupKFold(5, shuffle=True,
  seed=42)`, com subamostra estratificada. Todo o fold fica fora do banco de
  exemplos. Agrupamento por HTML-unescape, espaços normalizados e lowercase.
- Teste1 já foi examinado em comparação isolada solicitada pelo usuário.
  Declarar essa exposição; não usá-lo para escolher novas configurações.
- Entrega oficial: relatório com código/reprodução, apresentação PDF de até
  dez minutos e teste2 rotulado, em ZIP do grupo. Prazo do enunciado: 08/11;
  apresentações 10/11 e 17/11. Faltam identificação do grupo e entrega final.

Detalhes: `docs/ep2-enunciado.md`, `docs/ep2-modelo-relatorio-com-analise.md` e
`docs/decisoes-e-requisitos.md`. Acurácia tem maior peso; não confundir métricas
internas de treino, teste1 e teste2.

## Implementação e ambiente

- Python >=3.11; ambiente validado: 3.14.8. Pacote em `src/gpt_classifier/`;
  CLI: `.venv/bin/python -m gpt_classifier` ou `src/main.py`.
- API Responses com SDK opcional `[api]`, JSON Schema e snapshot configurado
  em `configs/gpt.json`. Codex CLI >=0.165.0, login existente, `codex exec`,
  modelo `gpt-6.1-sol`, configurado em `configs/codex.json`.
- Codex executa isolado em diretório temporário, sem AGENTS, shell, ferramentas
  ou acesso ao repositório durante a classificação. O alias remoto pode mudar.
  Não presumir que parâmetro da API exista no CLI. Sem retry automático de
  processos Codex; sua contagem não mede retentativas HTTP internas.
- Padrões: prompt original, `similar`, um exemplo por classe, `batch_size=1`.
  Banco deduplicado, sem grupos conflitantes, exemplos inteiros até 1.600
  caracteres; alvo até 14.000. TF-IDF é recuperação, não o classificador GPT.
- Lotes Codex de 1–8 alvos: IDs opacos, exemplos deduplicados, parser estrito,
  divisão pelo limite de bytes. API continua individual. Variante econômica
  com prompt curto e variante híbrida permanecem experimentais.
- `hybrid`: TF-IDF + multilingual E5-small fixado por revisão, ONNX FP32 em CPU,
  prefixo query nos dois lados, 512 tokens, pooling mascarado/L2, fusão de ranks
  com pesos 0,5/0,5. Peso de comprimento zero. Sem fine-tuning.
- SQLite persiste respostas válidas por pedido e embeddings por texto/modelo.
  `prepare-retrieval` baixa pesos; inferência usa somente arquivos locais.
  `requirements-lock.txt` fixa ambiente base; `requirements-rag-lock.txt`, RAG.
- NVIDIA MX550 de 2 GB detectada fora do sandbox. Usuário autorizou GPU se
  necessário, mas o runtime atual oferece CPU. Embeddings já foram calculados
  e reaproveitados; esta GPU local não acelera chamadas remotas ao Codex.
- Última suíte: **83 testes aprovados**. Testes remotos são simulados.

## Resultados medidos

| Avaliação | Lexical/GPT | Híbrido/GPT | Referência |
|---|---:|---:|---|
| Teste1, 900 linhas | 382/900 = 42,44% | não executado | NorBERTo entregue: 432/900 = 48% |
| Piloto de 12 linhas do treino, lote 3 | 2/12 | 3/12 | 8 chamadas no total; amostra pequena |
| Validação de 300 linhas do treino, lote 3 | 121/300 = 40,33% | 127/300 = 42,33% | F1 macro: 39,76% / 41,53% |

Na comparação de 300 linhas: c1=96, c234=99, c5=105; 287 textos normalizados
únicos. Recall c1 igual nos dois: 26/96 = 27,08%. Híbrido ganhou quatro acertos
em c234 e dois em c5. Acertos exclusivos: lexical 16, híbrido 22. Mesmo rótulo
previsto em 252/300 casos. Nenhum alvo truncado no prompt.

Foram **192 processos Codex** (96 por braço), **1.572.211 tokens de entrada**,
468.864 informados como cache remoto e 20.283 de saída. Tempo: aproximadamente
22min18s, além da prévia. Ganho de dois pontos na validação não demonstra
superioridade ao EP1 em teste1; padrões não foram substituídos automaticamente.

Em teste1, McNemar exato bilateral p=0,010755 favoreceu EP1, não GPT. Repetições
também limitam independência entre pares. Relatório e apresentação não podem
afirmar melhora significativa do EP2 com esses resultados.

Fontes versionadas: `docs/analise-primeiro-resultado-gpt.md`,
`docs/recuperacao-hibrida.md`, `docs/resultados-validacao-300.json` e
`docs/verificacao.md`.

## Artefatos locais e retomada

Ignorados pelo Git; não estão disponíveis automaticamente em outro clone:

- `results/test1-codex-comparison/`: 900 previsões concluídas e XLSX separado.
- `results/comparison-test1-codex/`: métricas EP1 x GPT e análises locais.
- `results/pilot-efficiency-20261104/`: 10 chamadas, seis alvos em três variantes.
  Lotes reduziram entrada em cerca de 57%, sem demonstrar ganho de qualidade.
- `results/retrieval-hibrida-v2-20261104/`: piloto concluído de 12 linhas.
- `results/retrieval-hibrida-300-20261104/`: estudo concluído de 300 linhas,
  com `scores.json`, `paired-analysis.json` e planos/previsões dos dois braços.
- `.cache/retrieval/`: pesos e embeddings locais; `.cache/responses.sqlite3`
  e caches por estudo preservam classificações. Não apagar para reiniciar.
- Teste2 tem apenas prévias documentadas, não entrega final concluída.

Comandos a partir da raiz deste repositório:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m gpt_classifier --help
# Somente download do encoder, sem enviar a base:
.venv/bin/python -m gpt_classifier prepare-retrieval
# Prévia de um novo estudo, sem chamadas GPT:
.venv/bin/python -m gpt_classifier retrieval-study \
  --pilot-limit 300 --local-limit 300 --max-new-requests 200 \
  --output-dir results/novo-estudo-preview
```

O estudo padrão é prévia, 12 registros e orçamento de oito. Para executar é
necessário `--execute`; o comando acima é apenas exemplo de preparação, não
autorização para repetir a rodada já concluída. Preserve código, ambiente,
configuração, orçamento e diretório ao retomar uma execução incompleta.

## Git e continuação

Repositório próprio: o conteúdo de `EP2/` é a raiz Git; EP1 e o projeto pai
ficam fora. Remoto solicitado: `git@github.com:felipe-txrod/pln-ep2.git`, branch
`main`. Histórico usa Conventional Commits em português. Não usar force push.

`evaluate` usa por padrão `../EP1/deliveries/test1.xlsx` como baseline: esse
arquivo não acompanha este repositório. Em outro clone, fornecer a entrega
original com `--baseline /caminho/test1-ep1.xlsx`; não inventar nem reconstruir
previsões para substituir a entrega efetiva.

Próximas decisões: escolher a estratégia final a partir das evidências e de
eventuais novos estudos explicitamente solicitados; congelar o método, gerar
teste2, fazer a comparação vinculada em teste1 e preparar os entregáveis.
Não há nova rodada remota em andamento nem orçamento aberto após o estudo de
300. O usuário prefere execução prática, pouco consumo de tokens e tempo,
com commits incrementais em português.
