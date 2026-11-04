# EP2 — Classificação de clareza com GPT pelo terminal

O pipeline oferece dois backends: **API Responses** com o SDK Python oficial e
**Codex CLI** com `codex exec`, ambos com JSON Schema estrito.
Recebe `resp_text`, retorna uma classe entre `c1`, `c234` e `c5`, sem fine-tuning.
Na API, o snapshot inicial é `gpt-5.4-mini-2026-03-17` (`configs/gpt.json`).
No Codex, a configuração inicial usa `gpt-6.1-sol` (`configs/codex.json`).
O primeiro resultado e as variantes experimentais estão descritos abaixo.

**Regra do enunciado:** teste2 (897 linhas) é a entrega; teste1 (900 linhas) fica
reservado à comparação final por McNemar. A busca de exemplos/prompts usa somente
treino. Consulte [requisitos e decisões](docs/decisoes-e-requisitos.md) para a
análise do EP1, das bases, das restrições e as fontes oficiais consultadas.

## Instalação

Execute os comandos abaixo em `EP2/`, com Python >=3.11, preferencialmente 3.14
(ambiente usado na verificação). Não é preciso GPU nem dependências do NorBERTo.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m gpt_classifier --help
```

O lock fixa as versões instaladas e testadas. `pyproject.toml` permite instalar
com resolução de dependências atualizada: `pip install -e '.[dev]'`. Para usar
somente Codex, `pip install -e .` não exige o SDK `openai`; para a API, use
`pip install -e '.[api]'`. O lock completo inclui API e testes. Preserve o
lock, o ambiente e os resultados de cada experimento. Para usar sem instalação
editável, `PYTHONPATH=src .venv/bin/python -m gpt_classifier ...` também funciona.

## Usar com Codex, sem chave de API

É necessário **Codex CLI >=0.165.0**, versão verificada neste projeto. O login
ChatGPT do Codex pode ser reutilizado pelo processo, conforme acesso e limites
do seu plano. O classificador não lê nem copia o arquivo de credenciais.

```bash
codex --version
codex login status
# Se ainda não houver sessão válida:
codex login

.venv/bin/python -m gpt_classifier develop --backend codex \
  --output-dir results/development-codex --limit 150

.venv/bin/python -m gpt_classifier predict \
  --config results/development-codex/selected-config.json \
  --output-dir results/test2-codex

.venv/bin/python -m gpt_classifier export \
  --run results/test2-codex --output deliveries/test2-codex.xlsx

.venv/bin/python -m gpt_classifier predict --phase statistical \
  --test2-run results/test2-codex \
  --input data/test1-rotulado-para-teste-estatistico.xlsx \
  --config results/development-codex/selected-config.json \
  --output-dir results/test1-codex

.venv/bin/python -m gpt_classifier evaluate \
  --run results/test1-codex --output-dir results/comparison-codex
```

Antes de consumir o plano, acrescente `--dry-run` e escolha outro diretório para
preparar os requests. A consulta local `codex --version` ocorre no dry-run, sem
iniciar classificação. Para predizer diretamente com a configuração inicial,
sem a seleção de desenvolvimento, use `predict --backend codex --output-dir
results/test2-codex-direto` e declare essa escolha no relatório.

`--backend codex` sem `--config` seleciona `configs/codex.json`; um arquivo salvo
pelo estudo já contém o backend e não exige repetir a opção. `--model` permite
escolher explicitamente outro modelo disponível à conta. Para outro executável,
use `--codex-bin /caminho/codex`. Não há troca automática de modelo ou backend.
O `gpt-5.4-mini` da configuração de API não é reaproveitado automaticamente:
a documentação oficial informa sua retirada do Codex com login ChatGPT.

Cada classificação inicia uma sessão nova e efêmera em diretório temporário
fora do repositório, com prompt via **stdin**, `--output-schema`,
`--output-last-message`, `--json`, sandbox `read-only` e `--ignore-user-config`.
Instruções de classificação substituem as instruções padrão de programação.
Shell, apps, plugins, hooks, navegação, memória e delegação são desativados;
AGENTS.md e instruções automáticas de skills não são incluídos. Não se resume
uma sessão anterior. Eventos que indiquem ferramenta ou falha são rejeitados.
Esses controles impedem que a classificação procure contexto nas planilhas.

O cache distingue API e Codex e inclui modelo, prompt, schema, versão/hash do
CLI e política/hash do adaptador. A passagem de teste2 para teste1 exige o mesmo
backend e runtime. Como `gpt-6.1-sol` é um alias, a versão de serviço pode mudar;
`model_is_requested_alias` sinaliza esse limite da reprodução no Codex.

O timeout padrão do Codex é **300 segundos por execução**. `--max-new-requests`
conta processos Codex iniciados, não requests HTTP internos do CLI. O adaptador
não reinicia processos em caso de erro ou timeout; o CLI pode fazer suas próprias
retentativas internas. `--retries` é exclusivo da API. Repetir o comando após
corrigir a falha retoma o cache. Timeout/interrupção encerra o grupo de processos.
O Codex não expõe um equivalente verificado de `max_output_tokens` ou
`temperature` para este fluxo: ambos devem ser `null` na configuração; são
rejeitados em vez de ignorados. O consumo pode incluir contexto adicional do
CLI e limites da assinatura. Tokens são obtidos de `turn.completed.usage`, sem
inventar preços ou garantir equivalência de resultados à API.

Fontes: [execução não interativa](https://learn.chatgpt.com/docs/non-interactive-mode),
[referência do CLI](https://learn.chatgpt.com/docs/developer-commands?surface=cli),
[configuração](https://learn.chatgpt.com/docs/config-file/config-reference),
[modelos disponíveis](https://learn.chatgpt.com/docs/models) e
[autenticação](https://learn.chatgpt.com/docs/auth), consultadas em 03/11/2026.

## Usar com a API

A chave deve estar apenas no ambiente do terminal; não a grave em código,
configuração, notebook, Git ou em arquivos de resultados. Por exemplo, em Bash:

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

Usa-se o endpoint oficial `https://api.openai.com/v1`. É necessária uma chave
com acesso ao modelo e faturamento da API. O código não acessa credenciais ou
sessões de outros aplicativos. O cliente remoto só é criado quando há uma
classificação ausente do cache. **Sem chave é possível auditar, executar todos
os testes e preparar requests com `--dry-run`.**

## Preparação e desenvolvimento

### Recuperação híbrida de exemplos (RAG experimental)

`strategy=hybrid` combina TF-IDF e embeddings locais do multilingual E5-small.
O encoder tem revisão fixada e executa em CPU via ONNX, sem API de embeddings.
O prompt continua com um exemplo inteiro por classe (três no total), filtrando
textos repetidos, conflitantes e o próprio alvo. Os padrões anteriores continuam
disponíveis. Consulte [método e avaliação](docs/recuperacao-hibrida.md).
No piloto real, o híbrido acertou 3/12 contra 2/12 do lexical, em oito chamadas
totais. Isso não demonstra ganho confiável: os padrões não foram substituídos.

```bash
# Instalação reproduzível do complemento opcional:
.venv/bin/python -m pip install -r requirements-rag-lock.txt
.venv/bin/python -m pip install --no-deps -e .
# Alternativa com resolução das dependências: pip install -e '.[rag]'

# Só baixa os pesos públicos. Não envia textos da base:
.venv/bin/python -m gpt_classifier prepare-retrieval

# Diagnóstico local (300 registros) e prévia do piloto, sem chamadas GPT:
.venv/bin/python -m gpt_classifier retrieval-study \
  --output-dir results/retrieval-hibrida

# Mesmo estudo e mesma amostra; até oito novas execuções Codex:
.venv/bin/python -m gpt_classifier retrieval-study --execute \
  --max-new-requests 8 --output-dir results/retrieval-hibrida
```

O piloto compara 12 textos do treino, com o mesmo prompt e lotes de três nos
dois braços. A prévia verifica quantas chamadas serão necessárias e recusa
iniciar se exceder o teto. Ele não escolhe automaticamente uma configuração
nem demonstra ganho de generalização. Uma execução interrompida pode ser
retomada com o mesmo comando. Modelos e vetores ficam em `.cache/retrieval/`;
`--retrieval-cache` muda esse local. `predict` nunca baixa modelos implicitamente.

Para preparar predições experimentais, sem iniciar classificação:

```bash
.venv/bin/python -m gpt_classifier predict \
  --config configs/codex-hibrido-experimental.json --dry-run \
  --output-dir results/test2-hibrido-preview
```

Para usar a mesma recuperação com API, acrescente `--strategy hybrid` ao comando
com `configs/gpt.json` (mantém `batch_size=1`). Em `develop`, uma configuração
híbrida acrescenta um quinto candidato à grade original; isso pode consumir
muitas chamadas. Prefira o piloto limitado antes de executar a grade.
Mudanças no código/ambiente invalidam a identidade de execuções anteriores:
preserve os resultados completos existentes e use novos diretórios.

Para ampliar explicitamente a comparação aos mesmos 300 registros do treino,
mantendo três textos por chamada e teto compartilhado de 200 processos:

```bash
# Prévia, sem chamadas GPT:
.venv/bin/python -m gpt_classifier retrieval-study \
  --pilot-limit 300 --local-limit 300 --max-new-requests 200 \
  --output-dir results/retrieval-hibrida-300

# Executar (ou retomar) exatamente o plano preparado:
.venv/bin/python -m gpt_classifier retrieval-study --execute \
  --pilot-limit 300 --local-limit 300 --max-new-requests 200 \
  --output-dir results/retrieval-hibrida-300
```

O limite de oito chamadas continua sendo o padrão; a ampliação depende do valor
explícito. Na amostra de seed 42, a deduplicação reduz o plano para 96 chamadas
por seletor, totalizando 192. São 300 linhas da amostra, avaliadas
pelos dois métodos, sem incluir teste1/teste2. Há textos normalizados repetidos;
isso deve ser considerado ao interpretar a comparação.

Na execução concluída, o GPT com recuperação híbrida acertou **127/300 (42,33%)**,
contra **121/300 (40,33%)** com TF-IDF. O ganho foi de seis acertos; o recall de
`c1` permaneceu em 27,08%. Foram 192 processos Codex e 1.572.211 tokens de entrada.
Veja a [análise completa](docs/recuperacao-hibrida.md#resultado-dos-300-registros)
e as [métricas agregadas](docs/resultados-validacao-300.json). O resultado é de
validação interna do treino e não substitui a comparação final com EP1.

### Resultados anteriores

O primeiro resultado real e o piloto de eficiência estão documentados na
[análise de 04/11/2026](docs/analise-primeiro-resultado-gpt.md): o GPT obteve
42,44% contra 48,00% do EP1. Ainda não há uma configuração nova com ganho de
acurácia demonstrado. Evite repetir as 900 chamadas apenas para tentar outro
prompt; faça o estudo no treino.

### Lotes experimentais para economizar chamadas Codex

O padrão continua sendo `batch_size=1` e `prompt_profile=original`. O modo em
lote aceita de 2 a 8 textos por processo, com IDs, JSON validado, deduplicação
de exemplos e retomada por cache. As opções são gravadas na configuração
efetiva; devem permanecer iguais entre teste2 e teste1. O limite de bytes pode
reduzir o tamanho de um lote sem truncar alvos adicionais.

```bash
# Preparar um estudo pequeno, sem consumir classificações remotas:
.venv/bin/python -m gpt_classifier develop --backend codex \
  --batch-size 3 --limit 6 --max-new-requests 10 --dry-run \
  --output-dir results/development-lotes-preview
```

No piloto de seis textos, o modo em lote reduziu tokens, mas mudou algumas
predições e não mostrou ganho de acurácia. Portanto, permanece experimental.
`configs/codex-economico-experimental.json` usa também um prompt curto
(`prompt_profile=empirical`); não substitui `configs/codex.json`.
O comando de estudo acima prepara quatro candidatos, nominalmente até oito
execuções com lotes de três; se o limite de bytes exigir lotes menores, o teto
`--max-new-requests` ainda interrompe a execução antes de exceder dez processos.
Lotes não são suportados pelo backend API nesta implementação.

Para reproduzir o piloto pareado (individual/original em lote/curto em lote),
`scripts/pilot_efficiency.py` prepara os requests por padrão. `--execute` inicia
até dez processos Codex e pode consumir o plano. A amostra de seis registros
serve apenas à triagem; não é suficiente para escolher o modelo final.

### Grade de exemplos

Inspecione as bases e prepare a grade sem consumir API:

```bash
.venv/bin/python -m gpt_classifier audit --output results/auditoria.json
.venv/bin/python -m gpt_classifier develop \
  --output-dir results/development-preview --limit 150 --dry-run
```

São comparadas quatro configurações, na mesma amostra estratificada de um fold
agrupado do treino: zero-shot, um exemplo representativo por classe, um semelhante
por classe, dois semelhantes por classe. IDF/vocabulário usam somente o banco de
treino do fold. Todos os registros do fold ficam fora do banco, inclusive os não
selecionados para a amostra de 150 linhas. A chave de agrupamento normaliza HTML,
espaços e maiúsculas; duplicatas aproximadas podem atravessar a partição.

Para executar a investigação real:

```bash
.venv/bin/python -m gpt_classifier develop \
  --output-dir results/development --limit 150 --max-new-requests 1250
```

O máximo nominal é 600 classificações antes de deduplicação/cache. Retentativas
também consomem o limite de tentativas novas. A busca salva `split.json`,
`scores.json`, `selection.json` e **`selected-config.json`**, escolhido por
acurácia; empate favorece a primeira configuração da grade. Macro F1, matriz de
confusão e tokens são diagnósticos. Repetir buscas na mesma amostra aumenta viés
de seleção. Não usar resultados do teste1 para alterar esse arquivo.

Não há garantia de que GPT ou mais exemplos superem o NorBERTo. Para comparar
outros modelos ou limites de exemplos, crie outra configuração explícita e
outro diretório de desenvolvimento. Consulte o suporte a `reasoning` e
`temperature` daquele modelo: valores `null` são omitidos, erros não acionam
troca automática nem remoção silenciosa de parâmetros. Seed é apenas local.
O snapshot fixa a versão, mas a API não garante determinismo absoluto.

## Congelar o método e produzir teste2

Primeiro confira o plano usando a configuração escolhida (se não houve estudo,
use `--config configs/gpt.json` e declare que não houve otimização):

```bash
.venv/bin/python -m gpt_classifier predict \
  --config results/development/selected-config.json \
  --output-dir results/test2-preview --dry-run

.venv/bin/python -m gpt_classifier predict \
  --config results/development/selected-config.json \
  --output-dir results/test2 --max-new-requests 2000

.venv/bin/python -m gpt_classifier export \
  --run results/test2 --output deliveries/test2.xlsx
```

`predict` usa `data/train_ep1.xlsx` e `data/test2.xlsx` por padrão. A exportação
exige todas as 897 previsões, rótulos válidos, hashes de texto e ordem das linhas
corretos. Preenche apenas `clarity` em uma cópia do arquivo original: mesmas
abas/linhas/colunas, estilos e demais entradas do arquivo XLSX. Nunca sobrescreve
originais ou saídas existentes e nunca gera entrega parcial. O exportador não
aceita teste1 como entrega. Metadados ficam em JSON, fora da planilha oficial.

## Comparar teste1 separadamente e salvar outra planilha

Com `data/test1.xlsx` sem rótulos, use o modo explícito `comparison`. Os exemplos
vêm de `train_ep1.xlsx`; a configuração Codex padrão seleciona um exemplo
semelhante por classe (três no total) para cada resposta. O gabarito só é lido
na etapa local de avaliação:

```bash
.venv/bin/python -m gpt_classifier predict --backend codex --phase comparison \
  --config configs/codex.json --train data/train_ep1.xlsx \
  --input data/test1.xlsx --output-dir results/test1-codex-comparison

.venv/bin/python -m gpt_classifier export --purpose comparison \
  --input data/test1.xlsx --run results/test1-codex-comparison \
  --output results/test1-codex-comparison/test1-predito.xlsx

.venv/bin/python -m gpt_classifier evaluate \
  --run results/test1-codex-comparison \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --train data/train_ep1.xlsx --output-dir results/comparison-test1-codex
```

`test1-predito.xlsx` é uma cópia para comparação, com 900 rótulos; não é a
entrega oficial de teste2. `metrics.json` e `report-data.json` ficam no diretório
de avaliação, com acurácia, métricas por classe e McNemar contra a entrega EP1.
Esse modo não exige uma execução prévia de teste2. Não ajuste o método usando
o resultado do teste1; para a sequência oficial vinculada a teste2, use o modo
`statistical` abaixo. A entrada, o treino e o gabarito são preservados.

## Comparação estatística final em teste1

Após concluir teste2, aplique **a mesma configuração e o mesmo treino** a teste1:

```bash
.venv/bin/python -m gpt_classifier predict \
  --phase statistical --test2-run results/test2 \
  --input data/test1-rotulado-para-teste-estatistico.xlsx \
  --config results/development/selected-config.json \
  --output-dir results/test1 --max-new-requests 2000

.venv/bin/python -m gpt_classifier evaluate \
  --run results/test1 --output-dir results/comparison

.venv/bin/python -m gpt_classifier explain \
  --run results/test1 --output results/interpretation.json
```

O leitor de inferência descarta `clarity` e outras colunas. Só textos dos alvos
entram no construtor de requests; os rótulos são lidos apenas por `evaluate`,
`explain` e pelo estudo interno do treino. As demonstrações enviadas ao GPT
contêm apenas rótulos de treino. Nem o nome do arquivo nem o número da linha
do alvo são enviados ao modelo.

`evaluate` alinha as previsões por linha e SHA-256 do texto e confere a entrega
**efetiva** do EP1 em `../EP1/deliveries/test1.xlsx`. Sua acurácia é 432/900
(48,0%), diferente dos 45,31% do holdout interno histórico. Gera:

- `metrics.json`: acurácia, F1 macro, precisão/recall por classe e matriz de
  confusão, baseline majoritário do treino e resultados pareados.
- `report-data.json`: valores resumidos para o relatório.
- `errors.jsonl`: erros para inspeção e comentário **manual**, sem justificativas
  inventadas ou enviadas de volta ao classificador.

McNemar: a tabela tem linhas EP1 correto/incorreto e colunas GPT correto/incorreto.
`b` é apenas EP1 correto; `c` é apenas GPT correto. São reportados chi-quadrado
com/sem correção de continuidade e **p exato bilateral binomial** (teste primário).
Só há superioridade significativa quando `c>b` e `p<0,05`. Sem discordâncias,
estatística 0 e p=1. Textos repetidos e avaliações subjetivas limitam a
independência dos pares e a inferência; o teste não elimina esse problema.

`explain` ajusta uma regressão logística auxiliar só no treino e exporta pesos
globais e contribuições locais `TF-IDF × coeficiente`, com intercepto e score.
Isso atende à preparação da análise de características pedida no relatório;
**explica o modelo auxiliar**, sem alegar acesso aos mecanismos internos do GPT.
Os exemplos locais priorizam erros GPT. A discussão qualitativa fica a cargo
dos autores após ler os casos, não do algoritmo.

## Robustez, eficiência e auditoria

- Por padrão, uma classificação por request e saída estrita `{"label":"c1|c234|c5"}`
  (a string real é exatamente uma das três classes). Recusas, JSON inválido,
  chaves duplicadas e respostas incompletas interrompem sem inventar rótulos.
- Poucos exemplos balanceados, um por grupo e sem rótulos conflitantes. Textos
  exatamente iguais ao alvo são excluídos das demonstrações. Seleção TF-IDF com
  penalização de redundância, determinística em empates. Não é classificador
  lexical disfarçado: quem produz a etiqueta é o GPT.
- Normalização preserva caixa, acentos, pontuação e negação na entrada GPT.
  Exemplos inteiros têm até 1.600 caracteres. Alvos acima de 14.000 caracteres
  são truncados início/fim com marcador explícito e registro em `plan.jsonl`.
  Nenhum texto das bases fornecidas excede o limite inicial de alvos.
- `max_request_bytes` limita bytes UTF-8 do payload antes da chamada; **não é
  contagem exata de tokens nem garantia monetária**. Tokens reais de entrada,
  cache remoto e saída são registrados a partir de `usage`. Não há preços
  hard-coded ou alegação de custo zero para cache remoto.
- `--timeout`, `--retries`, `--min-interval` e `--max-new-requests` controlam a
  execução sequencial. SDK com retries desativados; retentativas próprias só em
  conexão/timeout, 408, 409, 429 e 5xx, com backoff e `Retry-After`. Autenticação,
  parâmetros inválidos e quota financeira não geram loops de retentativa.
  Esperas do servidor >60s param a execução para retomada posterior.
- SQLite faz commit a cada resposta válida, identificado por **payload completo,
  modelo, schema, prompt e endpoint**. Cache compartilhado entre runs, com trava
  contra execução concorrente. Repetir o mesmo comando retoma pelo cache.
  Há uma janela inevitável de duplicação se o processo morrer depois da cobrança
  remota e antes do commit local. Uma resposta incompleta/recusada pode consumir
  tokens e não é reutilizada como classificação.
- Cada diretório tem `run.json`, configuração, hashes do código/ambiente/dados,
  histórico das retomadas, plano por linha e primeiro request auditável.
  `predictions.jsonl` só é publicado após conclusão integral. Não reutilize o
  diretório após mudar código, modelo, dados, ambiente ou modo dry-run.
- `--max-new-requests` conta tentativas desta invocação, inclusive retries. Não
  é teto monetário e não inclui cobranças que não retornaram `usage` por falha
  de rede. Use os limites de faturamento da conta para teto financeiro.

Requests e cache contêm trechos de treino e textos de teste. Mantenha-os em
armazenamento apropriado; `.gitignore` exclui resultados/cache/credenciais.
`store=False` é enviado em cada chamada de API; no Codex usa-se `--ephemeral`,
sem prometer retenção zero de logs do provedor. O código não usa web, ferramentas externas, anexos ou perguntas
originais durante a classificação.

## Verificação e pendências acadêmicas

```bash
.venv/bin/python -m pytest -q
```

Os testes usam o SDK real sobre transporte HTTP simulado e um executável Codex
simulado: não consomem API ou plano ChatGPT. Cobrem
não vazamento de gabarito, seleção e divisão agrupadas, schema, falhas, limites,
retomada, McNemar e exportação XLSX usando o modelo real de 897 linhas em arquivo
temporário. As planilhas fornecidas e o EP1 permanecem preservados.

Ainda são necessários resultados GPT reais nas bases oficiais para afirmar melhora, conferir o
baseline oficial e concluir análise de erros. A entrega final acadêmica também
requer integrantes/números USP, URL do repositório, relatório e apresentação PDF
de até dez minutos no `grupoXX.zip`, junto a `deliveries/test2.xlsx`. Este projeto
implementa o experimento e prepara os dados do relatório, sem declarar esses
entregáveis concluídos antes de haver evidência.
