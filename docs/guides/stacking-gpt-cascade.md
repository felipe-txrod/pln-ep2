<a id="frozen-stacking-with-selective-gpt-decisions"></a>

# Stacking congelado com decisões seletivas do GPT

O [experimento medido concluído](../results/stacking-gpt-cascade.md) selecionou
zero chamadas ao GPT na validação; os três orçamentos positivos também reduziram
a qualidade em teste1. O pacote congelado é `models/stacking-gpt-cascade/`.
Roteamento configurável e comparações explícitas de orçamento permanecem
disponíveis conforme descrito abaixo.

A cascata reutiliza SVM, regressão logística de desenvolvimento, adaptador
original do NorBERTo do EP1, Qwen, calibradores congelados e stacking publicado
de quatro membros. Nenhum classificador-base é reajustado. O GPT substitui
stacking somente nos alvos incertos selecionados; falhas da API mantêm a
decisão original e são marcadas explicitamente como fallback.

<a id="data-and-selection-protocol"></a>

## Dados e protocolo de seleção

A partição original de desenvolvimento tem 16,093 linhas. As 3,999 reservadas
já têm dois papéis com grupos disjuntos: 2,027 ajustam calibradores e 1,972
treinam o meta-modelo de stacking. As linhas de calibração ficam fora do
desenvolvimento do roteamento. Não podem servir de validação independente
do roteamento para o pipeline calibrado completo, pois seus rótulos ajustaram
os calibradores.

O desenvolvimento do roteamento usa previsões **OOF aninhadas de cinco por cinco
folds agrupados** nas 1,972 linhas de meta-validação. Cada alvo externo e seu
grupo são excluídos do ajuste do meta-modelo e da seleção interna de regularização.
Os calibradores originais permanecem congelados e nunca usaram esses 1,972
rótulos. Somente o pequeno meta-modelo é ajustado para essas estimativas OOF;
stacking de produção e pesos-base não mudam. A auditoria salva inclui IDs de
treino/validação, modelos numéricos portáveis e resultados internos da validação cruzada.

A seleção do checkpoint Qwen usou todo o conjunto reservado original, e pesquisas
anteriores consultaram conjunto reservado/teste1. A OOF do meta-modelo não
desfaz essa exposição. O novo roteamento exclui seus alvos externos do ajuste
supervisionado do meta-modelo, mas não alega validação inédita de todo o processo
histórico de seleção.

Os 900 textos finais nunca selecionam regra de roteamento, prompt, limiar ou
orçamento. A predição carrega registros `Text` sem gabarito. Todos os arquivos
finais são persistidos e verificados antes da abertura do gabarito na avaliação
separada. Este é um experimento exploratório em teste1 explicitamente solicitado;
não estabelece a ordem de entrega teste2 antes de teste1 exigida pelo enunciado.

<a id="hybrid-uncertainty-and-budgets"></a>

## Incerteza híbrida e orçamentos

Para probabilidades p do stacking, defina entropia normalizada H, margem entre
as duas maiores M e probabilidade máxima P. A incerteza de confiança é
`U = (H + (1-M) + 1.5*(1-P))/3`. A divergência D é a fração dos seis pares de
decisões nativas dos quatro membros que divergem. O SVM usa o argmax original
das margens nesse voto, mesmo se a calibração mudar seu argmax.

O escore de ordenação é `w*U + (1-w)*D`. Um alvo é elegível se a probabilidade
máxima ou margem ficar abaixo do limiar configurado, ou se algum membro
divergir. Uma busca fixa de doze candidatos nas previsões OOF escolhe w e
limiares pelos erros de stacking capturados no orçamento inicial de 10%.
Empates estáveis usam checksums dos textos e IDs das linhas, sem rótulos.
A busca salva é diagnóstico de seleção, não estimativa independente da
generalização do roteamento.

Cada orçamento seleciona no máximo `floor(N * fraction)` instâncias elegíveis
na ordem. Assim, 5%, 10% e 20% selecionam até 98, 197 e 394 instâncias na
validação do roteamento, ou 45, 90 e 180 em teste1 final. As frações incidem
sobre o total N da coorte elegível de avaliação, inclusive linhas que permanecem
locais. As seleções são aninhadas. Orçamento zero também é comparado, sem
presumir melhora pelo GPT.

A primeira etapa remota compara prompts somente de texto e com previsões locais
no mesmo subconjunto inicial de 10%. Macro-F1 em todas as linhas OOF escolhe
o prompt; acurácia e depois somente texto desempatam. Só o prompt escolhido
é estendido a 20%, reutilizando respostas. Depois, macro-F1 OOF seleciona o
orçamento entre 0%, 5%, 10% e 20%, favorecendo o menor em empates. Ambas as
escolhas são congeladas antes da inferência final.

<a id="gpt-transport-and-accounting"></a>

## Transporte GPT e contabilização

O padrão é **gpt-6.1-sol, low**, via `CodexBackend` existente. Usa sessões novas
e isoladas de `codex exec`, esquema JSON estrito, sem ferramentas, contexto
do repositório, arquivos de gabarito ou continuação da sessão. O `resp_text`
original é enviado sem limpeza nem truncamento. Definições de classes e
instruções são explícitas. O contexto opcional inclui decisões nativas,
probabilidades calibradas e saída do stacking, identificadas como previsões falíveis.

A CLI não expõe temperatura, portanto ela é omitida. Suporta raciocínio baixo
e saída estruturada; veja a [documentação oficial do modelo](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
e o [modo não interativo do Codex](https://learn.chatgpt.com/docs/non-interactive-mode).

O identificador remoto é um alias solicitado. Reutilizar respostas persistidas
permite reprodução exata; novas solicitações futuras podem divergir mesmo com
o mesmo alias e configuração de raciocínio.

SQLite persiste respostas validadas e decisões individuais com chaves de modelo,
raciocínio, instruções, texto original, previsões opcionais e ambiente exatos.
O cache por item permite reutilização entre orçamentos e limites de lotes.
Até oito alvos independentes compartilham uma solicitação Codex; duas podem
executar simultaneamente. Não se copiam rótulos entre alvos. Pedidos exatamente
repetidos compartilham resposta. IDs dos lotes são opacos, sem relação com
IDs de origem ou classes.

Cada tentativa é registrada antes do envio. Retentativas são explicitamente
limitadas, inclusive após tempo esgotado que possa já ter consumido tokens.
Um lote com falha abre um disjuntor para lotes ainda não enviados. Chamadas
falhas não preenchem o cache de decisões válidas. Uma decisão GPT ausente
mantém o rótulo de stacking com estado `fallback` visível; nunca inventa resposta.

Relatórios distinguem chamadas tentadas, decisões válidas por alvo, acertos de
cache, fallback, tokens, soma de latências e tempo total. O custo equivalente
à API usa os preços padrão da configuração: USD 2 entrada, 0.10 entrada em
cache e 10 saída por milhão de tokens. São estimativas da página oficial do
modelo, conferidas em 2026-11-05, **não uma fatura do uso da assinatura ChatGPT**.
Falhas sem consumo disponível são contadas e sinalizadas; o custo desconhecido
não vira zero conhecido silenciosamente. A contabilização por orçamento usa
parcelas por instância dos lotes compartilhados; totais reais do experimento
ficam separados.

<a id="configuration-and-execution"></a>

## Configuração e execução

Use o ambiente Python do projeto com dependências CPU fixadas e Codex CLI
>=0.165.0 autenticado. `codex login` pode fornecer a autenticação ChatGPT
existente. Alternativamente, forneça `CODEX_API_KEY` pelo ambiente da execução.
Não grave credenciais em configurações ou no Git.

`configs/cascade.json` define o experimento. As variáveis opcionais
`CASCADE_MODEL`, `CASCADE_REASONING_EFFORT`, `CASCADE_BATCH_SIZE`,
`CASCADE_WORKERS`, `CASCADE_RETRIES`, `CASCADE_TIMEOUT` e as três
`CASCADE_*_PRICE_PER_MILLION` substituem valores **durante a preparação**.
A configuração resolvida fica então imutável. `CASCADE_CODEX_BIN` seleciona o
executável, e `CASCADE_CACHE`, o cache persistente. A identidade do ambiente
na inferência final deve coincidir com a validação. Os padrões continuam sendo
o modelo e esforço explicitamente solicitados.

Na raiz do EP2, use novos destinos de preparação/publicação/predição:

```bash
PYTHONPATH=src python3 -m cascade_classifier prepare \
  --config configs/cascade.json --output-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier validate-initial \
  --validation-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier validate-extended \
  --validation-dir results/cascade-validation-new

PYTHONPATH=src python3 -m cascade_classifier freeze \
  --validation-dir results/cascade-validation-new \
  --output-dir models/stacking-gpt-cascade-new
```

A preparação reutiliza escores fixados dos membros, sem GPU ou chamadas ao GPT.
Os dois comandos de validação fazem classificação remota. Etapas retomam com
cache/registro originais; mudar entradas exige novo diretório. Uma tentativa
interrompida com consumo desconhecido conta no limite de retentativas.

Para comparação final, reutilize uma execução verificada e concluída de
predição do stacking:

```bash
PYTHONPATH=src python3 -m cascade_classifier predict \
  --model models/stacking-gpt-cascade-new/model.json \
  --input data/test1.xlsx \
  --stacking-run results/ensemble-stacking-qwen-test1-20261105 \
  --compare-budgets --output-dir results/cascade-test1-new

PYTHONPATH=src python3 -m cascade_classifier evaluate \
  --run results/cascade-test1-new \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/cascade-test1-evaluation-new
```

`--compare-budgets` consulta explicitamente a união até o limite congelado de
20%; variantes menores reutilizam essas respostas. Sem a opção, `predict`
usa só o orçamento selecionado na validação e a referência local, evitando
chamadas experimentais desnecessárias. Se a validação escolher zero, a predição
de produção não faz pedidos GPT. Para outra entrada, gere primeiro uma execução
verificada de stacking pelo [fluxo de quatro modelos](ensemble-qwen.md) e
forneça-a aqui.

Os relatórios incluem macro-F1, acurácia, F1 por classe, matrizes de confusão,
instâncias corrigidas/prejudicadas, fallback e recursos consumidos. A avaliação
nunca altera regra, modelo, prompt ou orçamento congelados.
