<a id="rag-context-pilot-hybrid-retrieval-with-three-examples-selected"></a>

# Piloto de contexto RAG: recuperação híbrida com três exemplos selecionada

[Métricas agregadas e consumo](data/rag-cascade-pilot.json) · [Protocolo e comandos](../guides/rag-stacking.md) · [Registro original de preparação](rag-cascade-preparation.md)

Acompanhamento: a [confirmação concluída](rag-cascade-confirmation.md) selecionou
roteamento zero ao GPT; o ganho abaixo não persistiu. Este relatório preserva
o marco exploratório anterior e sua escolha original.

O piloto iniciado manualmente concluiu os **14 braços nos mesmos 48 alvos OOF
difíceis**, com 672 classificações válidas e nenhum fallback final. O melhor
braço observado foi **recuperação híbrida com três exemplos e previsões locais**:
19/48 acertos, acurácia 39.58%, macro-F1 36.90%, contra stacking com 16/48,
33.33% e 27.59%. Corrigiu oito erros e prejudicou cinco acertos do stacking.

A média pareada predefinida selecionou **recuperação híbrida com três exemplos**
para possível confirmação posterior. Isso escolhe só a receita de recuperação.
Estratégia de contexto e orçamento não foram confirmados; nenhum modelo RAG
foi congelado ou promovido. O padrão da cascata sem GPT não muda.

<a id="all-pilot-arms"></a>

## Todos os braços do piloto

Todas as linhas abaixo usam os mesmos 48 alvos. São casos difíceis selecionados,
não desempenho global reservado ou de teste. Custos vêm dos registros reais
de tentativas; alocações por item reutilizadas não são somadas ao total.

| Contexto | Acertos | Acurácia | Macro-F1 | Corrigidos | Prejudicados | USD estimados |
|---|---:|---:|---:|---:|---:|---:|
| Stacking | 16/48 | 33.33% | 27.59% | — | — | 0 |
| text | 14/48 | 29.17% | 26.30% | 9 | 11 | 0.102298 |
| local | 18/48 | 37.50% | 33.07% | 6 | 4 | 0.123320 |
| tfidf-3-rag | 17/48 | 35.42% | 32.25% | 12 | 11 | 0.145720 |
| tfidf-3-local | 14/48 | 29.17% | 25.41% | 7 | 9 | 0.150175 |
| tfidf-5-rag | 16/48 | 33.33% | 30.17% | 9 | 9 | 0.164544 |
| tfidf-5-local | 15/48 | 31.25% | 28.65% | 6 | 7 | 0.168521 |
| semantic-3-rag | 11/48 | 22.92% | 19.72% | 6 | 11 | 0.127561 |
| semantic-3-local | 14/48 | 29.17% | 26.27% | 3 | 5 | 0.145309 |
| semantic-5-rag | 15/48 | 31.25% | 27.82% | 10 | 11 | 0.150446 |
| semantic-5-local | 14/48 | 29.17% | 25.41% | 5 | 7 | 0.179230 |
| hybrid-3-rag | 16/48 | 33.33% | 29.91% | 10 | 10 | 0.116455 |
| hybrid-3-local | 19/48 | 39.58% | 36.90% | 8 | 5 | 0.117164 |
| hybrid-5-rag | 15/48 | 31.25% | 29.78% | 11 | 12 | 0.148044 |
| hybrid-5-local | 14/48 | 29.17% | 25.60% | 5 | 7 | 0.184201 |

Nomes terminados em `rag` têm exemplos recuperados sem previsões locais;
braços RAG `local` têm ambos. `text` é somente texto; o braço simples `local`
tem previsões locais sem exemplos. Cada método RAG compara três e cinco demonstrações.

<a id="what-helped-in-this-sample"></a>

## O que ajudou nesta amostra

- Previsões locais melhoraram GPT somente com texto de 14 para 18 acertos,
  macro-F1 de 26.30% para 33.07%.
- Adicionar três demonstrações híbridas às previsões locais trouxe mais um
  acerto, macro-F1 de 33.07% para 36.90%.
- O melhor braço só RAG foi TF-IDF com três demonstrações: 17 acertos,
  macro-F1 32.25%. Híbrido só RAG teve 16 acertos e macro-F1 29.91%.
- Cinco demonstrações não superaram a receita híbrida selecionada de três.
  Mais contexto não foi consistentemente melhor; recuperação só semântica
  foi particularmente fraca.
- O melhor braço combinado ainda recuperou apenas 3/17 casos verdadeiros c1.
  As três classes melhoraram em F1 frente ao stacking, mas c1 segue uma
  limitação importante.

Essas observações não identificam mecanismo causal para erros individuais.
Vocabulário ou assunto parecidos podem não transmitir uma nota subjetiva de
clareza; isso continua hipótese, não explicação medida.

<a id="per-class-results-and-confusion-matrices"></a>

## Resultados por classe e matrizes de confusão

| Sistema | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| Stacking | 16.67% | 47.06% | 19.05% |
| Texto | 10.00% | 38.89% | 30.00% |
| Local | 17.39% | 51.06% | 30.77% |
| Híbrido 3 + local | 25.00% | 50.00% | 35.71% |

Linhas são classes verdadeiras; colunas, preditas em ordem `c1`, `c234`, `c5`.

Stacking:

```text
 2 10  5
 5 12  4
 0  8  2
```

Híbrido 3 + local:

```text
 3  7  7
 4 11  6
 0  5  5
```

<a id="resources-and-cli-update-recovery"></a>

## Recursos e recuperação após atualização da CLI

O piloto inteiro usou **84 chamadas válidas em lotes** e **três tentativas
falhas arquivadas**, totalizando 87, dentro do limite original de 128.
A retomada final fez dez novas chamadas válidas e reutilizou 592 classificações
em cache. Uma atualização Codex mudou o executável padrão de 0.166.0 para
0.167.0 durante a primeira invocação. Retomar com o executável original
versionado 0.166.0 restaurou os metadados exatos do plano; prompts, identidades
das fontes, modelos-base e rótulos não mudaram.

- Consumo registrado: **1,065,917 tokens de entrada**, incluindo **147,840 em
  cache**, e **17,205 de saída**.
- Estimativa total equivalente à API padrão: **USD 2.022988**, pelos preços
  configurados USD 2 entrada / 0.10 entrada em cache / 10 saída por milhão.
  Não é fatura da assinatura ChatGPT.
- Latência média válida: **7.93 segundos**. Soma das latências válidas:
  **666.49 segundos**; pela simultaneidade, não é o tempo total decorrido.
- Falhas arquivadas acrescentaram 0.55 segundos de latência medida e têm
  consumo desconhecido. Zero tokens registrados não provam consumo remoto
  zero; a estimativa exclui o desconhecido.
- Relatórios da invocação final contêm tempos de reuso do cache para braços
  anteriores. Não há tempo cumulativo do piloto entre interrupções/retomadas.
- Nenhuma chamada GPT foi feita para avaliar ou verificar o piloto.

Os totais oficiais somam uma vez os 84 históricos ativos válidos e os três
históricos falhos arquivados. Estimativas amortizadas por braço podem reutilizar
lotes anteriores e não devem ser somadas como novas cobranças. A recuperação
preserva falhas, saídas parciais, checksums e notas de recursos.

<a id="isolation-verification-and-limits"></a>

## Isolamento, verificação e limites

O banco original de 16,093 linhas exclui 3,999 reservadas. Filtros/deduplicação
deixam 11,562 exemplos indexados. Todas as demonstrações vieram dele, com
três/cinco exemplos completos em cada alvo RAG. Stacking congelado de quatro
membros e previsões aninhadas OOF foram reutilizados sem retreinar bases.
A verificação prévia independente conferiu os 672 prompts: texto literal do
alvo, rótulos válidos só de treino, evidência local, duplicatas e diversidade.

Os 14 conjuntos de previsões estavam completos e verificados antes da abertura
do gabarito de validação. Cálculo independente da biblioteca padrão reproduziu
matrizes, acurácia, macro-F1 e correções/prejuízos. A coorte de confirmação de
1,884 linhas permanece sem uso e sem grupos compartilhados com o piloto.
Teste1 e teste2 não foram acessados pelo experimento.

O melhor braço observado ganha três acertos frente ao stacking. O p-valor
exato bilateral descritivo de McNemar é **0.5811**, antes de considerar a busca
em 14 braços. A pequena amostra exploratória **não** estabelece melhora
significativa de generalização. O protocolo OOF aninhado exclui o ajuste nos
próprios dados de avaliação do meta-modelo; a exposição herdada do checkpoint
Qwen ao conjunto reservado permanece.

Mais classificação exige autorização explícita. Não foi iniciada confirmação,
novo pacote RAG congelado, entrega teste2 ou avaliação retrospectiva teste1.

Evidência local:
`results/rag-context-pilot-20261106/pilot/{plan,run,selection,independent-summary}.json`,
solicitações/contextos/previsões/tentativas por braço,
`runtime-recovery-20261106/` e caches persistentes de respostas/embeddings.
Evidência bruta é local e ignorada pelo Git; o agregado vinculado preserva
métricas públicas sem textos ou rótulos individuais.
