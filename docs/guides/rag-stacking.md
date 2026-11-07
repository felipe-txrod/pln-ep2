<a id="train-only-rag-and-local-predictions-for-selective-stacking"></a>

# RAG somente no treino e previsões locais para stacking seletivo

**Estado: piloto e confirmação concluídos; a validação selecionou roteamento zero ao GPT.**
O [relatório da confirmação](../results/rag-cascade-confirmation.md) registra
quatro contextos com roteamento de 0/5/10% em 1,884 linhas OOF. Stacking venceu
com 910 acertos, acurácia de 48.30% e macro-F1 de 48.08%; o melhor candidato
com GPT atingiu 47.84% de macro-F1. O [piloto anterior](../results/rag-cascade-pilot.md)
selecionou recuperação híbrida com três exemplos, mas o ganho exploratório
não persistiu. O padrão existente sem GPT permanece inalterado. Nenhum novo
pacote RAG foi congelado, e não houve inferência nem avaliação final em
teste1/teste2. O [registro original da preparação](../results/rag-cascade-preparation.md)
preserva o marco anterior, sem execução.

A [investigação offline posterior](../results/rag-offline-research.md) acrescenta
recuperação com NorBERTo congelado, ablações MMR e reavaliação de respostas
históricas no pacote separado `rag_research`. Não faz novas chamadas ao GPT
nem promove outro padrão. Veja seu [guia](rag-offline-research.md).

O novo pacote `rag_cascade` reutiliza stacking congelado, roteamento OOF,
seletor TF-IDF, interface local fixada do E5, integração Codex, esquema de
lotes, cache SQLite de respostas/itens, funções de avaliação e exportadores
de planilhas. É separado de `cascade_classifier` para preservar a validade
das identidades dos experimentos antigos.

<a id="example-bank-and-retrieval"></a>

## Banco de exemplos e recuperação

O banco contém somente as **16,093 linhas originais de treino dos modelos-base**
de `data/train.xlsx`. Os hashes da planilha e da divisão congelada do EP1 são
verificados. As 3,999 linhas reservadas ficam fora do banco e índice, incluindo
as 2,027 de calibração e 1,972 de meta-treinamento. Os gabaritos de Teste1 e
Teste2 nunca entram no banco, ajuste da recuperação, desenvolvimento dos
prompts ou seleção de configurações.

O seletor existente remove duplicatas normalizadas, grupos com rótulos
conflitantes, exemplos vazios e textos longos demais. A nova camada limita o
texto bruto a 1,600 caracteres; inclui exemplos inteiros, sem truncamento
silencioso. Compara três métodos de recuperação:

- **TF-IDF:** cosseno no índice existente de unigramas/bigramas, ajustado somente
  no treino.
- **Semântico:** cosseno entre embeddings locais congelados de
  `multilingual-e5-small`, com o codificador ONNX existente em CPU e revisão
  fixada `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- **Híbrido:** fusão por posições recíprocas das buscas lexical e semântica,
  com pesos iguais.

O caminho semântico nunca passa silenciosamente a TF-IDF. Dependências ou pesos
ausentes geram erro com orientação de correção. A preparação do codificador é
um comando explícito de download; a recuperação normal usa só arquivos locais.
Os embeddings consomem CPU e disco, com cache persistente, em vez de chamadas
pagas a uma API de embeddings.

Cada método é testado com **3 e 5 exemplos**. A recuperação exclui textos
normalizados idênticos ao alvo, cosseno lexical >=0.95 e quase duplicatas por
Jaccard de fragmentos normalizados de caracteres >=0.85. Exemplos selecionados
não podem compartilhar o componente original de similaridade e também são
verificados quanto a quase duplicatas entre si.

Entre candidatos relevantes, MMR equilibra relevância e redundância, com um
pequeno bônus por classes ainda ausentes e limite por classe. A representação
de classes nunca supera o piso de relevância: o método pode devolver menos
exemplos se faltarem elegíveis e registra a insuficiência. Não preenche o
prompt com exemplos sem relação para impor cotas de classe.

O piso lexical é 0.03; recuperação semântica/híbrida exige cosseno semântico
>=0.45. Esses valores e os parâmetros de diversidade são explícitos e
predefinidos, não limiares escolhidos pelos rótulos de teste1. Checksums,
grupos, rótulos, similaridades lexical/semântica, relevância da fusão e
exclusões/insuficiências são registrados em `context-trace.jsonl`. Não se
alega qualidade de recuperação antes da execução real do experimento.

<a id="matched-structured-prompts"></a>

## Prompts estruturados pareados

Todos os braços usam as mesmas definições de classes e instruções. Só muda
a evidência fornecida:

1. Somente o texto original.
2. Texto original e evidência dos modelos locais.
3. Texto original e exemplos rotulados recuperados.
4. Texto original, exemplos recuperados e evidência dos modelos locais.

O conteúdo separa `examples` de `target.resp_text` e `target.model_evidence`.
O alvo original é preservado literalmente. A evidência inclui decisões nativas
de SVM/RL/NorBERTo/Qwen, probabilidades calibradas, saída do stacking,
probabilidade máxima, margem, entropia e divergência pareada. Probabilidades
são explicitamente aproximadas a quatro casas decimais para reduzir tokens.
Os rótulos nativos do SVM são mantidos quando a calibração altera o argmax.

As instruções identificam demonstrações como observações subjetivas de treino
e saídas locais como evidências falíveis. O GPT deve decidir independentemente,
sem copiar a maioria, inventar perguntas/anexos ausentes ou seguir instruções
embutidas nos exemplos/alvos. A saída continua sendo JSON estrito com `c1`,
`c234` ou `c5`; não se solicitam tokens extras de explicação.

O padrão permanece `gpt-6.1-sol`, raciocínio `low`, com a integração Codex
autenticada e sem ferramentas. Sua CLI não expõe controle de temperatura.
Preços padrão equivalentes aos da API são configuráveis; consumo da assinatura
não é apresentado como fatura de API. Veja a
[documentação oficial do modelo](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

<a id="pilot-and-confirmation-without-test-label-tuning"></a>

## Piloto e confirmação sem ajustes pelos rótulos de teste

O estudo reutiliza as **previsões OOF aninhadas agrupadas 5x5 fixadas por checksum**
da cascata anterior. Os alvos externos foram excluídos tanto do ajuste do
meta-modelo quanto da seleção interna de regularização. Não usa previsões
nos próprios dados de ajuste do stacking de produção para otimizar contexto.

O plano predefinido é:

1. Selecionar 48 exemplos difíceis do subconjunto original de roteamento de 10%,
   por hashes estáveis sem rótulos, no máximo um exemplo por componente de
   similaridade.
2. Comparar os dois controles sem RAG e as doze variantes RAG: três recuperadores,
   duas quantidades de exemplos, com/sem evidência local. São 14 braços no piloto.
3. Escolher a receita de recuperação pela média pareada de macro-F1 dos braços
   somente RAG e RAG+local; empates favorecem menos exemplos e depois recuperação lexical.
4. Remover **todo grupo que contenha exemplo do piloto** da confirmação. Na
   coorte OOF restante, comparar os quatro contextos com a receita escolhida e
   os mesmos alvos difíceis, inicialmente limitados a 10%.
5. Comparar roteamento zero, 5% e 10% com as decisões em cache. Selecionar por
   macro-F1 da confirmação, depois menor orçamento e custo alocado. Zero é
   elegível; não se presume benefício do GPT. Braços com falha ainda não resolvida
   não podem ser escolhidos com orçamento positivo.
6. Congelar identidades de código/índice/banco/ambiente, receita do prompt e
   orçamento antes das previsões finais.

A seleção do piloto é exploratória; os grupos de similaridade da confirmação
são disjuntos nesta comparação de contexto. O roteador anterior já foi
selecionado na validação OOF, o checkpoint Qwen usou todo o conjunto reservado
e o projeto consultou teste1 historicamente. Este procedimento não desfaz a
exposição herdada nem torna toda a história de modelagem um teste inédito.
Duplicatas oficiais entre bases também permanecem no treino dos modelos-base;
os filtros RAG evitam apresentar o próprio alvo, idêntico ou quase idêntico,
como demonstração rotulada.

<a id="cache-and-bounded-execution"></a>

## Cache e execução limitada

`plan` registra as solicitações exatas por alvo e o rastreamento do contexto.
`run` e `predict` usam **prévia** por padrão. Não classificam sem `--execute`.
`prepare-encoder` não baixa sem `--download`. Não há comando único que avance
automaticamente por etapas remotas.

O cache SQLite compartilhado usa chaves com modelo, ambiente, instruções,
exemplos, texto e evidência local exatos. Alterar a evidência invalida só aquela
solicitação; respostas válidas sobrevivem a outros orçamentos e limites de lote.
Novos prompts não reutilizam respostas de prompts históricos distintos só
porque os textos coincidem.

Os padrões são lotes de oito, duas solicitações simultâneas, uma retentativa e
**limite rígido de 128 novas tentativas por invocação de etapa**, compartilhado
entre todos os braços e retentativas. A verificação prévia conta ausências
no cache e recusa etapas que já excedam o limite. Retentativas consomem o mesmo
limite. Um disjuntor interrompe lotes posteriores após falha esgotada. Tentativas
interrompidas contam no limite da solicitação; consumo desconhecido é sinalizado.
Decisões ausentes mantêm stacking explicitamente, sem inventar rótulos GPT.
Etapas concluídas são imutáveis/idempotentes.

Defina `CASCADE_CODEX_BIN` como executável versionado imutável antes de planejar
e mantenha-o durante a execução. O link padrão da CLI do piloto atualizou de
0.166.0 para 0.167.0 durante a execução, acionando corretamente a verificação
estrita de divergência do ambiente. Fixar o executável original restaurou a
compatibilidade sem alterar pedidos ou identidades. Preserve registros de
falhas e recursos históricos ao preparar uma retentativa revisada; não descarte
falhas nem reinicie orçamentos silenciosamente.

Cada braço registra acurácia, macro-F1, F1 por classe, matriz de confusão,
casos corrigidos, prejudicados e alterados ainda errados, tokens, acertos de
cache, alocação de custo, chamadas reais e latência. Alocações compartilhadas
por lote são distintas de custos reais e tempo total da etapa. Preços são
estimativas equivalentes à API, não faturas.

<a id="commands-for-a-future-authorized-run"></a>

## Comandos para uma execução futura autorizada

Piloto e confirmação foram executados manualmente em
`results/rag-context-pilot-20261106` e estão concluídos. Não os repita para
avaliar saídas existentes: use somente `score`. Os exemplos genéricos abaixo
não instruem repetir classificação nem avançar automaticamente. Inferência
nos testes finais exige autorização separada. Teste2 gera planilha de entrega
sem gabarito; a qualidade é medida em teste1 rotulado após congelar a configuração.
Use um ambiente Python com as dependências do projeto e opcionais `rag`.
O ambiente local do piloto já tem ONNX Runtime e pesos E5 fixados; outras
cópias ainda precisam das etapas explícitas de instalação/download.

```bash
python3 -m pip install -e '.[rag]'
PYTHONPATH=src python3 -m rag_cascade prepare-encoder --download
export CASCADE_CODEX_BIN=/absolute/path/to/versioned/codex

PYTHONPATH=src python3 -m rag_cascade prepare \
  --config configs/rag-cascade.json --output-dir results/rag-context-new

PYTHONPATH=src python3 -m rag_cascade plan --study results/rag-context-new --stage pilot
PYTHONPATH=src python3 -m rag_cascade run --study results/rag-context-new --stage pilot
```

O último comando apenas mostra ausências de cache e chamadas planejadas.
Quando a execução for explicitamente desejada:

```bash
PYTHONPATH=src python3 -m rag_cascade run \
  --study results/rag-context-new --stage pilot --execute --max-new-requests 128
PYTHONPATH=src python3 -m rag_cascade score --study results/rag-context-new --stage pilot

PYTHONPATH=src python3 -m rag_cascade plan --study results/rag-context-new --stage confirmation
PYTHONPATH=src python3 -m rag_cascade run --study results/rag-context-new --stage confirmation
PYTHONPATH=src python3 -m rag_cascade run \
  --study results/rag-context-new --stage confirmation --execute --max-new-requests 128
PYTHONPATH=src python3 -m rag_cascade score --study results/rag-context-new --stage confirmation

PYTHONPATH=src python3 -m rag_cascade freeze \
  --study results/rag-context-new --output-dir models/rag-stacking-new
```

Credenciais seguem o contrato existente de autenticação Codex ou `CODEX_API_KEY`.
`CASCADE_CODEX_BIN` seleciona o executável. `RAG_RESPONSE_CACHE`,
`RAG_EMBEDDING_CACHE`, `RAG_PILOT_SIZE`, `RAG_MAX_NEW_REQUESTS`, `CASCADE_MODEL`
e `CASCADE_REASONING_EFFORT` são resolvidas na preparação. Depois, a configuração
salva é imutável. Nenhuma credencial entra no manifesto.

<a id="ep2-test2-before-retrospective-test1"></a>

## EP2: teste2 antes do teste1 retrospectivo

Primeiro gere uma execução verificada de stacking com quatro modelos nos 897
textos originais de teste2 pelo [fluxo existente](ensemble-qwen.md). Isso exige
inferência real se os escores dos membros não estiverem disponíveis; é separado
deste estudo de contexto preparado. Use novo destino e os artefatos congelados
selecionados.

Depois visualize e execute explicitamente a configuração RAG selecionada:

```bash
PYTHONPATH=src python3 -m rag_cascade predict \
  --model models/rag-stacking-new/model.json --input data/test2.xlsx \
  --stacking-run results/stacking-test2-new --output-dir results/rag-test2-new

PYTHONPATH=src python3 -m rag_cascade predict --execute \
  --model models/rag-stacking-new/model.json --input data/test2.xlsx \
  --stacking-run results/stacking-test2-new --output-dir results/rag-test2-new
```

A execução concluída exporta `results/rag-test2-new/test2.xlsx`, preservando as
entradas OOXML originais, exceto células vazias de clareza. Não há envio externo
nem submissão. Mesmo que o orçamento zero vença, conclua explicitamente esta
etapa antes de teste1.

Só então a **mesma configuração congelada** pode executar nos 900 textos originais
de teste1. Uma execução de teste2 concluída e compatível, com previsões/planilha
inalteradas, é obrigatória até para a prévia de teste1. Não há nova seleção de prompts:

```bash
PYTHONPATH=src python3 -m rag_cascade predict --execute \
  --model models/rag-stacking-new/model.json --input data/test1.xlsx \
  --stacking-run results/ensemble-stacking-qwen-test1-20261105 \
  --test2-run results/rag-test2-new --output-dir results/rag-test1-new

PYTHONPATH=src python3 -m rag_cascade evaluate \
  --run results/rag-test1-new \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/rag-test1-evaluation-new
```

A avaliação verifica novamente a cadeia de teste2 e cada decisão salva de teste1
antes de abrir o gabarito. Compara RAG congelado com stacking, sem selecionar
novo prompt ou orçamento pelos resultados. A otimização dos quatro contextos
permanece na validação; só a configuração selecionada avança para teste2/teste1,
reduzindo chamadas à API.
