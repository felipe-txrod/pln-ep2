<a id="final-svm-optimization-manual-offline-execution"></a>

# Otimização final do SVM: execução manual e offline

**Orçamento computacional substituído:** o usuário pediu busca mais rápida após
as variantes sem normalização se tornarem inviavelmente lentas. A execução
antiga foi interrompida e preservada. Use o
[fluxo com orçamento limitado](svm-budgeted-optimization.md) para o comando
ativo; as instruções abaixo documentam o plano original.

Esta investigação em etapas segue o escopo de 2026-11-06: melhorar SVM linear
e sua votação suave com NorBERTo inalterado. Não são permitidos GPT, stacking,
RAG, treino de outros modelos, promoção de padrão ou acesso a teste2. Comandos
longos são executados manualmente pelo usuário.

A implementação atual cobre auditoria de dados, reprodução da referência e
buscas das fases A e B. Retreinamento final, calibração/pesos aninhados,
congelamento e avaliação de teste1 são posteriores; têm protocolo predefinido,
mas **ainda não foram implementados nem executados**. Não apresente a busca
preparada como investigação concluída.

<a id="verified-inputs"></a>

## Entradas verificadas

- `data/train.xlsx` original, SHA-256
  `0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
- `data/ep1-development-holdout.csv` congelado, SHA-256
  `0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.
- Manifesto original de grupos de similaridade, divisão e auditoria em
  `../EP1/results/research/dataset-hypotheses-20260923/`.
- SVM original em `models/svm/`; referência e composição dos papéis salvas em
  `models/ensemble-soft-qwen/`. Só seus membros SVM/NorBERTo contribuem.
- Margens salvas do SVM e probabilidades NorBERTo em
  `results/ensemble-soft-qwen-20261105/scores/`. A preparação reproduz as 3999
  margens e a matriz de confusão/macro-F1 da referência nas 1972 linhas.

O diretório da referência inclui metadados históricos de LR/Qwen com peso zero;
a pesquisa não carrega, treina ou avalia esses modelos. A temperatura existente
do NorBERTo e todas as probabilidades permanecem inalteradas. O SVM antigo
nunca é reajustado.

<a id="preparation"></a>

## Preparação

No EP2, com um **novo** diretório de saída:

```bash
bash scripts/svm/research.sh --output-dir results/svm-final-20261106 prepare
```

A preparação audita 16093/2027/1972 linhas, IDs, hashes, rótulos, duplicatas
normalizadas, componentes originais de quase duplicatas e sua separação.
Congela protocolo, desenho da busca, ambiente, implementação e identidades das
entradas em `plan.json`; grava `audit.json`, `baseline.json`, composição completa
do treino e `report.md`. Não abre planilhas de teste1/teste2.

<a id="phase-a-broad-search"></a>

## Fase A: busca ampla

```bash
bash scripts/svm/research.sh --output-dir results/svm-final-20261106 search --phase a
```

São 256 candidatos e 768 ajustes. Quatro famílias usam oito desenhos
fracionários fixos cada, cruzados com quatro valores logarítmicos de C e dois
pesos de classe. TF-IDF aprende exclusivamente nos folds de treino de
StratifiedGroupKFold (três folds, semente 42). As configurações incluem a
referência original de palavras+char_wb. Todas as combinações exatas ficam em
`phase-a-candidates.json`; é um desenho limitado, não o produto cartesiano
exaustivo de todos os parâmetros.

O ambiente Python 3.12 local fixado usa pacotes existentes de
`.cache/cascade-runtime` e `.cache/rag-runtime`: numpy 2.3.3, scipy 1.16.2,
scikit-learn 1.7.2, joblib 1.5.2. Dois processos e uma thread numérica por ajuste
limitam a pressão de memória. Processos/cache são congelados na preparação;
para mudar, edite a configuração e prepare novo diretório. O cache joblib é
reduzido a 8 GiB entre representações; ajustes ativos podem exceder esse limite
temporariamente. O cache fica apenas neste experimento novo.

O comando pode levar tempo considerável; ainda não há duração medida para a
busca completa. Imprime progresso, salva cada fold com checksum e retoma com
o mesmo comando. Execuções simultâneas no mesmo diretório são recusadas. Não
edite fontes, ambiente, configuração ou entradas históricas congeladas durante
a execução. Folds interrompidos são recalculados; concluídos são verificados
e reutilizados. Avisos de convergência excluem candidatos; ajustes inválidos
interrompem o comando, sem fabricar métricas nem descartar observações silenciosamente.

<a id="phase-b-bounded-refinement"></a>

## Fase B: refinamento limitado

Após auditar os resultados concluídos da fase A:

```bash
bash scripts/svm/research.sh --output-dir results/svm-final-20261106 search --phase b
```

As três melhores representações distintas definem até 61 candidatos. C local
logarítmico, pesos de classe, solucionadores compatíveis hinge/squared-hinge e
pesos dos ramos de palavras/caracteres são avaliados em cinco folds agrupados,
semente 143. A referência original é mantida. O comando não executa calibração,
seleção de pesos, leitura de rótulos finais ou inferência de outros modelos.

As duas fases usam macro-F1 e registram dispersão entre folds, métricas por
classe, matrizes de confusão, tempos de ajuste/avaliação, convergência e número
de atributos. A regra de um erro-padrão favorece C menor e menos atributos
até um erro-padrão da melhor média. É uma heurística conservadora; reutilizar
observações nas duas fases ainda cria viés adaptativo de seleção. Os resultados
não são estimativas imparciais de validação cruzada aninhada.

<a id="outputs-and-remaining-stages"></a>

## Saídas e etapas restantes

`phase-{a,b}/search.json` contém todas as configurações e resultados dos folds;
`trials/<id>/fold-*.npz` preserva IDs e previsões de validação. `report.md` e
`trials.csv` por fase são gerados após a conclusão. Para regenerar sem experimentos:

```bash
bash scripts/svm/research.sh --output-dir results/svm-final-20261106 report
```

O protocolo congelado registra a comparação aninhada posterior de sigmoide/
isotônica nas 2027 linhas de calibração, seleção agrupada aninhada de cinco
pesos simples do SVM nas 1972 linhas de meta-validação, critérios conservadores
de adoção e bootstrap pareado por grupos. Reporte separadamente qualidade OOF
da política aninhada e diagnósticos de seleção de candidatos. A/B/C devem ser
congelados antes de novo acesso a teste1; ele não muda a decisão do desenvolvimento.
Preserve todo artefato histórico e o padrão atual, inclusive com resultado
negativo. A referência original de teste1 é macro-F1 0.49584421663197825, a
conferir com previsões exatas somente após congelar.

As 3999 linhas são dados de desenvolvimento historicamente reutilizados. Os
grupos originais de quase duplicatas foram formados sem rótulos no corpus
original completo e depois congelados; reutilizá-los não garante separação
semântica. Teste1 já foi exposto e tem sobreposição conhecida com treino;
preserve todas as linhas e trate a avaliação futura como retrospectiva. O
pedido atual mantém teste2 fora do desenvolvimento e conflita com a ordem
original teste2 antes de teste1 do enunciado; essa limitação é registrada sem
abrir teste2.
