<a id="compute-bounded-svm-optimization"></a>

# Otimização do SVM com orçamento computacional limitado

Este fluxo substitui o [executor da busca original](svm-final-optimization.md)
após o pedido de redução do tempo. A busca original foi interrompida com SIGINT;
os 129 folds concluídos, fontes/configuração, planos e cache foram preservados.
Não retome a busca de 256 candidatos.

<a id="changes-to-the-computational-budget"></a>

## Alterações do orçamento computacional

| Parâmetro | Original | Revisado |
|---|---:|---:|
| Candidatos da fase A | 256 | 96 |
| Folds da fase A | 768 | 288, incluindo 72 reutilizados |
| Novos ajustes da fase A | 639 restantes na execução antiga | 216 |
| Limite de candidatos da fase B | 61 | 25 |
| Limite de folds da fase B | 305 | 125 |
| Ajustes numéricos simultâneos | 2 | 4 |
| Limite de tempo total por ajuste | Sem limite | 120 segundos |

O Ryzen 5 5600 original tem seis núcleos físicos/doze CPUs lógicas e cerca de
16 GiB de RAM. Quatro ajustes simultâneos, com uma thread numérica cada, usam
mais CPU mantendo folga de memória. Doze processos de treinamento esparso
poderiam pressionar a memória e gerar custo de swap; o novo padrão é
intencionalmente quatro.

A fase A reduzida mantém quatro famílias de atributos (palavras, char, char_wb
e palavras+char_wb), desenhos fracionários 0/1/3/4 normalizados em L1/L2,
C=.005/.05/.5, ambos os pesos de classe, TF binário/sublinear, variações de
ngramas/min_df/max_df e a referência original exata. Exclui atributos sem
normalização, três outros desenhos fracionários e C=5. A fase B refina duas
representações distintas com três valores locais de C, pesos de classe,
verificações hinge e pesos dos ramos da mistura.

São **alterações de orçamento computacional**, motivadas por tempo/convergência
medidos e pelo pedido explícito do usuário. Desempenho dos candidatos e rótulos
finais não selecionam os desenhos reduzidos. Variantes excluídas não são
declaradas estatisticamente inferiores. O resultado é um estudo adaptativo de
desenvolvimento revisado, não a conclusão da grade original inteira.

<a id="shared-sparse-features-and-hard-limits"></a>

## Atributos esparsos compartilhados e limites rígidos

O novo motor ajusta o transformador original exatamente uma vez por representação
e fold de treino, e transforma a validação uma vez. As escolhas de C, perda e
pesos de classe compartilham as mesmas matrizes esparsas mapeadas em memória,
somente para leitura. Nenhum texto de validação entra no ajuste. Isso preserva
a semântica do pipeline local a cada fold, evitando repetir transformações e
cópias de carregamento dos atributos. Testes sintéticos comparam previsões e
margens ao pipeline original com tolerância 1e-12.

O solucionador nativo executa em subprocesso supervisionado. O limite de 120
segundos cobre inicialização, carregamento das matrizes, ajuste e avaliação.
Ao esgotá-lo, apenas esse subprocesso é encerrado; o fold recebe `timeout`,
sem previsão fabricada, e torna o candidato inelegível. Tempos esgotados são
terminais na retomada. Avisos de convergência também tornam candidatos
inelegíveis. Outras falhas interrompem visivelmente a execução. Os valores
originais max_iter=20000/tol/semente permanecem, permitindo reutilizar ajustes
antigos idênticos. O sinal Linux de morte do pai encerra filhos dos modelos
se seus supervisores forem interrompidos. A construção dos atributos não tem
limite próprio; os custos observados foram muito menores que os tempos
patológicos do solucionador. O cache esparso é exclusivo da nova execução e
pode crescer nas duas fases.

Resultados antigos só são copiados quando coincidem estimador/atributos,
ordem de classes, semente, ambiente, IDs de treino/validação e identidades de
rótulos/textos. Cada NPZ importado é conferido por checksum, cobertura completa
das linhas e índices de classes; suas métricas são reconstruídas antes da
importação. `reuse.json` registra cada fold histórico importado/excluído e seu
checksum de origem. Não modifique a execução antiga enquanto a revisada estiver ativa.

<a id="manual-commands"></a>

## Comandos manuais

No EP2, a preparação já foi concluída em `results/svm-final-budgeted-20261106/`.
Importou 72 folds compatíveis e preservou os 129 históricos. Não repita a
preparação nesse diretório.

A fase A está concluída: 96/96 candidatos, 288/288 folds, sem tempo esgotado ou
aviso de convergência, com verificação independente de previsões/métricas aprovada.
Não a execute novamente. O comando original de execução/retomada foi:

```bash
bash scripts/svm/research-budgeted.sh \
  --output-dir results/svm-final-budgeted-20261106 search --phase a
```

O comando imprime cada novo resultado de fold e o progresso dos candidatos.
O mesmo comando retoma folds concluídos/importados/com tempo esgotado sem
recalculá-los. Caminhos, ambiente, fontes, resultados originais e configuração
revisada têm identidades fixadas. Não os edite durante a execução. Relatórios
novos ficam em `phase-a/trials/<candidate-id>/`; metadados das matrizes e tempos
de construção ficam em `cache/phase-a/<representation>/fold-*/features.json`.
`progress.json` atualiza quando retorna um grupo representação/fold; relatórios
individuais e arquivos de estado dos processos detalham a situação em tempo real.

A fase B também está concluída e verificada independentemente: 20 candidatos /
100 ajustes, sem tempo esgotado ou aviso de convergência. Não repita nenhuma
busca. O comando original de execução/retomada foi:

```bash
bash scripts/svm/research-budgeted.sh \
  --output-dir results/svm-final-budgeted-20261106 search --phase b
```

Um verificador independente somente de leitura confere previsões salvas sem
chamar fit/predict nem abrir planilhas finais. Use outro nome de saída:

```bash
PYTHONPATH=.cache/cascade-runtime:.cache/rag-runtime:src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 scripts/svm/verify-search.py \
  --run results/svm-final-budgeted-20261106 --phase a \
  --output results/svm-final-budgeted-20261106/verification-phase-a-new.json
```

A verificação concluída já está em `verification-phase-a.json`. A da fase B
está em `verification-phase-b.json`. Os [resultados](../results/svm-budgeted-optimization.md)
distinguem o candidato de maior média da escolha fixa de um erro-padrão.
Os pequenos ganhos da fase A são diagnósticos adaptativos de desenvolvimento,
não estabelecem melhora do ensemble calibrado final.

A próxima etapa ativa é o [fluxo independente de finalização](svm-finalization.md):
um retreinamento do SVM selecionado, calibração e pesos aninhados e congelamento
de A/B/C antes da avaliação final separada.

Para outra execução nova, a preparação é:

```bash
bash scripts/svm/research-budgeted.sh --output-dir results/svm-budgeted-new prepare
```

O ambiente local Python 3.12 fixado e os caches existentes não mudam; não é
necessário pacote de GPU, chamada de rede ou download de modelo. O pacote de
pesquisa original do SVM também fica inalterado, mantendo seu plano verificável.

<a id="downstream-protocol-and-evidence"></a>

## Protocolo posterior e evidências

As 16093/2027/1972 linhas, sementes da validação cruzada agrupada e regra original
de um erro-padrão permanecem inalteradas. Calibração aninhada, seleção de pesos,
critérios de adoção no desenvolvimento, congelamento de A/B/C e avaliação
pareada retrospectiva em Teste1 ainda estão pendentes. Esta revisão não ajusta
nem executa inferência com NorBERTo/LR/Qwen, GPT, stacking ou RAG; não muda
padrões nem acessa planilhas de Teste1 ou Teste2.

Os novos `report.md`, `plan.json` e `reuse.json` explicitam a revisão e preservam
as ressalvas metodológicas originais. Tempos importados incluem custo de
atributos/cache do pipeline antigo; os novos tempos de ajuste medem trabalho
do classificador, enquanto a construção compartilhada é registrada uma vez
por representação/fold. Esses tempos não são um teste controlado de hardware.
A redução do tempo total ainda não foi medida na busca nova completa.
