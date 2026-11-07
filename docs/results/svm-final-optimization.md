<a id="final-svm-investigation-audited-preparation-search-pending"></a>

# Investigação final do SVM: preparação auditada, busca pendente

**Registro histórico de preparação.** A busca manual posterior foi interrompida
após 129 folds a pedido do usuário. A [revisão ativa do orçamento](svm-budgeted-optimization.md)
preserva a execução e reutiliza 72 folds compatíveis. As afirmações abaixo
registram o estado anterior ao início manual original.

Preparado em 2026-11-06. Este é **registro parcial da investigação**, não
resultado de otimização concluída. O usuário inicia comandos longos manualmente
e avisa ao concluir antes da etapa seguinte.

<a id="audited-data-and-original-model"></a>

## Dados auditados e modelo original

SHA-256 da fonte original:
`0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
SHA-256 da divisão congelada:
`0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.

| Papel | Linhas | Grupos históricos de similaridade |
|---|---:|---:|
| Treinamento original | 16093 | 12956 |
| Calibração | 2027 | 1627 |
| Meta-validação / pesos | 1972 | 1654 |

O agrupamento original sem rótulos combina mascaramento de identificadores e
cosseno TF-IDF exaustivo de caracteres 3–5 a .93. Bytes da divisão e procedência
dos grupos coincidem com o manifesto EP1 e a auditoria histórica aprovada.
IDs, hashes, rótulos e cobertura de grupos/papéis foram conferidos com a
planilha. Nenhum grupo ou duplicata normalizada cruza os três papéis.
Reutilizar o agrupamento histórico não estabelece separação semântica.

SHA-256 do SVM original:
`9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12`.
Combina TF-IDF de palavras (1,2), min_df=2, max_df=.98, max_features=60000 e
char_wb (3,5), min_df=3, max_df=1, max_features=50000. Ambos usam TF sublinear
não binário, IDF suavizado, norma L2 e float32. O pré-processamento decodifica
HTML, reduz espaços e converte para minúsculas sem remover acentos.
LinearSVC usa C=.05, hinge quadrática, penalidade L2, dual=auto, sem pesos de
classe, semente 42, tol=1e-4 e max_iter=10000. A seleção histórica usou
acurácia; a nova usa macro-F1. Calibração sigmoide externa fornece probabilidades
SVM; a calibração congelada do NorBERTo não muda.

<a id="reproduced-baseline-on-the-same-1972-development-rows"></a>

## Referência reproduzida nas mesmas 1972 linhas de desenvolvimento

As 3999 margens salvas foram reproduzidas do artefato original com
**diferença absoluta máxima 0.0**. Matriz de confusão e macro-F1 da referência
na meta-validação foram reproduzidos exatamente.

| Configuração | Acurácia | Macro-F1 |
|---|---:|---:|
| SVM nativo original | 0.465010141988 | 0.462600189230 |
| SVM original calibrado por sigmoide | 0.465517241379 | 0.465796980867 |
| A: SVM original 75% / NorBERTo 25% | 0.475152129817 | 0.475338716012 |

Essas 1972 observações já selecionaram os pesos originais; são diagnósticos de
desenvolvimento, não estimativas independentes de teste. Nenhuma planilha de
Teste1 ou Teste2 foi aberta na preparação. O macro-F1 histórico de Teste1
0.49584421663197825 permanece referência para comparação exata posterior,
não resultado recém-medido.

<a id="prepared-search-and-pending-work"></a>

## Busca preparada e trabalho pendente

O [guia de execução](../guides/svm-final-optimization.md) documenta fase A:
256 candidatos predefinidos, três folds agrupados, 768 ajustes. Quatro famílias
(word, char, char_wb, word+char_wb) recebem oito desenhos fixos cada, quatro
valores logarítmicos C e dois pesos de classe. Vetorizadores são ajustados
dentro do fold de treino. A fase B refina três representações, com até 61
candidatos e cinco folds. A regra macro-F1 de um erro-padrão controla a seleção,
mas não elimina o viés da busca adaptativa.

**Nenhuma busca foi executada.** Ajuste final otimizado, calibração aninhada
sigmoide/isotônica, seleção aninhada de pesos, congelamento A/B/C, análise
por classe/complementaridade, avaliação pareada retrospectiva de Teste1 e
recomendação continuam pendentes. O protocolo prospectivo completo está em
`results/svm-final-20261106/plan.json`, com critérios conservadores de adoção
e proibição de reajustar após Teste1. Preserve o ensemble atual até haver
evidência favorável à mudança.

<a id="verification-and-local-evidence"></a>

## Verificação e evidências locais

Onze testes sintéticos passaram, cobrindo isolamento de grupos, rejeição de
IDs/duplicatas, vocabulário só do fold, compatibilidade do solucionador,
orçamento, regra de um erro-padrão, retomada com checksums e execução/retomada
paralela. O teste paralelo usou só dados sintéticos. Sintaxe shell, verificações
básicas de importação/CLI e espaços Git passaram. Nenhum módulo histórico,
modelo, configuração ou resultado mudou; nenhum padrão foi promovido.

Saídas locais: `results/svm-final-20261106/{plan,audit,baseline}.json`,
`training-membership.jsonl`, `phase-a-candidates.json` e `report.md`.
O relatório explicita as seções pendentes. Não houve chamadas GPT ou ajuste/
inferência NorBERTo/Qwen/LR. A preparação gerou somente margens da referência
SVM; escores NorBERTo foram reutilizados.
