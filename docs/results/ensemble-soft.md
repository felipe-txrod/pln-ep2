# Weighted soft voting — avaliação final nas 900 linhas

Implementação e avaliação concluídas em 05/11/2026. O ensemble foi congelado
no commit `81e1cc5`, antes da leitura do gabarito nesta rodada. Nenhum dos
modelos-base foi retreinado e nenhum parâmetro foi alterado depois da avaliação.
O [guia](../guides/ensemble-soft.md) descreve a calibração, a grade e os comandos.
O [agregado auditado](data/ensemble-soft-test1.json) guarda todas as métricas.

## Partição e seleção

Os classificadores existentes foram ajustados nas 16.093 linhas de
desenvolvimento originais. A LR usada é `models/rl-development/`; a LR histórica
de corpus completo foi excluída porque já viu o antigo holdout. SVM e NorBERTo
preservam os artefatos do EP1.

Das 3.999 linhas do holdout, **2.027 calibraram** as probabilidades e **1.972
selecionaram os pesos**, sem linhas ou componentes de similaridade compartilhados.
O SVM não possui `predict_proba()` e recebeu calibração sigmoid/Platt. LR e
NorBERTo receberam temperaturas escalares escolhidas apenas por checagem
agrupada dentro da parte de calibração. As classes estão alinhadas como
`c1`, `c234`, `c5`.

Uma grade fixa de 16 misturas foi comparada por F1-macro na parte dos pesos.
Resultado congelado: **SVM 75%, LR 0%, NorBERTo 25%**. O zero é permitido no
simplex: a LR foi avaliada, mas não contribui para essa solução de fronteira.
F1-macro na seleção: 47,53%; não é estimativa independente de teste.

## Resultados finais

Todas as 900 linhas e posições originais foram mantidas. O gabarito foi aberto
somente depois de persistir as previsões nativas, calibradas e do ensemble.

| Classificador | Acertos/900 | Acurácia | F1-macro |
|---|---:|---:|---:|
| SVM original | 445 | 49,44% | 49,06% |
| LR de desenvolvimento | 447 | 49,67% | 49,32% |
| NorBERTo original | 432 | 48,00% | 47,86% |
| **Weighted soft voting** | **447** | **49,67%** | **49,58%** |

O ensemble empata com a LR em acurácia, com F1-macro 0,26 ponto percentual
maior. Sobre o NorBERTo, são 15 acertos e 1,67 ponto percentual de acurácia
a mais; sobre SVM, dois acertos. Isso não demonstra superioridade estatística.

### Matrizes de confusão

Linhas = classe verdadeira; colunas = prevista. Ordem: `c1`, `c234`, `c5`.

```text
SVM original          LR de desenvolvimento
[[142,  82,  76],     [[135,  89,  76],
 [ 82, 118, 100],      [ 73, 125, 102],
 [ 54,  61, 185]]      [ 49,  64, 187]]

NorBERTo original     Weighted soft voting
[[118, 113,  69],     [[139,  93,  68],
 [ 68, 155,  77],      [ 74, 135,  91],
 [ 54,  87, 159]]      [ 50,  77, 173]]
```

### Efeito da calibração nos modelos individuais

| Modelo após calibração | Acurácia | F1-macro |
|---|---:|---:|
| SVM sigmoid | 48,89% (440/900) | 48,77% |
| LR com temperatura | 49,67% (447/900) | 49,32% |
| NorBERTo com temperatura | 48,00% (432/900) | 47,86% |

A calibração do SVM muda o argmax e não necessariamente melhora acurácia;
neste teste, perdeu cinco acertos em relação ao SVM nativo. As temperaturas
preservaram os rótulos de LR/NorBERTo. O ensemble usa as distribuições calibradas,
não mistura margens SVM com probabilidades dos outros classificadores.

Matriz do SVM calibrado (as demais são iguais às nativas):

```text
[[142,  93,  65],
 [ 83, 126,  91],
 [ 55,  73, 172]]
```

Na comparação pareada, o ensemble acertou 42 casos que SVM errou e perdeu
40; contra LR, ganhou 36 e perdeu 36; contra NorBERTo, ganhou 100 e perdeu 85.
McNemar exato bilateral descritivo: p=0,9122, p=1,0000 e p=0,3033,
respectivamente. São análises exploratórias; repetições limitam independência,
e a entrega oficial de teste2 ainda não foi gerada por este método.

## Vazamento evitado e limitações herdadas

Nenhum rótulo do professor nas 900 linhas foi usado para calibrar, escolher
temperatura, selecionar pesos ou ajustar modelos. A separação de calibração e
pesos é por grupos, não apenas por linhas. O artefato foi versionado antes da
avaliação e conservou o hash
`f39a2200ecbaf124b95c78e249087f2f12302aa4f46768db756d828863301bec`.

Isso **não** torna as bases fornecidas completamente disjuntas em texto. A
[auditoria já existente](dataset-overlap.md) identifica 105 textos normalizados
do teste1 presentes no corpus original. O detalhamento desta execução é:

| Referência | Linhas do teste1 com texto idêntico | Com texto normalizado igual |
|---|---:|---:|
| Desenvolvimento original | 74 | 85 |
| Parte de calibração | 9 | 11 |
| Parte dos pesos | 5 | 9 |

Essas repetições são herdadas dos arquivos oficiais. Reutilizar modelos já
treinados impede torná-los retrospectivamente livres dos textos vistos em
desenvolvimento. Textos iguais podem ter rótulos distintos, e não há identificador
do pedido/contexto original para provar identidade de observação. Mantiveram-se
os conjuntos fornecidos, sem filtrar as 900 linhas ou refazer parâmetros após
ver o resultado. Portanto, o protocolo impede uso direto dos rótulos finais
no ajuste, mas não permite alegar generalização em dados inteiramente inéditos.
O histórico do projeto também já consultou métricas de holdout e teste1;
isso é declarado, não tratado como teste historicamente cego.

## Verificação e artefatos

A suíte terminou com **159 testes aprovados e 8 testes opcionais ignorados**.
Os testes específicos cobrem classes reordenadas, escores inválidos, separação
por grupo, ajuste proibido do estimador congelado, objetivo F1-macro, grade
limitada, leitura tardia do gabarito e detecção de arquivos adulterados.

Uma segunda conferência recalculou acurácia/F1 diretamente das matrizes,
reconstruiu as 900 misturas a partir das probabilidades salvas e comparou todos
os IDs/hashes. Os três artefatos-base e o modelo congelado permaneceram intactos.
O ajuste usou o runtime lexical já instalado no EP1; a predição/avaliação também
foi executada no ambiente do EP2 após instalar joblib 1.5.2, exigido pelos modelos.
A base pública NorBERTo foi copiada do cache local e verificada para satisfazer
um teste de integração; nenhum treinamento, download do modelo ou chamada GPT
foi realizado. Pandas 2.3.3 foi instalado apenas para um teste preexistente que
importa o leitor do EP1; o novo pipeline usa o leitor XLSX já disponível no EP2.

- Modelo distribuído: `models/ensemble-soft/`.
- Ajuste e escores nativos: `results/ensemble-soft-v2-20261105/`.
- Previsões finais: `results/ensemble-soft-test1-20261105/`.
- Métricas e auditoria local: `results/ensemble-soft-evaluation-20261105/`.
- Agregado versionado: `docs/results/data/ensemble-soft-test1.json`.

Os presets existentes de maioria não foram alterados. Para novos dados, aplique
o modelo congelado com `predict-soft`; não retune a partir desses resultados.
