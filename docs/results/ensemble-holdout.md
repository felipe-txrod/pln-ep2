<a id="local-ensemble-frozen-holdout-isolation-and-accuracy"></a>

# Ensemble local: isolamento e acurácia no conjunto reservado congelado

[Índice dos resultados](README.md) · [Agregado da auditoria](data/ensemble-holdout-audit.json)

NorBERTo, SVM e RL selecionados usam as mesmas 16,093 linhas de desenvolvimento.
As 3,999 reservadas foram excluídas do ajuste. A revisão da divisão encontrou
**zero IDs compartilhados, zero grupos de similaridade compartilhados e zero
textos reservados normalizados presentes no desenvolvimento**. Qwen está
excluído desta avaliação e do ensemble padrão ativo.

<a id="recomputed-holdout-results"></a>

## Resultados recalculados do conjunto reservado

Todas as previsões cobrem os mesmos 3,999 IDs originais, sem filtrar ou reordenar.
Os resultados foram recalculados das previsões salvas; nenhum modelo foi
treinado ou executado novamente.

| Configuração | Acertos / 3,999 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| NorBERTo original do EP1 | 1,812 | 45.31% | 45.31% |
| NorBERTo + SVM + RL, desempate RL | 1,803 | 45.09% | 44.78% |
| RL somente de desenvolvimento | 1,803 | 45.09% | 44.78% |
| NorBERTo + SVM + RL, desempate SVM | 1,802 | 45.06% | 44.76% |
| NorBERTo + SVM + RL, desempate NorBERTo | 1,799 | 44.99% | 44.68% |
| SVM somente de desenvolvimento | 1,792 | 44.81% | 44.47% |

A melhor maioria de três membros novamente usa RL nos 28 empates triplos.
Empata em acurácia com RL, mas tem previsões e macro-F1 ligeiramente distintos.
NorBERTo isolado lidera este conjunto por nove acertos. A
[comparação em teste1](ensemble-majority.md) tem outro líder individual:
RL acerta 447/900, e a mesma maioria, 443/900.

<a id="evidence-that-the-holdout-was-excluded"></a>

## Evidências da exclusão do conjunto reservado

O corpus inalterado contém 20,092 linhas, SHA-256
`0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
A divisão congelada comum de
`EP1/results/research/dataset-hypotheses-20260923/splits.csv` tem SHA-256
`0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.
Desenvolvimento contém 12,956 grupos; conjunto reservado, 3,281.

| Modelo | Evidência de treino conferida | Linhas reservadas no ajuste |
|---|---|---:|
| NorBERTo | Todos os IDs de treino/grupos salvos, checksum do manifesto de treino, checksum do adaptador original e identidade da avaliação reservada original | 0 |
| SVM | Checksum original do modelo, contrato de metadados exportado inalterado, divisão fixada e caminho de treino somente no desenvolvimento | 0 |
| RL | Todos os IDs de treino, hashes de texto e rótulos, checksum da identidade ordenada, checksum do modelo e identidade do script de treino | 0 |

O `training-rows.csv` salvo do NorBERTo coincide exatamente com todas as linhas
e grupos de desenvolvimento na ordem. O SHA-256 do adaptador original é
`d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`.
O relatório original reservado identifica o mesmo adaptador e manifesto de treino.

`models/rl-development/training-rows.jsonl` coincide exatamente com os 16,093
IDs, rótulos e hashes originais. A identidade ordenada coincide com seu checksum.
O SHA-256 selecionado dos pesos é
`05d69e147a10b7c49ebe3893f247e0f23e8d61005aaf4fd7b7eb45845d49a20f`.
A execução avaliada identifica esses mesmos pesos.

O SHA-256 selecionado do SVM é
`9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12`.
Coincide com o artefato preservado do EP1. Os metadados registram a mesma divisão
e 16,093 linhas. O pipeline passa apenas `x[development]`, `y[development]` e
`groups[development]` à busca/ajuste; o conjunto reservado só entra em predição
e avaliação. O SVM não salvou lista própria de treino; a evidência vem da divisão
fixada, procedência do artefato e protocolo registrado. `dataset.rows=20092`
descreve a planilha inteira; `fixed_validation.development_rows=16093`, a
partição de ajuste.

<a id="verification-and-limitations"></a>

## Verificação e limitações

A auditoria leu a planilha independentemente por ZIP/XML da biblioteca padrão,
incluindo a conversão existente de texto numérico na linha Excel 5028.
Conferiu listas de treino, fontes, rótulos, grupos, artefatos e procedência das
previsões. A função de maioria de produção recebeu só textos e votos salvos.
Cada voto foi reconferido por contagem independente de maioria estrita, e
métricas foram calculadas diretamente das matrizes. Quinze configurações
produziram 59,985 linhas verificadas. As métricas históricas individuais
foram reproduzidas.

A evidência bruta está em `results/ensemble-holdout-audit-20261105/`, com
`audit.py`, `workbook_reader.py`, `isolation.json`, `summary.json` e previsões/
métricas de cada candidato. O agregado público preserva hashes dos arquivos
originais de previsões e da evidência de treino.

Excluir do ajuste supervisionado não torna este um teste final inédito: o
conjunto reservado já foi inspecionado na pesquisa anterior. São resultados
de validação fixa. Modelos, adaptadores, caches e resultados históricos não
mudaram. Não houve nova inferência teste1/teste2 nem entrega.

O usuário excluiu Qwen explicitamente. As duas configurações regulares agora
selecionam `norberto svm rl`, estratégia `majority` e desempate `rl`.
Presets separados mantêm os mesmos parâmetros. Qwen só fica disponível por
substituição explícita futura dos membros; seus artefatos salvos são preservados.
