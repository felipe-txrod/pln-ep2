<a id="norberto-svm-and-rl-majority-comparison"></a>

# Comparação de maioria NorBERTo, SVM e RL

[Índice dos resultados](README.md) · [Agregado verificado](data/ensemble-majority-test1.json)

A melhor maioria de três membros nas 900 linhas rotuladas foi **NorBERTo + SVM
+ RL, com RL nos empates triplos**: 443 acertos, acurácia 49.22% e macro-F1
48.86%. RL isolada acertou 447 (49.67%); o melhor ensemble efetivo perdeu quatro
acertos, ou 0.44 pontos percentuais, frente ao melhor modelo individual.

| Configuração | Acertos / 900 | Acurácia | Macro-F1 | Linhas de desempate |
|---|---:|---:|---:|---:|
| RL isolada | 447 | 49.67% | 49.32% | 0 |
| SVM isolado | 445 | 49.44% | 49.06% | 0 |
| Os três, desempate RL | 443 | 49.22% | 48.86% | 5 |
| Os três, desempate NorBERTo | 442 | 49.11% | 48.75% | 5 |
| Os três, desempate SVM | 440 | 48.89% | 48.52% | 5 |
| NorBERTo original do EP1 isolado | 432 | 48.00% | 47.86% | 0 |

<a id="voting-rule-and-candidate-coverage"></a>

## Regra de votação e cobertura dos candidatos

Cada membro contribui um rótulo de classe. A maioria estrita vence; caso
contrário, o árbitro local fornece sua previsão salva. Não há limiares de
confiança, média de probabilidades, pesos, calibração, nova inferência,
retreinamento ou pedidos remotos.

A comparação cobriu 15 configurações: três modelos isolados, três pares com
cada um dos três árbitros e o trio com cada árbitro. Produzem seis sequências
distintas. Um par que usa um votante para desempatar reproduz exatamente esse
votante. Os pares vencedores aparentes de 447 acertos duplicam RL; não a
melhoram. Um par com o terceiro modelo como árbitro reproduz a maioria do trio
com esse terceiro como desempate.

Os três concordaram em 579 linhas. Outras 316 tiveram maioria de 2 contra 1,
e cinco tiveram três rótulos distintos. A concordância entre pares foi:

| Par | Concordâncias / 900 | Fração |
|---|---:|---:|
| SVM e RL | 861 | 95.67% |
| NorBERTo e RL | 608 | 67.56% |
| NorBERTo e SVM | 584 | 64.89% |

A concordância elevada dos modelos lexicais limita a diversidade dos votos.
Com RL no desempate, o trio muda só cinco previsões dela; perde quatro acertos
e não ganha nenhum. Isso descreve o teste1 observado, sem garantir comportamento
em bases futuras.

<a id="artifacts-verification-and-scope"></a>

## Artefatos, verificação e escopo

Uma [auditoria posterior do conjunto reservado](ensemble-holdout.md) verificou
isolamento do treino e avaliou os mesmos artefatos na partição congelada de
3,999 linhas.

Os modelos treinaram nas 16,093 linhas congeladas de desenvolvimento do EP1,
excluindo 3,999 reservadas. NorBERTo é o adaptador original inalterado; SVM usa
C=0.05 fixo, e RL, C=0.25. Manifestos e identidades das execuções foram
verificados antes da importação. Modelos históricos completos foram excluídos.

Todas as previsões/configurações foram persistidas antes de ler o gabarito.
Os 900 textos, IDs e posições coincidiram com
`data/test1-rotulado-para-teste-estatistico.xlsx`. Um leitor ZIP/XML independente
verificou as 13,500 previsões, roteamento de maioria, empates, matrizes,
acurácia, macro-F1 e hashes dos modelos/execuções/arquivos. A comparação em CPU
levou 1.27 segundos.

A evidência bruta está em `results/ensemble-majority-comparison-20261105/`,
com configurações, previsões e métricas por candidato; `summary.json` guarda
classificação e procedência. `verify.py` e `independent-verification.json`
registram a verificação independente.

Esta é uma **comparação exploratória em rótulos de teste1 já consultados**.
Ordenar configurações não fornece avaliação final independente nem estabelece
superioridade estatística. Na comparação, o padrão de entrega era Qwen +
NorBERTo. Depois o usuário excluiu Qwen; o padrão atual é NorBERTo + SVM + RL
com arbitragem RL. O protocolo oficial teste2 antes de teste1 não muda.
Nenhuma previsão ou entrega de teste2 foi gerada.

<a id="reproduction"></a>

## Reprodução

Com execuções salvas dos membros e dependências do projeto disponíveis:

```bash
PYTHONPATH=src python3 -m ensemble_classifier compare-majority \
  --config configs/ensemble-majority-test1.json \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/ensemble-majority-comparison-new
```

Use novo diretório. A configuração separada de produção,
`configs/ensemble-majority.json`, seleciona os mesmos modelos, maioria estrita
e desempate RL para teste2 quando as previsões dos membros existirem.
