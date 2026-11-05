# ACH2118 Introdução ao Processamento de Língua Natural

## Modelo para elaboração de relatórios – EP2 – com análise

O relatório é composto dos itens a seguir, e deve ser entregue em pasta grupoXX.ZIP pelo mesmo responsável pela entrega anterior, onde XX é o número do grupo, juntamente com os outros dois itens solicitados (pdf da apresentação e planilha xlxs com os rótulos de teste).

1. Nome e número USP de todos os integrantes efetivos do grupo – esta será a informação levada em conta para eventual aplicação do bônus por ausências/desistências.

2. Um parágrafo introdutório com uma breve descrição da estratégia principal escolhida para cada uma das tarefas de classificação, resumindo qual o tipo de representação textual utilizada (palavras, caracteres, embeddings etc.) e o tipo de classificador utilizado. É permitido descrever mais de um modelo para cada tarefa desde que o modelo final escolhido para cada tarefa seja claramente indicado.

3. Pré-processamento: como os textos foram tratados antes de treinar o classificador.

4. Parâmetros do classificador que foram avaliados na busca em grade: uma tabela com a lista dos parâmetros e a faixa de valores considerada para cada um. Exemplo:

   | Parâmetro | Valores |
   |---|---|
   | Tol | 1e-5..-3 |
   | Neurônios | {100,150,200,500} |

5. Valores ótimos que foram obtidos para cada parâmetro e para cada tarefa de classificação (se houver mais de uma). Exemplo:

   | Tarefa | Parâmetro | Valor |
   |---|---|---|
   | Tarefa1 | Tol | 1e-5 |
   | | Neurônios | 150 |
   | Tarefa2 | Tol | 1e-3 |
   | | Neurônios | 300 |

6. Procedimento: se houve separação entre treinamento e teste (e qual a separação usada), e quaisquer outros passos necessários para a reprodução do experimento.

7. Resultado. A acurácia da tarefa de classificação dos modelos desenvolvidos.

8. Evolução em relação à versão anterior (do ep1): testar o modelo no conjunto de teste do EP1, e mostrar de forma conjunta os resultados do modelo anterior e atual com teste de significância estatística demonstrando a superioridade efetiva do modelo atual. Exemplo:

   | Estratégia | Teste | Acurácia |
   |---|---|---|
   | Modelo ep1 | ... | ... |
   | Modelo ep2 | ... | ... |

   A vantagem do modelo2 é estatisticamente significativa de acordo com um teste de McNemar para p= tal... [reporte a estatística McNemar e o p-valor)

9. Análise. Discutir (i) resultados de interpretação global do modelo (i.e., as características mais importantes de cada classe), (ii) exemplos de interpretação local, e (iii) uma análise de erros com considerações sobre o resultado. Para os itens (i) e (ii), use a biblioteca eli5 ou outra se preferir, com um classificador simples (e.g. reglog baseado em palavras). Para o item (iii), selecione manualmente alguns casos incorretamente classificados (tantas quanto achar necessário) e faça considerações sobre o porquê do erro.

10. Link para o repositório de código (github, colab ou similar) e passo-a-passo do que é preciso fazer para reproduzir o resultado a partir do código apresentado: bibliotecas, dependências etc.

---

1 de 1
