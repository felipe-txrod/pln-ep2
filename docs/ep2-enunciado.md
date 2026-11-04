# Exercício prático 2

**ACH2118 Introdução ao Processamento de Língua Natural**

**Ivandré Paraboni**  
**EACH - USP**

---

## Objetivos

- Evoluir a solução do EP1 de modo a obter resultados superiores aos anteriores.
- Treino: o mesmo do EP1
- Teste 2: um novo conjunto de 897 instancias será divulgado, e cuja resposta deve ser entregue.
- Teste 1 (anterior): será fornecido agora com seus 900 rótulos, apenas para condução do teste estatístico comparando EP1 x EP2.

---

## O que fazer

- Evolua seu método e gere as predições do teste2 (897 rótulos). Isso é a entrega principal da tarefa.
- Feito isso, aplique seu método ao conjunto teste1 anterior (900 instâncias).
- A comparação entre estes 900 rótulos e os 900 rótulos entregues no EP1 é feita com o teste de McNemar, e reportada no relatório.

---

## O que entregar

Entrega de 3 itens em arquivo grupoXX.zip, feita pelo mesmo responsável pela entrega 1:

1. Relatório do projeto (novo modelo - com análise) disponibilizado
2. Apresentação em formato pdf (não ppt/pptx) com duração de até 10min.
3. Planilha do conjunto de teste2 rotulado (897 rótulos).

Grupos: https://docs.google.com/spreadsheets/d/146ssklR6dNVHaOkIM4VM1XCbTdND7IS5LlX-x5X_LMw/edit?usp=sharing

---

## O que será avaliado

A acurácia de teste determina a ordem de apresentação em aula.

Nota final: acurácia de teste (70%), método inovador / bem elaborado (10%), análise de características (5%), resultado significativamente superior ao EP1 a p<0,05 (5%), relatório (5%), apresentação em aula (5%).

Modelos de Tf-Idf/BoW com regressão logística, SVM, Naive Bayes e afins, ou a repetição do método do EP1, não pontuam no critério “inovação”.

(-) as penalidades previstas  
(+) normalização pela maior nota final.

---

## O que será penalizado

- Baixo desempenho: EPs cuja acurácia média seja inferior à média do baseline de classe majoritária recebem nota 3,0 e não são avaliados, e também não qualificam para o critério ‘inovação’.
- Erro de entrega: entrega de rótulos do teste1, ou comparação entre modelos usando o teste2, também resulta em nota 3,0 sem avaliação.
- Descontos de 20% sobre a nota-base, aplicado a cada um dos casos a seguir de forma cumulativa:
  - Resultado inferior ao baseline oficial (regressão logística + TF-IDF);
  - Não entregar de relatório ou código no prazo;
  - Não entregar do PDF da apresentação;
  - Formato inválido do arquivo de teste (e.g., mais/menos linhas ou colunas), ou rótulos diferentes/ fora da posição, tipo de arquivo não xlsx etc.

---

## O que será valorizado

- Desistências e desaparecimentos podem, opcionalmente, ser indicados no relatório.
- Bônus de 5% na nota final para cada integrante “perdido”.
- Objetivo é valorizar grupos que tiveram que trabalhar mais.

---

## Cronograma

- Entrega (resultado + apresentação + relatório com código) até domingo 08/11.
- Apresentação dos trabalhos em aula em 10/11 e 17/11 em ordem de resultado de acurácia do teste – começando pelo último colocado.
- Ausência na hora da apresentação anula anota da apresentação (mas não do trabalho como um todo, desde que atenda aos demais critérios)

---

## E não esqueça

- Conjunto teste2 = avaliação oficial do EP2
- Conjunto teste1 = apenas para comparar com o EP1

---

## Bom trabalho!
