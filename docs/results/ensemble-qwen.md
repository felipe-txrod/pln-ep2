<a id="qwen-integration-into-soft-voting-and-stacking"></a>

# Integração do Qwen à votação suave e ao stacking

[Índice dos resultados](README.md) · [Comandos](../guides/ensemble-qwen.md)
· [Agregado verificado](data/ensemble-qwen.json)

O Qwen selecionado com a divisão do EP1 foi adicionado às duas famílias de
ensemble, sem alterar seus pesos-base. A integração preserva os artefatos
originais de três modelos e usa novos pacotes em `models/ensemble-soft-qwen/`
e `models/ensemble-stacking-qwen/`.

<a id="test1-comparison"></a>

## Comparação em Teste1

Os 900 textos, IDs e posições originais foram mantidos. As previsões foram
salvas antes da leitura do gabarito, após congelar calibração, pesos e regularização.

| Método | Acertos / 900 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Votação suave, três membros | 447 | 49.67% | 49.58% |
| Votação suave, quatro candidatos incluindo Qwen | 447 | 49.67% | 49.58% |
| Stacking, três membros | 442 | 49.11% | 49.03% |
| Stacking, quatro membros incluindo Qwen | 440 | 48.89% | 48.80% |
| Qwen selecionado isolado | 414 | 46.00% | 45.53% |

A busca suave selecionou **SVM 75%, RL 0%, NorBERTo 25%, Qwen 0%**. Qwen foi
avaliado, calibrado e incluído em todos os candidatos pertinentes, mas a soma
escolhida não lhe atribui contribuição. Os 900 rótulos finais coincidem com a
votação suave de três membros. A inclusão não melhorou a qualidade medida.

Stacking usa as 12 probabilidades calibradas e selecionou C=0.1. Os coeficientes
aprendidos do Qwen são não nulos. Frente ao stacking original, corrige 25 erros
e prejudica 27 acertos, perdendo dois líquidos, ou 0.22 pontos percentuais.
Frente à votação suave, corrige 47 erros e prejudica 54 acertos; outras 30
mudanças continuam erradas.

As matrizes usam classes verdadeiras nas linhas e preditas nas colunas,
em ordem `c1, c234, c5`:

```text
Soft with four candidates     Stacking with four members
[[139, 93, 68],              [[116, 125, 59],
 [ 74,135, 91],               [ 55, 158, 87],
 [ 50, 77,173]]               [ 36, 98,166]]
```

Frente à entrega real do NorBERTo do EP1, votação suave ganha 15 acertos com
McNemar exato p=0.303327; stacking ganha oito, p=0.615381. Nenhum estabelece
superioridade significativa. As comparações continuam exploratórias, pois
teste1 já foi consultado em outras etapas. Nenhum parâmetro mudou em resposta
a esses resultados.

<a id="frozen-fitting-protocol"></a>

## Protocolo congelado de ajuste

Os quatro modelos-base foram ajustados em 16,093 linhas, excluindo 3,999
reservadas das atualizações de gradiente ou ajuste dos classificadores.
Qwen é o adaptador selecionado de identidade
`a045c5726a85c466914c55f9cee9fb0d412af4d66387e14b590e53b4782beed5`.
O adaptador antigo com outra divisão foi excluído.

A divisão reservada permanece inalterada:

| Papel | Linhas | Grupos de similaridade |
|---|---:|---:|
| Calibração de probabilidades | 2,027 | 1,627 |
| Seleção de pesos suaves / meta-treinamento do stacking | 1,972 | 1,654 |

Os dois arquivos de partição são idênticos em bytes aos originais de três
membros, incluindo a atribuição dos folds do stacking. Nenhuma linha de
calibração entra no meta-modelo, e grupos não cruzam papéis.

A sigmoide do SVM e as receitas existentes de temperatura foram mantidas.
A temperatura Qwen foi escolhida somente na calibração, em três folds agrupados:
perda logarítmica nativa 1.05293 contra OOF com temperatura 1.04078; temperatura
final 1.515480709210816. O argmax de classe permanece inalterado.

A votação suave comparou os 35 candidatos fixos em incrementos de um quarto
no simplex por macro-F1 de seleção. Stacking usou cinco folds agrupados,
candidatos fixos de C `[0.01, 0.1, 1, 10]` e a regra existente de um erro-padrão.
O macro-F1 OOF selecionado é 47.30%; participou da seleção, não é estimativa
independente de desempenho.

O checkpoint Qwen foi escolhido no conjunto reservado original completo.
Essas linhas ficaram fora da atualização dos pesos, mas influenciaram a
seleção do checkpoint. Com a pesquisa histórica, essa exposição herdada deve
ser declarada ao interpretar calibração e validação do meta-modelo.

<a id="inference-auditing-and-preserved-artifacts"></a>

## Inferência, auditoria e artefatos preservados

Novas probabilidades Qwen foram geradas em lotes sequenciais de 16 nas 3,999
linhas reservadas e 900 textos de teste1, sem passar gabaritos ao modelo.
Isso levou 427.40 e 98.69 segundos, respectivamente. Cada saída mantém ID e
checksum do texto. Nenhuma exportação de classes virou probabilidade artificial,
e a exportação de validação recuperada anteriormente não forneceu probabilidades.

A nova avaliação reservada novamente acerta 1,791/3,999 (44.79%). Macro-F1 é
44.4371%; poucas decisões diferem da avaliação original agrupada do treino,
cujo macro-F1 era 44.4429%. O ajuste de quatro membros usa consistentemente
as novas distribuições inferidas.

Um leitor ZIP/XML independente verificou IDs, textos e rótulos. Cálculos
independentes reconstruíram todas as probabilidades calibradas de teste,
os escores dos 35 candidatos, as somas suaves e saídas do meta-modelo de 12
atributos. A diferença máxima de probabilidade foi 1.11e-16. Matrizes, acurácia
e macro-F1 reproduzem os relatórios; os artefatos originais de três membros
continuam carregáveis e reproduzem controles 447/900 e 442/900.

A evidência bruta está em `results/ensemble-qwen-verification-20261105/`.
Escores Qwen estão em `results/qwen-ensemble-scores-20261105/`, com a execução
selecionada de teste1 também em `results/qwen-ep1-test1/`. Ajustes e previsões
usam diretórios separados `ensemble-soft-qwen-*` e `ensemble-stacking-qwen-*`.
Todos os pesos, modelos e resultados históricos originais permanecem
inalterados. Nenhum modelo-base foi retreinado nem entrega de teste2 gerada.

A suíte offline disponível passou em 171 testes em 30.15 segundos; dois módulos
opcionais não relacionados do SDK OpenAI foram excluídos neste contêiner.
Ambos os comandos padrão documentados também executaram em CPU com escores
salvos, reproduzindo as previsões auditadas byte a byte, sem ler o gabarito.
