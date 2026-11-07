<a id="historical-training-scale-audit-recipe-3"></a>

# Auditoria histórica da escala de treinamento: receita 3

Desativado em 2026-11-05. O modelo selecionado é o adaptador original da entrega
do EP1, sem alterações. As medições abaixo são históricas; as implementações
de treino e diagnóstico foram removidas das fontes ativas e preservadas em
`results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

Uma nova auditoria restrita ao treino comparou os dois adaptadores salvos da
receita 3 na mesma amostra estratificada de 300 linhas de desenvolvimento
(100 por classe), semente 42. Ambos treinaram nessas linhas. Nenhum otimizador
foi construído, e teste1/teste2 não foram acessados nesta auditoria.

| Medida | Adaptador de desenvolvimento | Adaptador do corpus completo |
|---|---:|---:|
| Acertos na amostra compartilhada de treino | 156/300 (52.00%) | 113/300 (37.67%) |
| Entropia cruzada na amostra compartilhada | 0.98624 | 1.08960 |
| Previsões na amostra compartilhada `c1 / c234 / c5` | 96 / 111 / 93 | 62 / 15 / 223 |
| Pico registrado da norma do gradiente do otimizador antes da limitação | 22.58 | 1,196.86 |
| Norma do gradiente em CPU do adaptador final em 12 respostas curtas de treino | 1.60418 | 1.63878 |

O adaptador completo ajusta pior essa amostra compartilhada e já exibe nela o
viés para `c5`. Isso sustenta uma falha de otimização/ajuste. Os resultados na
amostra de treino não estimam generalização em dados reservados.

Os 16,093 conjuntos compartilhados de atributos em cache coincidem exatamente,
e nova tokenização coincide nas 20,092 linhas de treino completo. Manifestos,
hashes das fontes, cobertura da partição, tensores exportados e hashes
inalterados dos pesos passaram na verificação. A proporção de `c5` no treino
caiu de 34.59% para 34.30%; aumento da frequência prévia da classe não explica
o viés das previsões.

A primeira norma registrada da execução completa acima de 25, em unidades por
amostra, ocorre na atualização 1,100, antes da janela parcial de acumulação da
primeira época em 1,256. Logo, essa cauda não pode ser o gatilho inicial da
escalada observada. As derivadas finais são normais no lote curto verificado:
CPU SDPA/eager e GPU SDPA concordam, com diferenças relativas de gradiente em
GPU inferiores a 0.000011 nos dois adaptadores. Só 5.29% da energia do gradiente
está na primeira atenção do adaptador completo (12.74% no desenvolvimento).
A concentração na primeira atenção do checkpoint anterior da receita 2 não
se reproduz aqui.

A receita 3 preserva as unidades pretendidas da perda e do otimizador, mas não
garante treino estável com mais dados. Duas épocas mudaram as atualizações de
2,012 para 2,512, o aquecimento de 202 para 252 e a ordem de amostragem. A causa
específica entre inclusão de linhas, ordem, agendamento e suas interações segue
sem isolamento. Registros a cada 20 atualizações não verificam a limitação
real nem localizam exatamente a primeira instabilidade; derivadas finais
normais não certificam a trajetória. O acompanhamento controlado exige novo
treino com validação interna agrupada, alterando um fator por vez e rastreando
cada atualização próxima ao pico. O gabarito de teste1/teste2 não pode selecionar
a correção.

O [agregado da auditoria](../../results/data/norberto-training-scale-audit.json)
registra todas as medidas, hashes e limitações. A auditoria levou 74.40 segundos
no ambiente ROCm offline registrado, e 34 testes sintéticos específicos em CPU
passaram. Receitas de produção, adaptadores e seleção do ensemble foram
preservados. Na imagem validada com as montagens padrão de EP1/cache, reproduza com:

```bash
PYTHONPATH=src HSA_OVERRIDE_GFX_VERSION=10.3.0 \
python3 scripts/diagnostics/norberto/audit-training-scale.py \
  --ep1-root /ep1 --device gpu \
  --output-dir results/norberto-scale-audit-another-new-run
```

A comparação forense posterior identifica por que a correção anterior não
resolveu o treino completo: o commit `f947d88` substituiu o Trainer normalizado
pelo compatível com o histórico nas duas partições. As **62/62 amostras de
gradientes e taxas da execução completa anteriores à atualização 1,256**
coincidem exatamente com a execução histórica instável. Os picos coincidentes
incluem 1,100 e 1,120; a primeira divergência registrada só ocorre em 1,260,
após a mudança de ponderação da cauda. Em janelas completas, multiplicar a
perda normalizada por quatro restaura a soma histórica das médias dos microlotes.
Isso confirma a recorrência da trajetória amostrada conhecida, não igualdade
bit a bit de todos os tensores ocultos.

O diagnóstico anterior de 600 atualizações com modelo real fixava a partição
de desenvolvimento no código. Sua concordância com o EP1 e as referências
intencionalmente escaladas dos testes pequenos validaram compatibilidade, sem
cobrir a regressão da execução completa. Agora aceita `--partition full`.
O invólucro de produção também exige a configuração congelada original,
incluindo taxa de aprendizado 0.001. Nem a normalização nem a mudança de
compatibilidade validaram nova configuração do otimizador para os dois tamanhos
de base.

A comparação restrita aos registros não carrega modelos nem executa otimização:

```bash
PYTHONPATH=src python3 scripts/diagnostics/norberto/audit-training-scale.py \
  --ep1-root /ep1 --history-only \
  --output-dir results/norberto-history-regression-another-new-run
```

O acompanhamento também realizou **treinamento real limitado**, separado da
auditoria inicial dos adaptadores salvos. Um rastreamento por atualização
atingiu 376 antes de ser interrompido para restringir o experimento. Um novo
prefixo de 320 atualizações reproduziu as identidades dos 320 primeiros lotes,
suas normas de gradiente e as 16 amostras históricas disponíveis. Salvou pesos,
AdamW, agendador e estado aleatório em 200. Retomar sem intervenção reproduziu
exatamente as identidades dos 120 lotes seguintes, as normas dos gradientes e
das atualizações; o binário final do adaptador foi idêntico ao prefixo contínuo.

O tratamento controlado retomou o mesmo checkpoint e multiplicou apenas a taxa
agendada do otimizador por 0.1 nas atualizações 201–320. Todas as identidades dos
lotes coincidiram, o primeiro gradiente foi idêntico e a primeira norma da
atualização dos parâmetros foi 0.10000013 vezes a referência. Normalização da
perda, escala de compatibilidade, estado aleatório do dropout, momentos do
otimizador, limitação e tempos do agendador foram mantidos na intervenção.

| Janela 201–320 | Taxas originais do otimizador | Taxas multiplicadas por 0.1 |
|---|---:|---:|
| Pico da norma do gradiente do otimizador antes da limitação | 576.49658 | 29.24076 |
| Atualizações com norma acima de 100 | 10 | 0 |
| Perda média por exemplo por atualização | 1.11307 | 1.08339 |
| Norma média da atualização dos parâmetros | 0.12863 | 0.01396 |

Todas as normas após limitação ficaram em até 1.000001. No pico original,
atualização 276, o LoRA da primeira atenção concentrou 99.76% da energia do
gradiente; em 257, 99.73%. Os registros antigos a cada 20 atualizações não
capturaram esses picos transitórios, distintos de derivadas finais normais.
Isso fornece evidência causal local de que atualizações menores evitam esses
picos no estado reproduzido. Não isola a escala dos gradientes nem estabelece
receita estável de ponta a ponta ou ganho de acurácia. Os novos prefixos são
diagnósticos, sem substituir produção nem acessar teste1/teste2. O
[agregado](../../results/data/norberto-training-scale-audit.json) registra hashes
do rastreamento/checkpoint, taxas reais, controle negativo, 936 atualizações
diagnósticas no total e limites da validação.
