<a id="norberto-optimizer-unit-compatibility-correction"></a>

# Correção de compatibilidade das unidades do otimizador do NorBERTo

Fluxo histórico desativado em 2026-11-05. O modelo selecionado é o adaptador
original da entrega do EP1, sem alterações. Os comandos abaixo descrevem o
experimento anterior; sua implementação ativa de treino/avaliação foi removida.
Cópias locais do código estão em `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Índice da documentação](../../README.md)

A correção inicial da normalização corrigiu a ponderação dos exemplos, mas
alterou as unidades dos gradientes consumidos pelo limiar herdado de limitação
e pelo otimizador AdamW. A receita versão 3 torna explícita essa conversão.
Isso resolve um defeito de compatibilidade da receita; a acurácia ainda exige
uma avaliação independente concluída no conjunto reservado.

<a id="confirmed-limitation-of-the-correction"></a>

## Limitação confirmada da correção

A auditoria posterior da regressão no corpus completo constatou que a receita
3 reproduz **as 62 amostras de normas dos gradientes e taxas de aprendizado
anteriores à cauda da execução histórica completa sabidamente instável**,
inclusive os picos nas atualizações 1,100 e 1,120. A primeira diferença
registrada ocorre em 1,260, após a cauda em 1,256. Isso decorre da conversão
intencional de compatibilidade: em janelas completas de 4 por 4,
`4 * sum(CE) / 16` é a soma original das médias dos microlotes. A conversão
foi implantada no commit `f947d88` para ambas as partições.

O diagnóstico anterior de prefixo com modelo real fixava `development` no
código e executava só 600 atualizações; a comparação bem-sucedida com o EP1
não testou a trajetória completa que falhava. Os testes com modelo pequeno
comparam intencionalmente com gradientes multiplicados por quatro. Verificam
a conversão, não a estabilidade no corpus completo. O diagnóstico agora aceita
`--partition full` e deriva o limite dessa partição. Um rastreador separado
registra limitação real, hashes dos lotes e atualizações dos parâmetros, e salva
o estado do otimizador/agendador/gerador aleatório para intervenções controladas.

A receita 3 implementa compatibilidade com o EP1; não é uma correção validada
para a instabilidade no corpus completo. O invólucro de produção também fixa
a taxa original de 0.001 e não oferece configuração separada de treino.
Remover apenas a conversão não é solução geral estabelecida: a receita 2
também falhou no desenvolvimento. Veja a [auditoria atual](../../results/README.md#training-scale-audit-recipe-3)
para as evidências e os limites da atribuição.

<a id="gradient-units"></a>

## Unidades dos gradientes

Para o microlote j com n_j exemplos rotulados e N exemplos na atualização
acumulada, a contribuição reportada é `sum(CE_j) / N`. A retropropagação usa
`K * sum(CE_j) / N`, sendo K a acumulação configurada (4). A perda retornada
pelo Trainer é dividida por K, e a avaliação não recebe escala. A taxa de
aprendizado, limiar de limitação, epsilon do Adam, decaimento de pesos, semente,
camadas-alvo, partição e orçamento de épocas originais são preservados.
O `grad_norm` registrado usa unidades do otimizador, como no EP1; divida-o por
K para comparar normas anteriores à limitação com a versão 2. A perda
registrada já é a média por exemplo e não deve ser dividida novamente.

Com quatro microlotes iguais, isso reproduz as unidades originais dos gradientes
do EP1. Em grupos desiguais ou incompletos, cada exemplo recebe o mesmo peso.
Não reproduz a ponderação excessiva original da cauda com um exemplo. A escala
de compatibilidade é deliberadamente distinta da média do objetivo; fica
registrada explicitamente, sem depender da heurística de kwargs do Transformers.

A implementação está isolada em ``training_recipe.py``. O auxiliar existente
de perda por amostra permanece inalterado para preservar manifestos e validação
históricos. O invólucro de produção seleciona a receita versão 3; o avaliador
do conjunto reservado aceita versões 2 e 3 e verifica o novo hash da receita e
o contrato de escala do otimizador ao avaliar a versão 3.

<a id="verification"></a>

## Verificação

Os 24 testes em CPU cobrem perda por amostra, partições, grupos completos e
parciais de acumulação, gradientes abaixo/acima do limiar de limitação, três
atualizações AdamW sequenciais e perda de avaliação sem escala. Todos passaram
em 8.01 segundos.

Uma execução completa de quatro atualizações do Trainer com ModernBERT/PEFT
pequeno em GPU foi comparada com acumulação manual ponderada/escalada de
gradientes, limitação e AdamW em CPU. O erro máximo dos parâmetros foi 1.49e-8;
o erro relativo máximo do gradiente, 2.68e-7. Parâmetros congelados ficaram
idênticos. A perda reportada coincidiu com a média por exemplo. Isso verifica
a integração, incluindo a cauda desigual 4+4+4+1.

O diagnóstico limitado com modelo real concluiu as 600 atualizações solicitadas
em 559.68 segundos. Usou as 16,093 linhas de desenvolvimento como base de treino.
As 30 normas e taxas de aprendizado registradas coincidiram exatamente com o
EP1 original. Na atualização 540, a norma foi 7.78463, contra 253.149 na versão
2 (ou 1.94616 contra 253.149 em unidades comparáveis do gradiente médio).
A norma original nessa atualização também era 7.78463. O diagnóstico mantém
o agendamento e aquecimento originais de 2,012 atualizações; reduzir `max_steps`
alteraria esses agendamentos e invalidaria a comparação. As 3,999 linhas
reservadas são excluídas. A saída usa novo diretório, e o estado de treinamento
é intencionalmente `incomplete`; não deve ser apresentado como classificador final.

Reproduza o diagnóstico limitado dentro da imagem ROCm validada:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/diagnostics/norberto/probe-training-prefix.py \
  --ep1-root /ep1 --updates 600 \
  --output-dir /workspace/results/norberto-v3-prefix-new
```

Um treinamento completo somente no desenvolvimento pode ser iniciado separadamente:

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 bash scripts/norberto/train.sh \
  --partition development \
  --output-dir /workspace/results/norberto-development-compatible-v3
```

A execução completa posterior concluiu duas épocas e 2,012 atualizações. Sua
[avaliação no conjunto reservado](../../results/norberto-recipe3-holdout.md)
registra a qualidade final e os resumos de gradientes da execução completa.
Um prefixo curto não estabelece estabilidade nem qualidade a longo prazo.
Esta correção não usou rótulos de teste1/teste2 nem substituiu adaptadores originais.

Para predição e avaliação separadas em teste1, veja os
[comandos em duas etapas](../../guides/norberto-test1.md). A predição aceita
este adaptador de desenvolvimento e registra sua identidade; a avaliação usa
somente as previsões persistidas, em CPU.
