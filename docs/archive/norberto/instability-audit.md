<a id="norberto-instability-optimizer-and-backend-audit"></a>

# Instabilidade do NorBERTo: auditoria do otimizador e da implementação computacional

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

O adaptador corrigido treinado somente no desenvolvimento é instável, mas as
novas verificações **não identificam perda incorretamente normalizada por
amostra nem divisão adicional na acumulação**. O ciclo completo do Trainer
com modelo pequeno coincide com uma implementação manual independente em CPU,
inclusive quando executado em GPU. As derivadas elevadas do adaptador salvo
também ocorrem em CPU e concordam localmente com diferenças finitas. A
sensibilidade medida concentra-se no **LoRA da primeira atenção**.

A causa exata da trajetória de treinamento permanece sem solução. Passar nessas
verificações não certifica todo o pipeline nem recupera a qualidade do modelo.

<a id="complete-optimizer-loop-verification"></a>

## Verificação completa do ciclo do otimizador

``audit-norberto-optimizer.py`` usa um ModernBERT aleatório de duas camadas com
PEFT, 29 exemplos sintéticos e dropout zero. Executa quatro atualizações em
duas épocas, alternando acumulação de lotes `4+4+4+4` e `4+4+4+1`. O contexto
de injeção da perda de produção é usado sem alterações. O ciclo completo
`Trainer.train()` é comparado com outro modelo e entropia cruzada, limitação
de gradiente e atualizações AdamW manuais em CPU.

| Verificação | Trainer em CPU | Trainer em GPU contra referência em CPU |
|---|---:|---:|
| Atualizações verificadas | 4 | 4 |
| Erro relativo máximo do gradiente | 9.48e-8 | 2.57e-7 |
| Diferença absoluta máxima dos parâmetros | 1.50e-8 | 1.50e-8 |
| Norma máxima do gradiente após limitação | < 1.00000 | < 1.00000 |
| Parâmetros congelados inalterados | Sim | Sim |

Os totais reais dos lotes são 16 e 13, respectivamente; cada microlote recebe
a contagem acumulada correta. A perda acumulada coincide com a entropia cruzada
média manual. O teste verifica independentemente as quatro taxas de aprendizado
do agendador linear, limpeza dos gradientes entre atualizações e preservação dos
parâmetros congelados. O divisor de acumulação do acelerador é 1 nesta
integração com Trainer; a perda personalizada já trata os quatro microlotes.
Não há divisão adicional por quatro na execução verificada.

As suítes existentes de perda e partição também passaram: **18 testes em 5.19
segundos**. Esses testes fornecem evidência específica de integração, não prova
de todas as atualizações de um treinamento completo.

<a id="real-saved-adapter-gradients-on-cpu-and-gpu"></a>

## Gradientes do adaptador real salvo em CPU e GPU

``audit-norberto-backends.py`` selecionou as primeiras quatro respostas por
classe com até 64 tokens das linhas compartilhadas de treino de desenvolvimento.
Esse lote fixo tem 12 exemplos, preenchidos até 64 tokens. O dropout foi
desativado para isolar diferenças numéricas. Nenhum otimizador foi construído
para qualquer modelo de produção.

| Adaptador | Norma CPU SDPA | Norma CPU eager | Norma GPU SDPA |
|---|---:|---:|---:|
| EP1 original | 1.84432 | 1.84432 | 1.84432 |
| Retreinamento corrigido no corpus completo | 1.71013 | 1.71012 | 1.71011 |
| Retreinamento corrigido somente no desenvolvimento | 158.82942 | 158.83560 | 158.98747 |

Para o adaptador somente de desenvolvimento, CPU eager difere de CPU SDPA em
**4.30e-5** de diferença relativa do vetor de gradientes. GPU SDPA difere de
CPU SDPA em **0.001058**, com diferença máxima de logits **0.0003555**. O
gradiente elevado está presente nas três implementações. Assim, uma falha
exclusiva da substituição de GPU ou de SDPA é insuficiente para explicar a
sensibilidade medida no checkpoint. Isso não exclui efeitos da implementação
computacional durante o treino.

Todos os tensores carregados coincidem exatamente com os salvos após os testes;
os três hashes dos pesos permanecem inalterados. Duas novas construções a
partir da base pública fixada e semente 42 reproduzem o mesmo hash dos parâmetros
treináveis no ambiente atual. Isso não prova identidade histórica bit a bit.

<a id="local-derivative-verification-and-first-layer-localization"></a>

## Verificação local de derivadas e localização na primeira camada

``audit-norberto-sensitivity.py`` recarregou o adaptador somente de desenvolvimento
em CPU e usou as mesmas 12 respostas de treino. A norma medida foi aproximadamente
**158.83**, com **99.911% da norma quadrática do gradiente** nos parâmetros LoRA
da primeira camada de atenção.

Uma coordenada com a maior derivada absoluta foi verificada por diferenças
finitas centrais em FP32, considerando o deslocamento realmente representável
do parâmetro:

| Perturbação | Derivada por diferenças finitas | Derivada Autograd | Erro relativo |
|---|---:|---:|---:|
| 0.001 | 17.44067 | 18.15042 | 3.9103% |
| 0.0001 | 18.18004 | 18.15042 | 0.1632% |
| 0.00001 | 18.24995 | 18.15042 | 0.5484% |

Isso verifica uma direção sensível real no modelo salvo; não verifica todas as
coordenadas da retropropagação. Definir temporariamente em memória como zero
apenas o multiplicador do LoRA da primeira atenção reduziu a norma total para
**2.42088**. Em seguida, o multiplicador e os parâmetros foram restaurados, e
todos os tensores do adaptador foram conferidos contra o checkpoint salvo.

A ablação localiza a sensibilidade observada. Não é um adaptador substituto
avaliado, uma afirmação de melhora de acurácia nem evidência de que a primeira
camada tenha iniciado a regressão do treinamento.

<a id="loss-normalization-changes-the-clipping-regime"></a>

## A normalização da perda altera o regime de limitação de gradientes

Manter a mesma configuração numérica após corrigir a perda não preserva toda
a trajetória histórica de otimização. Com microlotes iguais, o gradiente
histórico era aproximadamente `4g`, sendo `g` o gradiente correto da perda média.
As duas receitas usam `max_grad_norm=1`.

Por exemplo, considere `g=(0.3, 0.4)`:

| Receita | Norma bruta do gradiente | Gradiente aproximado após limitação |
|---|---:|---|
| Acumulação histórica | 2.0 | `(0.6, 0.8)` |
| Perda média correta | 0.5 | `(0.3, 0.4)` |

Se os dois gradientes brutos excedem o limiar, os vetores limitados são quase
idênticos. Se ficam em lados opostos do limiar, diferem. Conforme essa condição
muda entre atualizações, as estimativas dos momentos do AdamW podem evoluir de
modo diferente. Ignorando estabilizadores numéricos, `clip(4g, 1) = 4 * clip(g, 0.25)`;
manter o limiar corrigido em 1 muda seu regime efetivo de limitação. Epsilon do
Adam, microlotes finais desiguais e diferenças de ponto flutuante também impedem
afirmar treinamento idêntico em geral. Nenhum parâmetro de limitação ou taxa de
aprendizado mudou nesta auditoria.

Os registros históricos de desenvolvimento coincidem inicialmente após
considerar o fator quatro: na atualização 20, a norma original dividida por
quatro é 3.8596323, contra 3.8596320 na execução corrigida. As perdas coincidem
na precisão do registro. Na atualização 80, as normas correspondentes são
4.6441612 e 4.6441574. As diferenças crescem depois; a primeira norma corrigida
registrada acima de 25 é **253.149 na atualização 540, época 0.5368**. Os
registros são amostrados a cada 20 atualizações, portanto não identificam a
primeira atualização instável exata nem sua causa.

A interação com a limitação é uma diferença matemática estabelecida entre as
receitas e um mecanismo candidato para trajetórias distintas. **Não é uma
explicação demonstrada para as 386 previsões corretas perdidas no conjunto reservado**.

Os detalhes da implementação são sustentados pelo
[código do Trainer no Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py),
[código de classificação do ModernBERT](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/modernbert/modeling_modernbert.py)
e documentação PyTorch 2.8 sobre
[limitação de gradientes](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html)
e [AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).

<a id="scope-and-remaining-causal-question"></a>

## Escopo e questão causal pendente

Todos os modelos de produção e entradas foram montados somente para leitura,
com rede desativada. Não houve atualização de otimizador em modelos completos,
nova inferência no conjunto reservado, acesso a teste1/teste2, substituição de
modelo ou alteração de ensemble. Somente os pequenos modelos aleatórios foram
atualizados. A correção confirmada da acumulação permanece inalterada.

Ainda é necessária uma reprodução limitada somente no treino, rastreando cada
atualização próxima à primeira divergência, para separar efeitos da limitação,
sensibilidade numérica, ordem dos dados e outros problemas da implementação.
A evidência atual não justifica afirmar um novo erro de perda confirmado nem
uma regressão totalmente explicada. Outro treinamento completo não substitui
esse isolamento.

A [correção posterior das unidades do otimizador](recipe3-optimizer-units.md)
e sua [avaliação concluída no conjunto reservado](../../results/norberto-recipe3-holdout.md)
documentam um prefixo controlado e o acompanhamento com execução completa.
A receita 3 reproduziu as normas históricas até a atualização 1,000, evitou a
escalada anterior e recuperou acurácia de 44.81% no conjunto reservado. A
auditoria acima descreve a evidência disponível antes dessa correção.

Dentro da imagem validada, com projeto, EP1 e cache montados, os diagnósticos
podem ser reproduzidos com novos caminhos de saída:

```bash
PYTHONPATH=src python3 scripts/diagnostics/norberto/audit-optimizer.py \
  --device cpu --output results/new-audit/optimizer-cpu.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/diagnostics/norberto/audit-optimizer.py \
  --device gpu --output results/new-audit/optimizer-gpu.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/diagnostics/norberto/audit-backends.py --ep1-root /ep1 \
  --output results/new-audit/backends.json
# Run this final command in a CPU-only container without GPU device exposure.
PYTHONPATH=src python3 scripts/diagnostics/norberto/audit-sensitivity.py --ep1-root /ep1 \
  --backend-report results/new-audit/backends.json \
  --output results/new-audit/sensitivity.json
```

Os comandos de implementação computacional e sensibilidade exigem um diretório
pai de saída existente. Saídas brutas estão em `results/norberto-deep-audit-20261105/`.
O [agregado público](../../results/data/norberto-deep-audit-results.json) registra
valores medidos, hashes, escopo dos testes e limitações. A
[validação anterior do conjunto reservado](recipe2-development-holdout.md) permanece inalterada.
