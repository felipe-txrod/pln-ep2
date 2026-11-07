<a id="investigation-of-the-full-corpus-norberto-regression"></a>

# Investigação da regressão do NorBERTo no corpus completo

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

Acompanhamento: o [retreinamento normalizado concluído](recipe2-full-test1.md)
obteve 408/900 (45.33%), contra 341/900 do retreinamento completo histórico e
432/900 da entrega original do EP1. A investigação inicial abaixo permanece
um diagnóstico restrito ao treino; o acompanhamento é uma comparação separada
em teste1.

A evidência aponta **otimização instável durante a execução no corpus completo**.
Um erro separado de normalização da perda foi reproduzido, mas afeta ambas as
execuções históricas e não é causa única comprovada da regressão. Nenhum modelo
completo foi retreinado, não houve atualização do otimizador e os diagnósticos
não acessaram teste1.

<a id="verified-inputs-and-parameters"></a>

## Entradas e parâmetros verificados

Os dois leitores retornaram textos e rótulos idênticos nas 20,092 linhas.
IDs de tokens, máscaras e IDs de tipos de tokens coincidiram nas 16,093 linhas
compartilhadas de desenvolvimento. Os argumentos numéricos de treino salvos
coincidiram; só caminhos de saída/registro diferiram. As contagens em ordem
`c1/c234/c5` eram 5,090/5,436/5,567 no desenvolvimento e 6,347/6,853/6,892 no
corpus completo. Os hashes dos pesos da base e do adaptador coincidiram.

O código original do treinador foi recuperado do commit do EP1
`fcae20328b49ef312b39babbe41a67acfd6acb1c`, coincidindo com o SHA-256 registrado.
Mudanças posteriores trataram seleção de dispositivo, registros, organização
e inferência. Inicialização, tokenização supervisionada e argumentos numéricos
de treino permaneceram equivalentes no caminho de GPU selecionado. Isso não
estabelece execução histórica idêntica bit a bit nem exclui todas as diferenças
entre implementações computacionais.

<a id="instability-already-present-in-training"></a>

## Instabilidade já presente no treinamento

Médias das normas registradas dos gradientes, antes da limitação:

| Intervalo de épocas | Treinamento original | Treinamento no corpus completo |
|---|---:|---:|
| 0–0.5 | 11.51 | 15.59 |
| 0.5–1 | 8.09 | 106.16 |
| 1–1.5 | 7.58 | 368.82 |
| 1.5–2 | 8.14 | 319.61 |

As normas máximas foram 22.58 e 1,221.64. O primeiro registro completo acima de
100 ocorreu no passo 1,100, época 0.876: norma 1,090.99, taxa 0.00062522.
A limitação estava ativa em 1.0; essas normas não são passadas diretamente ao
AdamW. A perda média registrada ao final foi 4.1214 contra 4.4352. Essas perdas
históricas são escaladas pelo defeito abaixo e não são entropia cruzada comum
por resposta.

<a id="training-only-probes"></a>

## Diagnósticos restritos ao treinamento

Uma amostra fixa usou 300 respostas compartilhadas de treino, 100 por classe,
semente 42:

| Métrica | Adaptador original | Retreinamento completo |
|---|---:|---:|
| Acertos | 151 | 111 |
| Acurácia na amostra de treino | 50.33% | 37.00% |
| Entropia cruzada média | 0.989255 | 1.103177 |

São diagnósticos de ajuste, não estimativas de generalização. Em outro lote
fixo de 12 respostas curtas de treino, sem dropout nem otimizador:

| Medida | Adaptador original | Retreinamento completo |
|---|---:|---:|
| Norma do gradiente em CPU | 1.8443 | 189.7469 |
| Norma do gradiente em GPU | 1.8443 | 190.1226 |
| Diferença máxima de logits CPU/GPU | 0.00000501 | 0.00034493 |
| Diferença relativa do gradiente CPU/GPU | 0.00119% | 0.40856% |

Gradientes elevados também ocorrem em CPU, enfraquecendo a explicação baseada
apenas em inferência incorreta na GPU. Um diagnóstico pequeno não certifica
todos os núcleos computacionais de treinamento. Todos os tensores coincidiram
com os salvos após os cálculos. As normas médias das atualizações LoRA foram
semelhantes: 6.8376 e 6.8396; não se demonstrou explosão uniforme da magnitude
dos parâmetros.

<a id="confirmed-loss-normalization-bug"></a>

## Erro confirmado de normalização da perda

O Trainer interpreta `**kwargs` como suporte à contagem acumulada de exemplos.
A perda nativa de classificação do ModernBERT calcula a média do microlote e
ignora essa contagem. Veja a [implementação primária do Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py)
e a [implementação do ModernBERT](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/modernbert/modeling_modernbert.py).

Cálculos sintéticos de propagação e retropropagação com ModernBERT/PEFT em CPU reproduziram:

| Tamanhos dos microlotes | Perda histórica / correta | Norma do gradiente histórica / correta |
|---|---:|---:|
| 4 + 4 + 4 + 4 | 4.0000 | 4.0000 |
| 4 + 4 + 4 + 1 | 4.1404 | 5.3711 |

A cauda desigual também distorceu a direção do gradiente, com erro de direção
normalizada 0.4244. A soma da entropia cruzada dividida pelo total de exemplos
coincidiu com um lote completo, com erro relativo de gradiente abaixo de 0.0000001.

**O erro não é causa única comprovada da perda de acurácia.** Existia nas duas
execuções. Uma escala constante em microlotes iguais é em grande parte anulada
quando ambos os gradientes são limitados à mesma norma. Não se deve descrever
isso simplesmente como multiplicar a taxa do AdamW por quatro. Caudas desiguais
e precisão finita continuam relevantes.

<a id="causal-limits-and-correction"></a>

## Limites causais e correção

Duas épocas com mais dados aumentaram as atualizações de **2,012 para 2,512
(+24.85%)**, o aquecimento de 202 para 252 e mudaram a ordem de amostragem,
mantendo taxa 0.001. Hiperparâmetros iguais não preservaram a trajetória de
otimização. O retreinamento completo chegou a um estado mal condicionado e
com ajuste ruim. O gatilho específico entre composição dos dados, ordem,
orçamento de atualizações, sensibilidade computacional e suas interações não
foi isolado por retreinamento controlado.

O invólucro de treinamento agora registra receita versão 2 e usa perda explícita
normalizada por amostra de [training_loss.py](../../../src/norberto_classifier/training_loss.py).
Hiperparâmetros numéricos e fontes do EP1 permanecem inalterados; o construtor
histórico do Trainer é restaurado após a chamada. Artefatos históricos foram
preservados. Nove testes específicos em CPU passaram, incluindo equivalência
real de gradientes ModernBERT/PEFT/Trainer em cauda desigual. A simulação do
corpus completo passou sem importar modelos. Naquele estágio, nenhum treino
completo nem reavaliação de teste1 havia usado a correção; o acompanhamento
posterior está vinculado acima.

Experimentos causais adicionais devem isolar normalização, inclusão de dados e
orçamento de atualizações usando controles somente de desenvolvimento e
validação interna. Teste1 não pode escolher hiperparâmetros nem regras do
ensemble enquanto for chamado de não consultado.

Scripts: `CPU audit` e `training-only stability probe`.
O [agregado público](../../results/data/norberto-training-investigation-results.json)
contém as medições; os relatórios brutos estão em
`results/norberto-full-investigation-20261105/`.
