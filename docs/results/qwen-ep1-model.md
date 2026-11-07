<a id="selected-qwen-exact-ep1-development-partition"></a>

# Qwen selecionado: partição exata de desenvolvimento do EP1

[Índice dos resultados](README.md) · [Pacote do modelo](../../models/qwen/README.md)
· [Agregado de verificação](data/qwen-ep1-selected-model.json)

A execução concluída `results/qwen-ep1-training-20261105/` é o modelo Qwen
padrão. O adaptador selecionado foi copiado, sem alterar pesos, para
`models/qwen/adapter/`, com tokenizador completo, procedência, métricas de
validação, divisão congelada e manifesto SHA-256.

<a id="training-and-validation"></a>

## Treinamento e validação

A configuração é Qwen3-1.7B revisão
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, QLoRA NF4, posto 8/alpha 16,
contexto 256, lote 4 e acumulação 4, por três épocas. As 3,018 atualizações
terminaram sem passos ignorados por estouro FP16. O treino levou 11,273.37
segundos, cerca de 3 h 8 min. O checkpoint selecionado é `checkpoint-3018`,
e seus pesos coincidem exatamente com o adaptador preservado.

| Partição | Linhas | Uso |
|---|---:|---|
| Desenvolvimento congelado do EP1 | 16,093 | Ajuste dos pesos |
| Conjunto reservado congelado do EP1 | 3,999 | Avaliação e seleção do melhor checkpoint |

Todos os 20,092 textos, rótulos e IDs foram conferidos. IDs de treino coincidem
com a RL de desenvolvimento; os de validação, com a divisão congelada.
Linhas, grupos e textos normalizados não cruzam a fronteira do treino.

A validação obteve **1,791/3,999 acertos (44.79%)**, macro-F1 **44.44%**.
A matriz, com classes verdadeiras nas linhas e preditas nas colunas em ordem
`c1, c234, c5`, é:

```text
[[571, 359, 327],
 [432, 463, 522],
 [260, 308, 757]]
```

O conjunto reservado foi consultado historicamente e seleciona o checkpoint
Qwen; o resultado é validação, não teste final inédito. Teste1 e teste2 não
foram avaliados durante a preservação deste modelo.

<a id="per-row-export-ordering-audit"></a>

## Auditoria da ordem da exportação por linha

O Trainer instalado aplica `group_by_length` também à avaliação. As métricas
agregadas pareiam corretamente logits e rótulos do Trainer, mas a exportação
original associava essas saídas reordenadas aos registros sequenciais originais.
Consequentemente, o `validation-predictions.json` histórico tem associações
incorretas entre linhas e previsões. Não deve ser usado para análise por linha.

A permutação exata foi recuperada em CPU usando
`checkpoint-3018/rng_state.pth` salvo localmente e o amostrador original
agrupado por comprimento. A ordem reconstruída muda 3,998 de 3,999 posições
e reproduz toda a matriz registrada, incluindo 1,791 acertos. Previsões
alinhadas e permutação foram salvas separadamente em
`results/qwen-ep1-publication-20261105/`; fonte histórica e pesos foram preservados.

Treinos futuros mantêm agrupamento por comprimento nos lotes de treino, mas
avaliação sequencial. Uma verificação explícita da ordem dos rótulos impede
exportações desalinhadas, e um teste real do Trainer em CPU verifica a ordem
com comprimentos variados. Não foi necessário retreinar nem alterar pesos
para preservar o adaptador.

<a id="default-inference-and-artifact-verification"></a>

## Inferência padrão e verificação dos artefatos

Todos os arquivos do pacote foram conferidos por checksum. Os 225 tensores
salvos são finitos, incluindo a cabeça treinada de 3 por 2,048. O padrão
selecionado foi recarregado na GPU ROCm real e classificou três textos
sintéticos com logits finitos e probabilidades normalizadas; os arquivos
não mudaram.

A predição agora usa o pacote salvo por padrão:

```bash
QWEN_IMAGE=pln-rocm:6.4.1 \
QWEN_RUNTIME_PATH=/workspace/.cache/qwen-runtime \
bash scripts/qwen/container.sh predict \
  --input data/test2.xlsx --output-dir results/qwen-ep1-test2
```

Este exemplo exige novo diretório e executa inferência quando chamado
explicitamente. Não foi executado na preservação do modelo. Use `--adapter`
apenas para alternativa explícita. O script separado de teste1 também usa o
pacote verificado por padrão; `--training-run` continua selecionando execuções
históricas explicitamente, com suas verificações originais de identidade.

A inferência posterior da [integração de votação suave e stacking](ensemble-qwen.md)
acertou **414/900 (46.00%)** em teste1, macro-F1 **45.53%**. Qwen participa
desses pipelines, enquanto os presets de maioria mantêm três votantes.
Seus caminhos usam `models/qwen/adapter` e os diretórios selecionados
`results/qwen-ep1-test1`/`-test2`, sem reutilizar previsões do adaptador antigo.
Teste2 não foi gerado nesta integração.

Para reproduzir a exportação preservando bytes da execução local original:

```bash
python3 scripts/models/freeze.py --model qwen \
  --training-run results/qwen-ep1-training-20261105
```
