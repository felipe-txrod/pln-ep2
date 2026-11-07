<a id="development-only-logistic-regression"></a>

# Regressão logística somente de desenvolvimento

`RLClassifier` é o membro `rl`. O padrão agora é `models/rl-development/`,
treinado nas 16,093 linhas fixas EP1, preservando 3,999 reservadas. O pacote
local concluiu treino/verificação: 38 iterações, 25.08 segundos, cobertura
exata e nenhuma sobreposição de grupos. Veja o
[modelo treinado](../../models/rl-development/README.md). A comparação lexical
anterior salvou só o SVM vencedor; `models/rl/` foi ajustado nas 20,092 linhas
e permanece histórico.

<a id="reproduce-the-selected-model"></a>

## Reproduzir o modelo selecionado

Na raiz do EP2, com imagem CPU offline existente e EP1 vizinho:

```bash
bash scripts/rl/train.sh --dry-run
bash scripts/rl/train.sh
```

Os comandos exigem novo diretório; não sobrescreva o pacote concluído.
O primeiro valida hashes, contagens e componentes disjuntos sem ajustar ou
gravar artefatos. O segundo ajusta só desenvolvimento, com construtor original
EP1 de TF-IDF palavras/caracteres e `C=0.25` fixo, L-BFGS, L2, 1,500 iterações,
semente 42 e duas threads. Sem busca, avaliação reservada, acesso a testes
oficiais ou ajuste completo. Não convergência gera falha. O treino grava
procedência segura, identidades exatas, joblib/metadados/configuração com
checksums e manifesto após recarregar/validar o pipeline exportado.

`RL_EP1_ROOT` e `RL_IMAGE` substituem cópia vizinha e imagem local.
Diretório existente é rejeitado. `--output-dir` seleciona novo pacote;
alternativas também precisam ser selecionadas explicitamente na inferência
e no manifesto do ensemble. Inferência não inicia treino.

<a id="inference"></a>

## Inferência

Use dependências fixadas em `requirements-lexical.txt`:

```bash
PYTHONPATH=src python3 -m rl_classifier predict \
  --input data/test2.xlsx --output-dir results/rl-development-test2
```

Para teste1, use `data/test1.xlsx` e `results/rl-development-test1`. A predição
lê só textos e retorna rótulos, probabilidades e escores nomeados por classe.
O ensemble confere o hash contra `models/rl-development/frozen_manifest.json`,
impedindo reutilizar previsões completas como de desenvolvimento.

Pacote de desenvolvimento ausente gera erro com orientação e nunca recorre
ao histórico completo. `RL_BUNDLE_DIR` seleciona alternativa explícita;
`--artifact`/`RL_ARTIFACT_PATH` selecionam outra cópia dentro do contrato
de integridade. Carregamento/predição continuam só em CPU.

Os pesos históricos completos de `models/rl/` ficam preservados. A acurácia
antiga de teste1 49.33% só vale para eles. O modelo de desenvolvimento foi
avaliado separadamente: 1,803/3,999 no conjunto reservado (45.09% acurácia,
44.78% macro-F1) e 447/900 em teste1 rotulado (49.67%, 49.32%). Linhas e
métricas foram verificadas independentemente, sem reajuste ou mudança de
parâmetros. Veja o [registro](../../docs/results/data/rl-development-evaluation.json).
