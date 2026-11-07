<a id="frozen-inference-models"></a>

# Modelos congelados de inferência

Este diretório distribui artefatos finais de inferência sem retreinamento.
Pesos treinados, vocabulários TF-IDF, tokenizadores e configurações podem ser
versionados no Git. Cada pacote inclui `manifest.json`, que associa os
arquivos de inferência a checksums SHA-256.

| Diretório | Arquivos versionados | Dependência externa |
|---|---|---|
| [svm/](svm/README.md) | Pipeline treinado TF-IDF + LinearSVC, `C=0.05`, em `model.joblib` | Nenhuma |
| [rl-development/](rl-development/README.md) | Pipeline logístico verificado somente de desenvolvimento, `C=0.25`, treinado em 16,093 linhas | Nenhuma para inferência |
| [rl/](rl/README.md) | Pipeline logístico histórico do corpus completo; não é o padrão | Nenhuma |
| [norberto/](norberto/README.md) | Adaptador LoRA, cabeça, tokenizador e configuração | Base pública NorBERTo, revisão e checksums fixados |
| [qwen/](qwen/README.md) | Adaptador QLoRA selecionado na divisão EP1, cabeça, tokenizador e procedência com checksums | Base pública Qwen3-1.7B na revisão fixada |
| [gpt/](gpt/README.md) | Cópias de configuração e prompts | Serviço remoto; pesos indisponíveis localmente |
| [ensemble-soft-qwen/](ensemble-soft-qwen/README.md) | Calibração padrão de quatro candidatos e pesos suaves; Qwen selecionado com peso zero | SVM, RL de desenvolvimento, NorBERTo original e Qwen na divisão EP1 selecionados |
| [ensemble-stacking-qwen/](ensemble-stacking-qwen/README.md) | Meta-RL padrão com doze probabilidades calibradas | Quatro pacotes selecionados e calibradores congelados |
| [stacking-gpt-cascade/](stacking-gpt-cascade/README.md) | Experimento congelado de roteamento seletivo; validação escolheu zero chamadas GPT | Stacking inalterado de quatro membros e Codex GPT 6.1 Sol Low opcional |
| [ensemble-soft/](ensemble-soft/README.md) | Calibração original de três modelos, pesos e papéis reservados preservados | SVM, RL de desenvolvimento e NorBERTo original |
| [ensemble-stacking/](ensemble-stacking/README.md) | Meta-RL multinomial original de três modelos e procedência da validação agrupada preservadas | Nove probabilidades calibradas originais |

Qwen agora usa `models/qwen/adapter/` por padrão; diretórios históricos são
preservados. Checkpoints Qwen, estados do otimizador e bases públicas ficam
fora do pacote versionado. Participa de votação suave e stacking; presets
separados de maioria mantêm NorBERTo, SVM e RL.

<a id="artifact-formats"></a>

## Formatos dos artefatos

Os `.joblib` já contêm coeficientes aprendidos, vocabulários e IDF. O arquivo
`.safetensors` do adaptador contém parâmetros LoRA e cabeça treinada.
Não é necessário treino, conversão GGUF ou fusão do adaptador.

As cópias preservam bytes originais. Os carregadores conferem hashes, classes
e configuração antes dos pesos. SVM, regressão logística e NorBERTo usam seus
pacotes em `models/`. RL usa `rl-development/` e falha se o pacote de
desenvolvimento não tiver sido treinado. Não recorre ao histórico completo `rl/`.

<a id="restore-the-public-base"></a>

## Restaurar a base pública

Os pesos-base NorBERTo ficam fora do Git. `norberto/base_reference.json`
fixa repositório público, commit e SHA-256 de cada arquivo necessário.
Após clonar, restaure ausentes sem carregar o modelo:

```bash
python3 scripts/models/prepare-base.py
```

O comando usa só a biblioteca padrão Python e confere downloads contra o
contrato congelado. Verifique uma cópia existente sem rede:

```bash
python3 scripts/models/prepare-base.py --verify-only
```

`models/norberto/base_model/model.safetensors` fica disponível localmente
após preparar e continua ignorado pelo Git. Restaurar exige acesso à revisão
pública fixada; o adaptador sozinho não substitui a base.

<a id="export-and-versioning"></a>

## Exportação e versionamento

O exportador copia artefatos existentes e confere hashes. Não importa
bibliotecas de modelos, desserializa pesos, treina ou faz inferência:

```bash
python3 scripts/models/freeze.py --model svm
python3 scripts/models/freeze.py --model rl
python3 scripts/models/freeze.py --model norberto
python3 scripts/models/freeze.py --model gpt
python3 scripts/models/freeze.py --model qwen \
  --training-run results/qwen-ep1-training-20261105
```

O exportador `--model rl` preserva o modelo histórico completo em `models/rl/`;
não seleciona nem gera a RL de desenvolvimento. Exportações usam artefatos
locais preservados do EP1. Uma nova clonagem recebe arquivos de inferência
versionados e não precisa exportá-los novamente. Arquivos idênticos são aceitos;
existentes diferentes nunca são sobrescritos. Use novo `--output` para outro
pacote. Para NorBERTo, `--with-local-base` também copia pesos-base locais,
mantendo-os ignorados pelo Git.

Os artefatos treinados cabem no Git comum; Git LFS é dispensável para eles.
GitHub bloqueia arquivos individuais acima de 100 MiB; veja os
[limites de tamanho](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Bases públicas grandes permanecem externas.

Use versões do ambiente registradas nos contratos/dependências. Mantenha
caches, checkpoints intermediários, estados dos otimizadores, credenciais e
registros brutos fora dos pacotes versionados.
