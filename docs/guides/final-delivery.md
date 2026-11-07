# Gerar o Excel final do EP2

[README](../../README.md) · [Índice dos scripts](../../scripts/README.md)

A entrega usa o **Soft Voting congelado SVM 75% / NorBERTo 25%**, em
`models/ensemble-soft/model.json`. O fluxo faz inferência e exportação, sem
treinar modelos, recalibrar probabilidades, ajustar pesos ou chamar GPT.
Docker não é necessário.

## Preparar o ambiente nativo

**CPU é o padrão tanto na preparação quanto na execução.** Na raiz do EP2,
o comando sem parâmetros prepara o ambiente completo para CPU em `.venv-delivery/`:

```bash
bash scripts/ensemble/setup-local.sh
```

O padrão instala PyTorch 2.8.0+cpu, a base de `requirements-delivery.txt` e as
bibliotecas neurais de `requirements-delivery-neural.txt`. Ele permite aplicar
o SVM e o NorBERTo sem GPU. Para instalar apenas a base lexical e exportar
previsões já calculadas, escolha `--minimal`.

| Opção do setup | Efeito |
|---|---|
| Sem opções ou `--with-cpu` | Instala PyTorch 2.8.0+cpu e as bibliotecas neurais |
| `--minimal` ou `--without-rocm` | Instala somente a base; não instala nem substitui PyTorch |
| `--with-neural` | Instala Transformers/PEFT/Accelerate, exigindo PyTorch 2.8.0 já visível no ambiente |
| `--without-neural` | Exclui as bibliotecas neurais |
| `--with-rocm` | Instala PyTorch 2.8.0+rocm6.4 e, por padrão, as bibliotecas neurais |
| `--venv CAMINHO` | Cria ou atualiza o ambiente indicado |
| `--system-site-packages` | Em ambiente novo, permite reutilizar pacotes do Python-base |
| `--dry-run` | Exibe o plano sem criar ambiente, instalar ou baixar pacotes |

As opções de ROCm e CPU são mutuamente exclusivas. É possível combinar
`--with-rocm --without-neural` para instalar somente o backend e a base.
Nenhum perfil instala ou altera o driver ROCm do sistema.

Para conferir o que seria instalado:

```bash
bash scripts/ensemble/setup-local.sh --dry-run
```

Para um ambiente GPU completo novo, selecione explicitamente:

```bash
bash scripts/ensemble/setup-local.sh --with-rocm
```

Somente esse perfil baixa PyTorch ROCm, a partir do
[índice oficial documentado pelo PyTorch](https://pytorch.org/get-started/previous-versions/#v280).
O Python-base padrão é 3.12; para usar 3.11, defina `EP2_SETUP_PYTHON=python3.11`.

### CPU e GPU na execução

A CPU continua disponível em todos os perfis. SVM e RL usam CPU, tanto no
fluxo com ROCm quanto no fluxo sem GPU. O parâmetro `--with-cpu` seleciona
uma distribuição do PyTorch sem suporte a GPU; não habilita ou desabilita o
processador da máquina.

NorBERTo precisa de PyTorch para executar. O perfil mínimo instala apenas as
dependências lexicais e de exportação: sem PyTorch já disponível no ambiente,
ele não permite classificar novos textos com NorBERTo.

Para executar tudo em CPU, basta usar os padrões:

```bash
bash scripts/ensemble/setup-local.sh
bash scripts/ensemble/deliver-test2.sh
```

Para usar GPU no NorBERTo e CPU nos modelos lexicais:

```bash
bash scripts/ensemble/setup-local.sh --with-rocm
bash scripts/ensemble/deliver-test2.sh --device gpu
```

A distribuição ROCm do PyTorch também executa operações em CPU. Se ela já
estiver instalada, é possível usar `deliver-test2.sh --device cpu` sem
instalar uma segunda distribuição do PyTorch.

Para a opção GPU, ROCm instalado no sistema e PyTorch compilado para ROCm
são requisitos distintos. A versão CUDA do PyTorch não utiliza a GPU AMD.
O executor verifica essa diferença quando `--device gpu` é solicitado.

Para acrescentar bibliotecas neurais a um ambiente que já possui PyTorch,
sem reinstalá-lo:

```bash
bash scripts/ensemble/setup-local.sh --with-neural \
  --venv /caminho/do/ambiente
```

Essa opção atualiza as dependências do ambiente indicado. Antes de qualquer
instalação de pacotes, verifica se PyTorch 2.8.0 já está presente. Se estiver
ausente, encerra com erro em vez de baixá-lo por dependência indireta.
O arquivo `requirements-delivery-neural.txt` mantém a restrição dessa versão.
Ambientes novos são isolados; reutilizar um PyTorch global exige
`--system-site-packages`. Essa opção não altera a visibilidade de um ambiente
isolado já existente: escolha outro `--venv` nesse caso.

Se já possui um ambiente completo compatível, basta indicá-lo ao executor:

```bash
EP2_DELIVERY_PYTHON=/caminho/do/ambiente/bin/python \
bash scripts/ensemble/deliver-test2.sh --check
```

A seleção do interpretador segue esta ordem: `EP2_DELIVERY_PYTHON`, ambiente
`.venv-delivery/`, ambiente ativo em `VIRTUAL_ENV`, `.venv/`, Python 3.12 e
Python 3 do sistema. Os caches binários de outros ambientes não são misturados.

## Verificar e gerar a planilha

Para verificar dependências, os arquivos congelados e o formato da entrada
sem classificar os textos:

```bash
bash scripts/ensemble/deliver-test2.sh --check
```

Essa verificação usa CPU por padrão. Para verificar também a GPU, use
`--check --device gpu`.

Para gerar a entrega:

```bash
bash scripts/ensemble/deliver-test2.sh
```

O script resolve a raiz do projeto a partir de seu próprio caminho e define
`PYTHONPATH`. O comando também funciona a partir de outra pasta se for chamado
pelo caminho absoluto do script. Isso evita `ModuleNotFoundError` causado
pelo diretório atual ou por um projeto que ainda não foi instalado no Python.

Arquivos utilizados e produzidos:

| Finalidade | Caminho |
|---|---|
| Entrada original | `data/test2.xlsx` |
| Previsões da execução e escores por membro | `results/soft-voting-test2-final/` |
| **Excel para entregar** | **`deliveries/test2-soft-voting.xlsx`** |
| Comprovante de verificações e hashes | `results/soft-voting-test2-final/delivery.json` |

A execução padrão usa CPU em todos os modelos. Com `--device gpu`, NorBERTo
usa GPU e os modelos lexicais continuam em CPU. Para a RX 6650 XT, o executor
mantém o ajuste já utilizado no projeto,
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. Uma variável fornecida pelo usuário tem
precedência. A inferência neural pode levar alguns minutos; não há novo treino.

O Excel é exportado somente após verificar todas as 897 previsões e seus
IDs, textos, classes e probabilidades. Apenas a coluna `clarity` é preenchida;
ordem, textos, estilos e demais partes do arquivo original são preservados.
A entrada precisa ter `clarity` vazia. O arquivo original não é sobrescrito.

## Executar novamente ou exportar resultados já calculados

Cada execução do comando de geração recalcula as previsões de todos os membros,
sem reutilizar escores salvos. Após a validação, substitui o conteúdo de
`results/soft-voting-test2-final/` e sobrescreve `deliveries/test2-soft-voting.xlsx`.
Não é necessário apagar arquivos nem fornecer uma opção adicional.

A inferência e a exportação são preparadas em diretórios temporários. Se houver
falha nessa preparação ou na validação, a última execução concluída e o Excel
anterior são preservados. Se houver interrupção, repetir o comando inicia uma
nova inferência completa. Modelos treinados, a entrada original e caches de
pesos e bibliotecas permanecem disponíveis; somente as previsões são refeitas.
Um diretório de resultados identificado como pertencente a outra entrada ou
modelo é recusado; nesse caso, escolha outro `--run`.

Se a predição já terminou e faltou somente o Excel, exporte sem carregar PyTorch:

```bash
bash scripts/ensemble/deliver-test2.sh --export-only
```

Essa opção exige as dependências lexicais e um `run.json` concluído e compatível.
Ela reutiliza as previsões concluídas somente para exportação, sem executar os
modelos. O Excel é sempre gerado novamente, verificado e substituído, mesmo
que já exista um arquivo no destino.

Para usar CPU explicitamente:

```bash
bash scripts/ensemble/deliver-test2.sh --device cpu
```

Para outros destinos, preservando resultados anteriores:

```bash
bash scripts/ensemble/deliver-test2.sh \
  --run results/soft-voting-test2-outra-execucao \
  --output deliveries/test2-outra-execucao.xlsx
```

## Erros antes da inferência

- `No module named ensemble_classifier`: use o executor acima; ele configura
  o caminho de importação independentemente da pasta atual.
- Dependência ausente ou versão diferente: prepare `.venv-delivery/` ou indique
  outro interpretador com `EP2_DELIVERY_PYTHON`. Para dependências neurais,
  escolha `--with-neural`, `--with-rocm` ou `--with-cpu` no setup.
- PyTorch sem ROCm: instale a distribuição ROCm no ambiente dedicado; ter
  somente ROCm no sistema não troca uma distribuição CUDA já instalada.
- GPU indisponível: confira acesso a `/dev/kfd` e `/dev/dri`, os grupos
  `render`/`video` e as permissões da sessão. O script não altera essas permissões.
- Base NorBERTo ausente: restaure os arquivos públicos fixados com
  `.venv-delivery/bin/python scripts/models/prepare-base.py`.

O fluxo de entrega não lê o gabarito de Teste1 e não calcula métricas para
Teste2. Avaliações estatísticas continuam separadas da geração do Excel.
