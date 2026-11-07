<a id="selected-qwen-model"></a>

# Modelo Qwen selecionado

Este pacote é o classificador padrão Qwen3-1.7B. Preserva adaptador QLoRA,
cabeça e tokenizador exatos de `results/qwen-ep1-training-20261105/`.
O treino concluiu três épocas e 3,018 atualizações na partição congelada EP1:
16,093 linhas de treino e 3,999 de validação, sem linhas/grupos compartilhados.

O adaptador está em `adapter/`. O SHA-256 dos pesos brutos é
`c419c68572ab8e5cd6bda6730714ec27cb28d2d7374d5954a812469346427e53`;
a identidade de inferência, incluindo configuração e tokenizadores, é
`a045c5726a85c466914c55f9cee9fb0d412af4d66387e14b590e53b4782beed5`.
`manifest.json` fixa cada arquivo de inferência e procedência do pacote.

A base pública permanece externa, fixada em Qwen/Qwen3-1.7B revisão
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, restaurada por `base_reference.json`
no cache existente `.cache/qwen-model/`. O adaptador salvo não é modelo fundido.
Checkpoints e estado do otimizador ficam no diretório original de treino,
ignorado pelo Git.

`qwen_classifier predict` usa este pacote verificado quando `--adapter` é
omitido. Alternativas históricas explícitas continuam disponíveis. Arquivos
padrão ausentes/alterados falham na validação, sem selecionar modelo antigo.
Entradas opcionais Qwen de votação suave/stacking apontam aqui. Presets
separados de maioria mantêm NorBERTo, SVM e RL. Inferência nativa posterior
em teste1 acertou 414/900 (46.00%); veja a
[integração dos ensembles](../../docs/results/ensemble-qwen.md).

Validação: **1,791/3,999 (44.79%)**, macro-F1 **44.44%**. Selecionou o melhor
checkpoint e não recebeu atualizações de gradiente. É conjunto historicamente
consultado, não teste final independente. O relatório de treino original
fica em `training.json`, sem tempos por passo.

A exportação original de validação associava saídas agrupadas a IDs sequenciais
da planilha. A ordem foi recuperada exatamente do estado aleatório e amostrador
do checkpoint final, reproduzindo a matriz registrada. Previsões alinhadas
ficam separadas em
`results/qwen-ep1-publication-20261105/validation-predictions-aligned.jsonl`.
A fonte histórica permanece inalterada. Avaliação futura usa amostrador
sequencial e confere a ordem antes de gravar; isso não muda os pesos do adaptador.

Veja o [guia de verificação e uso](../../docs/results/qwen-ep1-model.md).
