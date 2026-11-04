# Análise do primeiro resultado GPT — 04/11/2026

## Resultado observado e decisão

O primeiro método GPT ainda não melhora o EP1. Não promover outra configuração
com base no teste1. A prioridade é aprender melhor o padrão de rotulagem de
`c1` no treino e reduzir o custo de experimentação. Os modos e configurações
originais permanecem como padrão; lotes e prompt curto são experimentais.

Fontes locais: `results/comparison-test1-codex/metrics.json`, previsões e
manifesto de `results/test1-codex-comparison/`, bases originais e a entrega
efetiva `../EP1/deliveries/test1.xlsx`. A ordem e os hashes dos textos foram
novamente conferidos. Foram comparadas as mesmas 900 respostas.

| Métrica | NorBERTo entregue | GPT/Codex |
|---|---:|---:|
| Acurácia | 48,00% (432/900) | 42,44% (382/900) |
| F1 macro | 47,86% | 40,39% |
| Recall c1 | 39,33% (118/300) | 18,67% (56/300) |
| Recall c234 | 51,67% (155/300) | 47,67% (143/300) |
| Recall c5 | 53,00% (159/300) | 61,00% (183/300) |

O GPT perdeu 62 acertos em `c1` e 12 em `c234`, ganhando 24 em `c5`: saldo
de **−50 acertos**, ou **−5,56 pontos percentuais**. Previu `c1` apenas 116
vezes, contra 394 `c234` e 390 `c5`. Isso identifica o gargalo; não autoriza
impor 300 previsões por classe ou calibrar limiares usando este gabarito.

McNemar: somente EP1 acertou 210 casos; somente GPT acertou 160. Estatística
com correção de continuidade **6,4892**, p exato bilateral **0,010755**.
A direção observada favorece o EP1. Pares com textos repetidos limitam a
hipótese de independência: a significância não dispensa essa ressalva.
O baseline majoritário, escolhido no treino (`c5`), faz 33,33% no teste1;
isso não informa o valor do baseline oficial de regressão logística.

## O que a auditoria local permite concluir

### Textos longos e seleção dos exemplos

O banco atual exige exemplos inteiros de até 1.600 caracteres. Dos exemplos de
treino, 1.200/6.347 `c1`, 1.366/6.853 `c234` e 886/6.892 `c5` ultrapassam esse
teto. Após deduplicação e exclusão de conflitos, ficam 4.298 exemplos `c1`,
4.847 `c234` e 5.452 `c5`. O prompt continua com um exemplo por classe, mas a
cobertura de tipos de resposta dentro de cada classe difere.

No teste1, as 134 respostas acima desse tamanho tiveram 32,84% de acurácia GPT,
contra 44,13% nas 766 mais curtas. O NorBERTo fez respectivamente 47,01% e
48,17%. Essa associação sugere investigar recuperação para textos longos no
treino. **Não comprova** que aumentar o teto ou a quantidade de exemplos melhore.
Os alvos não foram truncados; o limite refere-se ao banco de demonstrações.

Para não testar essa hipótese no teste1, foi feito um diagnóstico de recuperação
em **300 registros do treino**, separados pelo primeiro fold agrupado, seed 42,
com vocabulário/IDF ajustados apenas no banco restante. Os rótulos aqui são
obtidos por vizinho lexical, não pelo GPT:

| Limite do exemplo | Banco elegível | Vizinho mais próximo | Voto de 5 vizinhos |
|---|---:|---:|---:|
| 1.600 caracteres | 11.661 | 42,67% | 42,67% |
| 3.200 caracteres | 13.690 | 41,67% | 43,00% |
| 14.000 caracteres | 14.242 | 42,00% | 42,67% |

Não há sinal convincente para simplesmente enviar exemplos maiores. Esses
números não estimam a acurácia que o GPT teria com os mesmos exemplos.

### Informação ausente e exemplos de erros

Na linha Excel 29, o texto contém somente “Prezado Senhor,”: gabarito `c1`,
previsão `c5`. No treino, esse texto normalizado tem cinco `c1` e quatro `c234`.
Na linha 693, “Sua resposta segue em anexo.” foi `c1` no gabarito e `c5` na
predição; o treino contém esse texto com `c1` e `c5`. O anexo não está disponível.
Esses casos permitem discutir contexto ausente e anotações subjetivas, mas
não concluir que todo erro resulte de ruído ou que toda menção a anexo seja `c1`.

Os 11 grupos contraditórios do próprio teste1 implicam pelo menos 15 erros para
qualquer regra determinística que use só o texto normalizado. Esse limite é
muito menor que os 518 erros GPT e não explica sozinho o desempenho.
Nas 105 respostas cujo texto aparece no treino, o GPT fez 36,19%, contra
43,27% nas 795 restantes. Nada disso foi usado para criar regras por registro.

### Interpretação lexical auxiliar

Foi ajustada uma regressão logística auxiliar em todo o treino, sem chamadas
remotas. Seus pesos mais positivos incluem `solicitação`, `demanda` e `pedido`
para `c1`; `sistemas`, `concurso` e `reitoria` para `c234`; `dnpm`, `questionário`
e `encaminhamos` para `c5`. Também aparecem nomes próprios. Isso evidencia
associações lexicais do corpus, não características universais de clareza nem
explicações internas do GPT. Uma recuperação só por assunto pode selecionar
textos similares sem transmitir bem o critério subjetivo de avaliação.
Pesos e contribuições locais estão em `results/comparison-test1-codex/interpretation-local.json`.

## Custo real e piloto limitado pelo usuário

O histórico da execução completa registra **5.642.206 tokens de entrada**,
dos quais 3.087.744 informados como cache remoto, e 29.040 de saída, em 879
tentativas de execução. É consumo informado pelo CLI, não custo monetário;
uma falha de rede/quota pode não retornar usage. A arquitetura anterior abre
uma sessão nova por texto, repetindo o contexto do CLI a cada chamada.

Foi implementado um modo opcional em pequenos lotes: exemplos repetidos dentro
do lote são enviados uma única vez; cada alvo tem um ID opaco e referências
às suas demonstrações. O JSON de resposta exige um rótulo por ID. O parser
rejeita duplicatas, omissões, IDs desconhecidos e classes inválidas, e restaura
a ordem original. Há cache por lote e retomada sem refazer lotes completos.

O usuário autorizou poucas novas chamadas. O piloto consumiu **exatamente dez
processos Codex**, com seis registros do treino, dois por classe, excluindo do
banco o fold inteiro de validação. Não leu gabarito de teste1 ou teste2 para
construir os requests. Não foram feitas outras chamadas após o teto.

| Variante | Processos | Entrada | Saída | Tempo total | Acertos |
|---|---:|---:|---:|---:|---:|
| Prompt original, individual | 6 | 39.088 | 184 | 29,29 s | 1/6 |
| Mesmo prompt, lotes de 3 | 2 | 16.733 | 210 | 17,32 s | 0/6 |
| Prompt empírico curto, lotes de 3 | 2 | 16.407 | 167 | 14,29 s | 0/6 |

Os lotes de três reduziram entrada em **57,19%** e tempo em **40,87%** nessa
medição, com 5/6 classificações iguais às individuais. O prompt curto concordou
em 4/6 casos. O resultado total foi 72.228 tokens de entrada e 561 de saída.
A entrada individual teve 4.224 tokens em cache remoto; as variantes em lote,
zero, portanto percentuais de tokens não são percentuais de preço ou limites
financeiros. O tempo inclui preparação local e varia com servidor e carga.

**Seis casos não permitem escolher um modelo ou estabelecer equivalência de
qualidade.** O piloto não mostrou melhora de acurácia. Nenhum perfil novo foi
promovido; não é correto transformar 0/6 em uma estimativa de desempenho geral,
nem assumir que os lotes preservem todas as decisões do modo individual.

Resultados: `results/pilot-efficiency-20261104/report.json`; reprodução em
`scripts/pilot_efficiency.py`, com dry-run como padrão e limite de dez processos.
O comando com `--execute` pode consumir o plano; não foi repetido nesta análise.

## Próximas melhorias, por prioridade

1. **Investigar uma cascata supervisionada + GPT**, com a decisão de encaminhar
   casos e a combinação escolhidas somente no desenvolvimento do treino.
   O NorBERTo já é uma referência mais forte e o GPT tem um perfil de erros
   diferente. Isso sugere complementaridade, não prova vantagem de um ensemble.
   Reservar o GPT a uma fração dos casos pode poupar mais que encurtar prompts.
2. **Rever a recuperação das demonstrações**, comparando zero-shot, exemplos
   representativos e vizinhos que considerem também estilo/comprimento ou uma
   representação supervisionada. Não atribuir a seleção à classe verdadeira
   do alvo. O diagnóstico lexical acima não justifica mais contexto por si só.
3. **Validar lotes de três em uma amostra maior de treino**, antes de usá-los
   em produção. Fazer comparação pareada, medir acurácia/F1/recall `c1`, tokens
   e tempo. Os resultados atuais justificam experimentar eficiência, não mudar
   o classificador escolhido. Aumentar reasoning não é a primeira tentativa,
   pois eleva custo sem ganho demonstrado.
4. **Congelar o procedimento antes do teste2**. O teste1 já foi inspecionado;
   o desenvolvimento seguinte deve usar treino e registrar essa exposição.
   Não selecionar limiares/contagens de classe por este resultado nem prometer
   que uma nova rodada no mesmo teste1 seja uma confirmação independente.

O primeiro passo concreto recomendado é um estudo local de uma cascata,
com uma pequena amostra do treino para medir quando o GPT acrescenta acertos.
Não recomendo gastar outra rodada de 900 chamadas no método atual.

## Referências oficiais e mudanças de código

A [documentação de prompting](https://developers.openai.com/api/docs/guides/prompt-engineering)
recomenda demonstrações diversas e avaliação das alterações. As
[práticas para reasoning](https://developers.openai.com/api/docs/guides/reasoning-best-practices)
também orientam testar zero-shot e manter exemplos coerentes com a tarefa;
não estabelecem que mais exemplos ou reasoning melhorem este corpus.
O transporte continua seguindo a [execução não interativa do Codex](https://learn.chatgpt.com/docs/non-interactive-mode).

Foram adicionados `batch_size` (padrão 1) e `prompt_profile` (padrão `original`)
à configuração, flags equivalentes na CLI, parser/cache para lotes, testes e
o script de piloto. `configs/codex-economico-experimental.json` contém a variante
curta em lotes de três e **não é uma configuração selecionada**. O uso de lotes
na API não foi implementado e é rejeitado explicitamente. Dados e resultados
originais foram preservados. O novo código muda os fingerprints: novos
experimentos exigem diretórios novos; os caches antigos não devem ser apagados.
