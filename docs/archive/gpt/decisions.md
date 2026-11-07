# Requisitos e decisões do experimento GPT

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

Leitura prévia: `ep2-enunciado.md` e `ep2-modelo-relatorio-com-analise.md`
(integralmente), enunciado transcrito do EP1, README, decisão do NorBERTo,
auditoria da entrega e implementação de leitura, normalização, treino e inferência.

## Contrato oficial

O enunciado tem prioridade sobre o pedido de usar teste1 como validação.
**Teste1 é exclusivamente comparação estatística final**, não seleção de prompt,
modelo, exemplos ou hiperparâmetros. A seleção usa uma divisão agrupada do treino.
Primeiro se congela o método e se prediz teste2, depois teste1. O comando de
avaliação oficial exige os 900 registros do teste1, alinhados à entrega original.
Não comparar modelos em teste2 nem entregar teste1 no lugar de teste2.

A entrega acadêmica requer `grupoXX.zip` contendo relatório com código/reprodução,
apresentação PDF de até 10 minutos e teste2.xlsx com **897** rótulos na posição
original, mesmas linhas e colunas. Prazo indicado: 08/11; apresentações 10/11 e
17/11. Pesos: acurácia 70%, inovação 10%, análise de características 5%, melhora
significativa (p<0,05) 5%, relatório 5%, apresentação 5%. Resultado inferior à
classe majoritária ou entrega/comparação do conjunto errado implica nota 3.
O enunciado prevê descontos cumulativos de 20% para resultado inferior ao
baseline oficial, falta de relatório/código ou PDF, e formato incorreto.

O relatório deve identificar integrantes e números USP, representação e modelo,
pré-processamento, grade e parâmetros escolhidos, partições e reprodução,
acurácia, comparação EP1/EP2 e estatística/p-valor McNemar, interpretação global
e local de um modelo simples e análise manual de erros, além do repositório.
O código prepara resultados para esses itens. Relatório acadêmico e apresentação
precisam dos resultados reais e da identificação do grupo; não são inventados.

## Problema e evidência prévia

Predição da nota subjetiva de clareza de respostas públicas e-SIC usando somente
`resp_text`: `c1` = nota 1 (baixa), `c234` = notas 2–4 (intermediárias), `c5` =
nota 5 (alta). O `c6` no slide antigo é inconsistente com os dados e com o EP1;
mantêm-se as três etiquetas efetivamente presentes. Pergunta original, anexos,
avaliador e nota individual 2/3/4 não estão disponíveis. Não inferir esses dados
nem equiparar automaticamente negativa, extensão ou linguagem formal a nota.

O NorBERTo vigente usa a revisão fixa `db73446f89c96044863ea05a39f680524b84bccb`,
LoRA rank 4/alpha 8/dropout 0,05, contexto 512, início/fim e entrada preservando
maiúsculas. Treinou duas épocas em 16.093 registros, excluindo 3.999 do holdout.
CV agrupada: 43,47% de acurácia; holdout: 45,31%. O holdout já havia sido
consultado em pesquisa. **Essas métricas não são o resultado de teste1.**
As 900 previsões realmente entregues em `../EP1/deliveries/test1.xlsx` acertam
432/900 = **48,0%** contra o gabarito novo, com textos e ordem conferidos.

## Inspeção das bases

| Base | Linhas | Distribuição |
|---|---:|---|
| train.xlsx | 20.092 | c1=6.347; c234=6.853; c5=6.892 |
| teste1 rotulado | 900 | 300 por classe |
| teste2 | 897 | clarity vazia |

Todas possuem `resp_text` e `clarity`. Há uma célula numérica no texto de treino:
é convertida explicitamente em string e registrada, preservando sua linha.
Após HTML-unescape, espaços e lowercase para agrupamento: 18.143 textos únicos,
316 grupos com rótulos conflitantes. Há 105 respostas de teste1 e 107 de teste2
com texto normalizado presente no treino. A seleção exclui exemplos exatamente
iguais ao alvo por precaução, mesmo que a repetição já exista nas bases oficiais.
Não excluir linhas da avaliação nem corrigir seu gabarito.

## Método proposto e limites

Sem fine-tuning. Comparar zero-shot, três exemplos representativos (um/classe),
três semelhantes e seis semelhantes (dois/classe), sempre com teto de tamanho.
TF-IDF serve **apenas à recuperação**; o GPT decide o rótulo. A alternativa
representativa usa centroide por classe com penalização de redundância. A seleção
semelhante aplica o mesmo princípio de diversidade à similaridade com o alvo.
Grupos conflitantes e vazios são excluídos **somente do banco de demonstrações**;
a validação conserva seus registros. Não presumir que mais exemplos melhora.

Escolher por acurácia na validação interna, com F1 macro e consumo para diagnóstico;
empate favorece a primeira configuração da grade, ordenada da mais simples à mais
complexa. A divisão usa StratifiedGroupKFold (5 folds, seed 42), normalização de
duplicatas exatas, sem ajustar vocabulário/IDF nos alvos. Essa divisão não isola
paráfrases/duplicatas aproximadas e a pequena amostra de desenvolvimento aumenta
incerteza. O resultado selecionado é exploratório; teste1 permanece fechado à busca.

Na API, snapshot inicial verificado: `gpt-5.4-mini-2026-03-17`, Responses,
saídas estruturadas, raciocínio `none`. Não é uma afirmação de melhor modelo
para português. O modelo é configurável; não há troca automática nem descarte
de parâmetros em caso de erro. Não se envia semente de API não documentada.
Temperatura é omitida por padrão para não presumir suporte; snapshot, ordem,
prompt e cache são fixados. Não se promete determinismo absoluto de API remota.

## Documentação oficial consultada em 03/11/2026

- [Responses / criação](https://developers.openai.com/api/reference/python/resources/responses/methods/create): `input`, `text.format`, `store`, limites e consumo.
- [Saídas estruturadas](https://developers.openai.com/api/docs/guides/structured-outputs): esquema estrito, enumeração, recusas e respostas incompletas.
- [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini): snapshot, Responses, saídas estruturadas e raciocínio none/low/medium/high/xhigh.
- [SDK oficial](https://developers.openai.com/api/docs/libraries): biblioteca Python e autenticação por variável de ambiente.
- [Erros](https://developers.openai.com/api/docs/guides/error-codes): limites, tempo esgotado, falhas transitórias e Retry-After.

Uma chamada independente por texto evita falhas de alinhamento entre respostas em
lote. Cache SQLite por payload completo evita repetir textos idênticos e chamadas
concluídas após interrupções. JSON é validado também localmente; nenhuma falha vira
rótulo majoritário. Retentativas são limitadas e só para falhas transitórias.
Os rótulos do alvo não integram sequer o objeto aceito pelo construtor de requests.

## Backend Codex CLI

Além da API, há o backend `codex`, selecionável na CLI ou em configuração.
Usa `codex exec` com autenticação do próprio CLI, inclusive login ChatGPT, sem
exigir `OPENAI_API_KEY` nem importar o SDK da API. A instalação Python mínima
agora deixa o SDK como extra `[api]`; o lock completo ainda o inclui para testes
e para a alternativa de API.

As duas interfaces não são presumidas equivalentes: o Codex tem instruções e
contexto próprios, limites do plano e modelos disponíveis distintos. Conforme
a [documentação de modelos](https://learn.chatgpt.com/docs/models), GPT-5.4 mini
foi retirado do Codex com login ChatGPT em 31/08/2026. A configuração Codex
inicial usa `gpt-6.1-sol`, explicitamente, sem mudar a configuração de API.
O alias pode evoluir no serviço e não oferece a mesma fixação de snapshot da API.

O CLI foi verificado localmente na versão 0.165.0. O adaptador exige essa versão
ou superior para os controles utilizados. Cada texto abre uma sessão efêmera
nova, fora do projeto, sem instruções AGENTS.md, configuração pessoal, plugins,
apps, shell ou outras ferramentas de acesso a dados. O prompt vai pelo stdin;
schema e instruções são os únicos arquivos de entrada no diretório temporário.
O JSON final é validado localmente pelo mesmo parser da API. A aceitação exige
um único evento `turn.completed`, tokens válidos e ausência de eventos de
ferramentas/falhas. O adaptador não faz retry automático de processos.

Cache e manifests registram backend, política, versão/hash do executável e hash
do adaptador, além dos dados já descritos. Teste1 exige o mesmo runtime de teste2.
`max_output_tokens` e `temperature` ficam `null` no Codex, pois não há equivalente
verificado nesse fluxo; valores numéricos são rejeitados. O orçamento local
limita execuções, não as possíveis retentativas HTTP internas do Codex. Houve
uma chamada real de um texto **sintético** com login ChatGPT e resposta JSON
válida; isso verifica integração, não acurácia nas bases oficiais.

Fontes adicionais: [execução não interativa](https://learn.chatgpt.com/docs/non-interactive-mode),
[referência do CLI](https://learn.chatgpt.com/docs/developer-commands?surface=cli),
[configurações](https://learn.chatgpt.com/docs/config-file/config-reference),
[autenticação](https://learn.chatgpt.com/docs/auth). A tentativa de sobrescrever
`model_providers.openai.*` foi recusada pelo CLI instalado por ser um provedor
reservado; tais overrides não são utilizados. Não se afirma ter desativado
retentativas internas do Codex.

## Comparação isolada de teste1

Após a inclusão de `data/test1.xlsx`, foi acrescentado o modo `comparison` para
classificar suas 900 respostas sem exigir uma execução anterior de teste2.
`export --purpose comparison` produz uma cópia XLSX separada, com validação de
linha e texto, sem aceitar o gabarito como entrada de exportação. O modo
`statistical` conserva a exigência de vínculo à execução oficial de teste2.
Ambos permitem avaliação local contra o gabarito; os rótulos verdadeiros nunca
entram no prompt. A cópia de teste1 não é a entrega do EP2 e a avaliação isolada
não deve orientar seleção de exemplos, prompt ou hiperparâmetros.
