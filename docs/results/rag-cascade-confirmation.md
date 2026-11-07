<a id="rag-context-confirmation-stacking-without-gpt-selected"></a>

# Confirmação de contexto RAG: stacking sem GPT selecionado

[Agregado e verificação independente](data/rag-cascade-confirmation.json) · [Relatório do piloto](rag-cascade-pilot.md) · [Protocolo](../guides/rag-stacking.md)

A confirmação iniciada manualmente concluiu as quatro variantes de contexto,
com **94 chamadas válidas, nenhuma tentativa falha e nenhum fallback**.
As 1,884 linhas OOF foram mantidas. O melhor resultado foi **stacking original
com roteamento zero ao GPT**: 910 acertos, acurácia **48.30%**, macro-F1
**48.08%**. Todas as configurações testadas com roteamento positivo perderam
acurácia e macro-F1 frente à mesma referência.

O melhor macro-F1 com GPT foi previsões locais sem RAG a 5%: 906 acertos,
acurácia 48.09%, macro-F1 47.84%. RAG híbrido com três exemplos e previsões
locais obteve 47.83% a 5% e 47.82% a 10%. O ganho exploratório do piloto não
persistiu. A avaliação não realizou nova execução remota, mudança de padrão,
congelamento de pacote ou avaliação de teste final.

<a id="complete-paired-comparison"></a>

## Comparação pareada completa

As métricas incluem todas as linhas da confirmação, não só as roteadas.
Cinco por cento encaminham 94 alvos; dez por cento, 188. Ambos reutilizam
as mesmas respostas salvas, sem execuções separadas de API. Correções e
prejuízos comparam com stacking nas mesmas linhas.

| Contexto | Roteamento GPT | Acertos | Acurácia | Macro-F1 | Corrigidos | Prejudicados |
|---|---:|---:|---:|---:|---:|---:|
| Stacking | 0% | 910/1884 | 48.30% | 48.08% | 0 | 0 |
| text | 5% | 907/1884 | 48.14% | 47.80% | 23 | 26 |
| text | 10% | 891/1884 | 47.29% | 46.83% | 36 | 55 |
| local | 5% | 906/1884 | 48.09% | 47.84% | 12 | 16 |
| local | 10% | 901/1884 | 47.82% | 47.52% | 23 | 32 |
| hybrid-3-rag | 5% | 901/1884 | 47.82% | 47.55% | 17 | 26 |
| hybrid-3-rag | 10% | 899/1884 | 47.72% | 47.37% | 36 | 47 |
| hybrid-3-local | 5% | 906/1884 | 48.09% | 47.83% | 17 | 21 |
| hybrid-3-local | 10% | 907/1884 | 48.14% | 47.82% | 29 | 32 |

A seleção armazenada tem `budget: 0.0` e nome de receita inativa `text`,
escolhido pelo desempate predefinido. Isso significa só stacking; não significa
vitória de GPT somente com texto. A cascata existente sem GPT já tem esse
comportamento e foi preservada.

<a id="per-class-f1-and-observed-error-pattern"></a>

## F1 por classe e padrão observado de erros

| Configuração | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| Stacking | 44.36% | 45.43% | 54.45% |
| Local, 5% | 44.44% | 44.59% | 54.49% |
| RAG híbrido + local, 5% | 44.22% | 44.41% | 54.87% |
| RAG híbrido + local, 10% | 44.24% | 44.14% | 55.08% |

Com RAG híbrido e previsões locais a 10%, os acertos c5 aumentaram de 361 para
377 e c1 de 238 para 240. Os de c234 caíram de 311 para 290. O saldo foi três
acertos a menos: 29 erros corrigidos contra 32 acertos prejudicados. A
frequência predita c5 subiu de 699 para 742, enquanto c234 caiu de 699 para
644. Observa-se deslocamento para rótulos de alta clareza; o experimento não
estabelece sua causa.

As matrizes usam linhas verdadeiras e colunas preditas em ordem `c1`, `c234`, `c5`.

Stacking:

```text
238 211 138
159 311 200
 89 177 361
```

RAG híbrido + local, 10%:

```text
240 196 151
166 290 214
 92 158 377
```

<a id="actual-resources"></a>

## Recursos reais

| Contexto | Chamadas válidas | USD estimados | Segundos totais do braço |
|---|---:|---:|---:|
| text | 23 | 0.341154 | 94.87 |
| local | 24 | 0.363062 | 100.76 |
| hybrid-3-rag | 23 | 0.459068 | 94.38 |
| hybrid-3-local | 24 | 0.496268 | 101.33 |

- Total da confirmação: **94 chamadas válidas**, zero falhas, dentro do limite
  de 128 tentativas. Pedidos idênticos foram deduplicados sem remover linhas.
- Tokens conhecidos: **957,003 de entrada**, incluindo **236,544 em cache**,
  e **19,498 de saída**.
- Estimativa configurada equivalente à API padrão: **USD 1.6595524**.
  Não é fatura da assinatura ChatGPT.
- Soma dos tempos dos braços: **391.34 segundos** (6 minutos e 31 segundos),
  sem preparação local. Latência média: **8.09 segundos**; soma das latências:
  **760.71 segundos**, com dois processos simultâneos.
- Piloto e confirmação juntos: **178 chamadas válidas e três tentativas falhas
  arquivadas do piloto**, 181 tentativas registradas. Estimativa com consumo
  conhecido: **USD 3.6825404**. As falhas mantêm consumo desconhecido excluído
  dessa estimativa.
- Avaliação e verificação independente fizeram **zero novas chamadas GPT**.

Modelo/ambiente, textos originais, exemplos RAG restritos ao treino e prompts
permaneceram congelados. Codex 0.166.0 foi selecionado pelo executável
versionado imutável, evitando a interrupção por atualização do piloto.
Cache, históricos de tentativas e experimentos anteriores foram mantidos.

<a id="isolation-verification-and-next-state-boundaries"></a>

## Isolamento, verificação e limites das próximas etapas

Os 48 grupos de similaridade do piloto ficam fora da confirmação; outras
40 linhas de meta-validação desses grupos também são excluídas, restando
1,884 das 1,972 OOF originais. A recuperação usa só o banco original de
16,093 linhas, excluindo 3,999 reservadas. Os quatro contextos são texto,
previsões locais, RAG híbrido com três exemplos e ambas as evidências.
O roteamento depende apenas das previsões OOF salvas e da incerteza.

Cada contexto tinha arquivo de 1,884 previsões persistido e verificado por
checksum antes da abertura dos rótulos. Um verificador independente da biblioteca
padrão reconstruiu roteamento 0/5/10%, os doze conjuntos de métricas/matrizes,
correções/prejuízos, arquivos finais de 10% e totais de recursos diretamente
dos registros de tentativas. Todas as verificações passaram.

Selecionar zero segue o critério predefinido de macro-F1. Pequenas perdas não
provam inferioridade universal do GPT. Piloto e confirmação têm composição
e cobertura de roteamento diferentes; seus níveis absolutos não devem ser
comparados como curva de aprendizado. Desenvolvimento anterior do roteador
na validação OOF e exposição do checkpoint Qwen continuam limitações herdadas.

**Teste1 e teste2 não foram abertos nesta etapa.** Teste2 não tem gabarito aqui
e só recebe planilha de predições para entrega; acurácia/F1 finais exigem
Teste1 rotulado após congelar. Nenhum comando final de predição ou congelamento
foi executado nesta verificação. A confirmação concluída não precisa ser repetida.

Evidência local:
`results/rag-context-pilot-20261106/confirmation/{plan,run,selection,independent-summary}.json`
e cada diretório de contexto com solicitações, recuperação, decisões,
previsões e tentativas. Agregados públicos não contêm textos originais nem
rótulos individuais verdadeiros.
