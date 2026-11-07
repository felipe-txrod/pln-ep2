<a id="stacking-with-selective-gpt-decisions"></a>

# Stacking com decisões seletivas do GPT

O pacote congela regra OOF de roteamento, modo do prompt GPT, modelo, esforço
de raciocínio, orçamentos, procedência e identidades de fontes/ambiente.
Referencia `models/ensemble-stacking-qwen/` inalterado. Pesos GPT são remotos;
respostas validadas ficam no cache SQLite persistente ignorado pelo Git.

**A validação escolheu orçamento GPT zero.** Macro-F1 OOF agrupado aninhado
foi 47.2977% no stacking, contra 47.2762%, 46.8407% e 47.1016% para 5%, 10%
e 20%. Previsões locais superaram texto isolado na ablação inicial de 10%,
mas ficaram abaixo do stacking. O pacote não presume benefício do GPT.

As configurações positivas congeladas foram avaliadas nas 900 linhas de teste1:

| Orçamento GPT | Acertos | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| 0% | 440 | 48.8889% | 48.8034% |
| 5% | 435 | 48.3333% | 48.1824% |
| 10% | 434 | 48.2222% | 48.0477% |
| 20% | 436 | 48.4444% | 48.1761% |

`cascade_classifier predict` em produção usa orçamento zero, mantendo stacking
sem chamadas GPT. `--compare-budgets` explícito reproduz a comparação congelada
0/5/10/20%, com união máxima de 20% dos alvos. Toda seleção precede o gabarito
final; nenhum resultado final mudou o pacote.

O classificador remoto solicita `gpt-6.1-sol`, raciocínio `low`, por sessões
Codex isoladas e saída estruturada estrita. A CLI não expõe temperatura.
Cache por item, retentativas limitadas, disjuntor e fallback explícito de
stacking estão no pacote separado `cascade_classifier`.

`manifest.json` fixa `model.json`, `validation.json`, `prompt-ablation.json`
e `preparation.json`. Este registra fontes/partição originais e procedência
OOF aninhada 5x5: 16,093 linhas de treino-base, 2,027 de calibração e 1,972 de
meta-validação/roteamento. O checkpoint Qwen já consultou o conjunto reservado
completo; a exposição herdada continua declarada.

Veja [comandos](../../docs/guides/stacking-gpt-cascade.md),
[experimento medido](../../docs/results/stacking-gpt-cascade.md) e
[agregado verificado](../../docs/results/data/stacking-gpt-cascade.json).
