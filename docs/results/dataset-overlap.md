<a id="ep1-and-ep2-dataset-overlap"></a>

# Sobreposição dos dados do EP1 e EP2

[Índice dos resultados](README.md) · [Auditoria em formato estruturado](data/dataset-overlap.json)

<a id="assignment-and-dataset-identity"></a>

## Enunciado e identidade dos dados

O [enunciado do EP2](../reference/assignment.md) reutiliza explicitamente o
corpus de treino do EP1. Introduz teste2 (897 linhas) como entrega oficial e
fornece rótulos de teste1 (900 linhas) apenas para comparação estatística final
pareada com a entrega do EP1.

A auditoria original inspecionou os arquivos de treino abaixo, idênticos em
bytes, com o mesmo SHA-256, 20,092 linhas, textos, rótulos e ordem:

- `EP1/data/train.xlsx`
- `EP2/data/train_ep1.xlsx`
- `EP2/data/train_ep2.xlsx`

As duas planilhas não rotuladas de teste1 também são idênticas em bytes.
A rotulada contém exatamente os mesmos 900 textos na mesma ordem. As cópias
de treino do EP2 foram depois consolidadas em `EP2/data/train.xlsx`:
`train_ep1.xlsx` foi renomeada, e a cópia idêntica `train_ep2.xlsx`, removida.
A planilha canônica preserva bytes, rótulos e ordem. O JSON da auditoria mantém
os nomes originais como procedência histórica.

<a id="text-overlap-across-partitions"></a>

## Sobreposição textual entre partições

A comparação exata usa as respostas completas, sem alterações. A normalizada
segue o agrupamento EP1: decodificação de entidades HTML, redução de espaços,
remoção de espaços nas extremidades e conversão para minúsculas. Preserva
acentos e pontuação.

| Corpus de referência | Partição comparada | Linhas com correspondência exata | Linhas com correspondência normalizada |
|---|---|---:|---:|
| Treino, 20,092 linhas | Teste1, 900 linhas | 88 (9.78%) | 105 (11.67%) |
| Treino, 20,092 linhas | Teste2, 897 linhas | 93 (10.37%) | 107 (11.93%) |
| Teste1, 900 linhas | Teste2, 897 linhas | 37 (4.12%) | 47 (5.24%) |

As contagens referem-se às linhas da partição comparada, contadas uma vez,
independentemente do número de correspondências no corpus de referência.
Correspondências exatas estão incluídas nas normalizadas. Treino/teste1
compartilham 72 textos únicos exatos ou 83 normalizados; treino/teste2,
67 exatos ou 73 normalizados.

<a id="repeated-texts-and-label-disagreement"></a>

## Textos repetidos e divergência de rótulos

O treino contém 18,432 textos únicos exatos ou 18,143 normalizados.
Há 291 grupos de textos exatos com rótulos diferentes (1,682 linhas),
aumentando para 316 grupos normalizados (1,975 linhas).

Entre as 105 linhas de teste1 com correspondência normalizada no treino:

- 21 têm rótulo verdadeiro igual a todas as ocorrências de treino do texto.
- 72 têm ocorrências de treino com rótulos iguais e diferentes.
- 12 não têm ocorrência de treino com seu rótulo verdadeiro.

O próprio teste1 rotulado tem 10 grupos exatos com rótulos distintos (24 linhas),
ou 11 normalizados (31 linhas). Encontrar o texto não garante recuperar seu
rótulo. As planilhas expõem só texto e clareza; não estabelecem se respostas
repetidas têm a mesma solicitação, autor ou contexto de anotação.

<a id="interpretation-and-reproduction"></a>

## Interpretação e reprodução

Os arquivos fornecidos têm sobreposição textual entre treino e ambos os testes;
os textos não são totalmente disjuntos. As contagens descrevem os dados
independentemente dos classificadores. Resultados oficiais incluem todas as
linhas; a sobreposição e a ambiguidade de anotação devem ser informadas na
interpretação de generalização e significância pareada. O gabarito de Teste1
permanece reservado à comparação, sem inclusão no treino, recuperação ou desempate.

A auditoria em CPU não treina nem executa inferência, não modifica planilhas
e publica só agregados e hashes. Execute com openpyxl disponível, na raiz do
EP2, escolhendo novo arquivo de saída:

```bash
PYTHONPATH=src python3 scripts/diagnostics/audit-dataset-overlap.py \
  --ep1-root ../EP1 \
  --output results/dataset-overlap-new/summary.json
```

A auditoria histórica salva inspecionou sete planilhas e os 21 pares. O comando
atual inspeciona seis e os 15 pares, usando a planilha canônica do EP2.
A comparação é exata ou normalizada; não avaliou quase duplicatas ou similaridade
semântica. Testes sintéticos verificam contagem de ocorrências, normalização,
conflitos e identidade ordenada.
