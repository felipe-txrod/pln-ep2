<a id="development-only-logistic-regression"></a>

# Regressão logística somente de desenvolvimento

Este é o pacote RL selecionado para EP2. Treinou nas 16,093 linhas fixas de
desenvolvimento, excluindo 3,999 reservadas. Usa o pipeline TF-IDF original EP1
de palavras/caracteres e C=0.25 fixo, L2/L-BFGS, semente 42 e duas threads CPU,
sem busca de parâmetros.

Convergiu em 38 iterações e 25.08 segundos. O artefato exportado foi recarregado
e validado. Verificação independente posterior conferiu os 16,093 IDs de
treino, hashes de texto, rótulos, hashes de fontes/divisão e nenhuma sobreposição
entre 12,956 grupos de treino e 3,281 reservados. Carregamento padrão e
inferência RL em três textos sintéticos passaram.

SHA-256 do artefato:

```text
05d69e147a10b7c49ebe3893f247e0f23e8d61005aaf4fd7b7eb45845d49a20f
```

`manifest.json` registra checksums de inferência; `frozen_manifest.json` define
o contrato de identidade do carregador/ensemble. Metadados e identidades das
linhas preservam procedência. Avaliação separada mediu 1,803/3,999 no conjunto
reservado (45.09% acurácia, 44.78% macro-F1) e 447/900 em teste1 (49.67%, 49.32%).
As 4,899 linhas e métricas foram verificadas independentemente, sem mudar
hashes. Teste2 não foi avaliado. Resultados antigos da RL completa não se
aplicam a este modelo.

O ensemble seleciona este artefato quando `rl` é membro ou árbitro local.
Sua composição padrão existente continua Qwen + NorBERTo. As previsões ainda
precisam ser geradas para a planilha desejada; o treino não gera previsões
de teste ou arquivos de entrega.

Veja o [fluxo RL](../../src/rl_classifier/README.md) e o
[registro da verificação](../../docs/results/data/rl-development-verification.json).
