<a id="final-svm-refit-calibration-and-frozen-soft-voting"></a>

# Retreinamento final do SVM, calibração e votação suave congelada

**Concluído:** desenvolvimento, congelamento e avaliação separada de Teste1
terminaram em `results/svm-finalization-20261106/`. Mantenha A original
(macro-F1 49.5844%, 447/900); B=C obtêm 49.5187%, 446/900. O
[relatório final](../results/svm-final-optimization-final.md) e o
[agregado](../results/data/svm-final-optimization.json) preservam o resultado
negativo. Os comandos abaixo documentam o protocolo executado e a retomada;
nenhum novo ajuste deve seguir a observação dos rótulos de Teste1.

As duas fases de validação cruzada com orçamento limitado estão concluídas e
verificadas independentemente. O modelo selecionado é **a escolha de um
erro-padrão da fase B**, não a maior média observada: palavras+char_wb binários,
C=.05, sem pesos de classe, palavras (1,2), min_df=5, max_df=.98, limite 60000;
char_wb (3,6), min_df=3, max_df=.95, limite 50000; ambos normalizados em L2,
perda hinge quadrática, normalização original e semente 42.

A finalização está no pacote independente `src/svm_finalization/`. Nenhum
pacote imutável de busca, configuração, resultado anterior, modelo original ou
ensemble padrão é alterado. As dependências fixadas de Python 3.12 existentes
são reutilizadas. Não são expostos GPU, rede, GPT, stacking, RAG ou ajuste/
inferência de outros modelos.

<a id="executed-development-command"></a>

## Comando de desenvolvimento executado

A preparação já foi concluída em `results/svm-finalization-20261106/`. Verifica
os 388 artefatos dos folds contra os dois manifestos independentes de auditoria,
fixa identidades de busca/fontes/ambiente/entradas e registra o candidato exato
e o protocolo posterior herdado. Lê apenas o relatório histórico de métricas
de Teste1 para fixar o checksum já conhecido do gabarito; não acessa as
planilhas finais nem novos rótulos finais.

Execute a etapa de desenvolvimento manualmente na raiz do EP2:

```bash
bash scripts/svm/finalize.sh --output-dir results/svm-finalization-20261106 develop
```

Este comando realiza:

1. **Um ajuste final do SVM** nas 16093 linhas originais e extração de escores
   nas 3999 reservadas inalteradas. Somente registros `Text` sem rótulo entram
   na inferência reservada. Salva o pipeline completo, hash da composição exata
   do treino e margens. O subprocesso nativo tem limite de 180 segundos; ajuste
   falho ou sem convergência não pode ser congelado. Artefatos de ajustes já
   concluídos são verificados e reutilizados na retomada.
2. **Calibração agrupada aninhada** nas 2027 linhas: cinco folds externos e
   três internos. Sigmoide é o padrão; isotônica exige melhora média pareada
   interna de perda logarítmica maior que seu erro-padrão. OOF externa de
   método fixo e OOF da política aninhada são reportadas separadamente,
   incluindo perda logarítmica, Brier multiclasse, ECE de confiança em dez
   intervalos, macro-F1 e confiabilidade. O método final é selecionado pela
   mesma regra de três folds em todas as linhas de calibração e depois
   ajustado nelas. Parâmetros numéricos portáveis são conferidos com sklearn;
   a calibração não ajusta classificadores-base.
3. **Seleção agrupada aninhada de pesos** nas 1972 linhas de meta-validação,
   com pesos SVM 0/.25/.5/.75/1 e probabilidades/temperatura NorBERTo inalteradas.
   Cada um dos cinco treinos externos escolhe um peso por três folds internos;
   a regra de um erro-padrão prefere proximidade a .75, depois F1 médio e menor
   peso. O peso final usa toda a meta-validação com a mesma regra. Escolhas
   externas, estabilidade, OOF da política e diagnósticos adaptativos do peso
   fixo permanecem distintos.
4. **Critérios conservadores de adoção no desenvolvimento**, herdados do plano:
   B só substitui A com limite inferior do bootstrap pareado por grupos de
   similaridade de macro-F1 acima de zero, diferenças positivas em ao menos
   quatro dos cinco folds e nenhuma perda de F1 por classe maior que .02.
   C precisa ainda superar B usando OOF externa da política de pesos e
   coincidir com o peso final em ao menos três folds externos. B deve primeiro
   passar contra A. O bootstrap usa 2000 sorteios pareados de grupos, semente
   42, e registra intervalos de confiança e contagens de correções/prejuízos.
   São critérios conservadores de desenvolvimento, não testes confirmatórios
   ou garantias de generalização de ponta a ponta.
5. **Congelamento de A/B/C e da recomendação principal** em `frozen.json` e
   manifesto com checksums, antes de abrir qualquer planilha de Teste1.
   O ensemble padrão ativo não é promovido, qualquer que seja a recomendação.

O mesmo comando retoma um ajuste concluído. Calibração e seleção são cálculos
determinísticos baratos e podem ser repetidos antes do congelamento. Arquivos
interrompidos ainda não confirmados são arquivados antes de nova tentativa.
Após congelar, `develop` verifica os artefatos e retorna sem reajustar nem
mudar a recomendação. Não edite fontes, ambiente, resultados ou plano fixados
durante a execução. Mudanças incompatíveis exigem novo diretório, sem reescrever
identidades anteriores.

Para um diretório inteiramente novo de finalização, a preparação é:

```bash
bash scripts/svm/finalize.sh --output-dir results/svm-finalization-new prepare
```

<a id="separate-test1-evaluation-after-freezing"></a>

## Avaliação separada de Teste1 após o congelamento

Somente após concluir e auditar o desenvolvimento, execute o comando separado:

```bash
bash scripts/svm/finalize.sh --output-dir results/svm-finalization-20261106 evaluate-test1
```

Ele verifica o manifesto congelado antes de acessar entradas finais, lê os
900 textos sem rótulos, reutiliza escores originais SVM/NorBERTo em cache e
gera somente novas margens SVM. A deve reproduzir exatamente cada ID histórico
de linha/texto, probabilidade e rótulo. Arquivos de A/B/C, dos cinco modelos
individuais e manifesto de previsões com checksums são persistidos antes de
ler os 900 rótulos do professor. O checksum do gabarito deve coincidir com a
planilha histórica fornecida. A deve reproduzir macro-F1 0.49584421663197825.
Nenhuma linha é filtrada, corrigida ou reordenada.

O resultado reporta acurácia, macro-F1, precisão/revocação/F1 por classe,
matrizes de confusão, Brier/perda logarítmica/ECE, contagens pareadas de
correções/prejuízos (também por classe), intervalos, McNemar exato descritivo
e complementaridade SVM/NorBERTo. O bootstrap reamostra grupos de textos
normalizados conjuntamente para todas as previsões. Trata duplicatas conhecidas,
mas não estabelece independência semântica. As comparações são retrospectivas
e descritivas após exposição histórica; Teste1 não pode mudar pesos, calibração,
modelo ou recomendação nem iniciar nova busca. Nenhuma entrada ou entrega de
Teste2 é lida ou gerada aqui.

`predict-test1` executa só a predição congelada, e `report` gera o relatório
salvo sem experimentos. O relatório inclui os 116 candidatos e escores por
fold, auditoria do SVM antigo, calibração, comparações de desenvolvimento,
análise de classes/complementaridade, comparações pareadas finais disponíveis,
limites metodológicos e recomendação de desenvolvimento inalterada.

<a id="evidence-and-validation"></a>

## Evidências e validação

Os artefatos incluem `plan.json`, `model.joblib`, `refit.json`,
`holdout-margins.npz`, `calibration.json`, `calibration-oof.npz`,
`development.json`, `meta-validation.npz`, `frozen.json`,
`frozen_manifest.json`, `report.md` e, somente após avaliar, `test1/`.

Testes sintéticos cobrem equivalência da calibração portável sigmoide/isotônica
(incluindo margens extremas reservadas), normalização isotônica com soma zero,
separação aninhada de grupos, regras padrão de método/peso, escolha externa
de pesos só no treino, contagens/intervalos pareados por grupo, rejeição da
adoção de modelos idênticos, proteções de ajuste/retomada/composição em
subprocesso e leitura tardia do gabarito após os oito arquivos de previsões
existirem. O caso sintético de 900 linhas reproduz deliberadamente a matriz
histórica sem abrir planilhas finais reais.

<a id="independent-read-only-verification"></a>

## Verificação independente somente de leitura

Auditorias salvas de desenvolvimento/final e reconstrução dos intervalos
passaram. Para verificar novamente as saídas existentes, use novos nomes:

```bash
PYTHONPATH=.cache/cascade-runtime:.cache/rag-runtime:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3.12 scripts/svm/verify-finalization.py --run results/svm-finalization-20261106 --stage development --output /tmp/svm-development-recheck.json
PYTHONPATH=.cache/cascade-runtime:.cache/rag-runtime:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3.12 scripts/svm/verify-finalization.py --run results/svm-finalization-20261106 --stage test1 --output /tmp/svm-test1-recheck.json
PYTHONPATH=.cache/cascade-runtime:.cache/rag-runtime:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3.12 scripts/svm/verify-confidence.py --run results/svm-finalization-20261106 --output /tmp/svm-confidence-recheck.json
```

Esses scripts não ajustam estimadores nem geram previsões. A verificação de
desenvolvimento não lê planilha final; as verificações final e de confiança
releem o gabarito de Teste1 já avaliado, com checksum histórico fixado.
Teste2 nunca é lido. O verificador de confiança reconstrói independentemente
20 intervalos de desenvolvimento e 72 finais, absolutos/pareados, usando
matrizes de confusão ponderadas por linha, 2000 sorteios compartilhados e
pareados de grupos e semente 42. Manifestos brutos são preservados com hashes
no agregado final versionado.
