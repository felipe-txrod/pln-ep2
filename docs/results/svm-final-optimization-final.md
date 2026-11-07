<a id="final-svm-optimization-and-soft-voting-investigation"></a>

# Investigação da otimização final do SVM e da votação suave

Investigação concluída em 2026-11-06. **Manter A: ensemble original SVM 75% /
NorBERTo 25% inalterado.** A recomendação foi congelada no desenvolvimento
antes da avaliação final de Teste1. O modelo otimizado não passou nos critérios
predefinidos de adoção; os rótulos finais não motivaram novos ajustes ou mudanças
do padrão.

| Configuração congelada | Acertos Teste1 / 900 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| A: SVM original, pesos originais 75/25 | 447 | 49.6667% | 49.5844% |
| B: SVM otimizado, pesos originais 75/25 | 446 | 49.5556% | 49.5187% |
| C: SVM otimizado, pesos selecionados no desenvolvimento | 446 | 49.5556% | 49.5187% |

A seleção manteve 75/25 nos cinco folds externos e na escolha final; B e C
são idênticos. B corrige 21 erros de A e prejudica 22 acertos. A diferença
pareada de macro-F1 é -0.0657 pontos percentuais, com intervalo de 95% por
bootstrap de grupos de textos normalizados [-1.5466, +1.3377] pp. Isso não
estabelece diferença significativa.

O [agregado versionado](data/svm-final-optimization.json) preserva métricas e
identidades de verificação. Artefatos locais completos, oito conjuntos de
previsões e relatório gerado ficam em `results/svm-finalization-20261106/`.
As seções abaixo mantêm as 116 configurações e seus resultados por fold,
além do registro separado da busca original interrompida de 129 folds.

<a id="1-original-svm-audit"></a>

## 1. Auditoria do SVM original

O artefato original permanece inalterado. Decodificação de entidades HTML,
redução de espaços e minúsculas preservam acentos/pontuação. TF-IDF de palavras
e char_wb são concatenados por FeatureUnion. O modelo original não tem
predict_proba; seu calibrador sigmoide antigo e a temperatura antiga do
NorBERTo ficam congelados na configuração A.

O contrato original completo e congelado de inferência é:

```json
{
  "artifact_sha256": "9a30abb664d5023d70f444641c4095cf800acba0c98fd8658087e7adad78ee12",
  "normalization": "html_unescape_whitespace_lowercase",
  "word": {
    "ngram_range": [
      1,
      2
    ],
    "min_df": 2,
    "max_df": 0.98,
    "sublinear_tf": true,
    "max_features": 60000,
    "analyzer": "word",
    "norm": "l2",
    "use_idf": true,
    "smooth_idf": true,
    "binary": false,
    "strip_accents": null,
    "token_pattern": "(?u)\\b\\w\\w+\\b",
    "lowercase": true
  },
  "char": {
    "ngram_range": [
      3,
      5
    ],
    "min_df": 3,
    "max_df": 1.0,
    "sublinear_tf": true,
    "max_features": 50000,
    "analyzer": "char_wb",
    "norm": "l2",
    "use_idf": true,
    "smooth_idf": true,
    "binary": false,
    "strip_accents": null,
    "lowercase": true
  },
  "classifier_parameters": {
    "C": 0.05,
    "dual": "auto",
    "max_iter": 10000,
    "tol": 0.0001,
    "random_state": 42,
    "penalty": "l2",
    "loss": "squared_hinge",
    "class_weight": null,
    "fit_intercept": true,
    "intercept_scaling": 1
  },
  "runtime_versions": {
    "numpy": "2.3.3",
    "scipy": "1.16.2",
    "scikit-learn": "1.7.2",
    "joblib": "1.5.2"
  }
}
```

A preparação reproduziu as 3999 linhas de margens originais com diferença
absoluta máxima 0.0 e reproduziu exatamente a matriz e macro-F1 de meta-validação
do ensemble antigo 75/25. A busca interna original otimizava acurácia;
este estudo otimiza macro-F1.

<a id="2-data-provenance-and-prospective-separation"></a>

## 2. Dados, procedência e separação prospectiva

| Papel | Linhas | Grupos de similaridade |
|---|---:|---:|
| Treinamento | 16093 | 12956 |
| Calibração | 2027 | 1627 |
| Seleção dos pesos | 1972 | 1654 |

Os rótulos vêm apenas de clarity na planilha original train.xlsx. SHA-256,
bytes da divisão EP1, IDs, hashes de texto, rótulos, manifesto de grupos e
auditoria histórica aprovada são verificados antes de cada etapa. Componentes
de similaridade foram formados sem rótulos com mascaramento de identificadores
e cosseno TF-IDF de caracteres 3–5 a .93. Nenhum componente ou texto normalizado
repetido cruza papéis. TF-IDF aprende só em cada fold de treino; o ajuste
final usa apenas as linhas originais de treinamento.

As 3999 observações são desenvolvimento reutilizado historicamente. Teste1
já foi examinado e tem sobreposição textual conhecida; sua avaliação posterior
é retrospectiva. Teste2 permanece totalmente fora da investigação. Não se
executam GPT, stacking, RAG, ajuste/inferência de outros modelos ou promoção
do padrão.

<a id="3-search-space-compute-amendment-and-selection"></a>

## 3. Espaço de busca, revisão computacional e seleção

A busca original de 256 candidatos parou após 129 folds, pois ajustes sem
normalização levavam 643–991 segundos e frequentemente não convergiam.
Todos os arquivos/cache/fontes originais são preservados. O usuário autorizou
revisão do orçamento: 96 candidatos na fase A, quatro famílias, desenhos
normalizados L1/L2, C=.005/.05/.5, ambos os pesos de classe, TF binário/sublinear
e variações de ngramas/df. Setenta e dois folds antigos exatamente compatíveis
foram verificados e importados. As exclusões tratam custo, não comprovam
inferioridade preditiva.

O motor reduzido usa quatro processos, uma thread numérica por ajuste e
limite de 120 segundos para cada filho. Ajusta cada representação uma vez
por fold de treino e transforma a validação uma vez, compartilhando matrizes
esparsas somente de leitura. Tempo esgotado/não convergência exclui o candidato;
nenhum rótulo fabricado substitui a falha. A fase B refina duas representações
predefinidas ordenadas pela média, com C local, perda e pesos de ramos.
Configurações, matrizes, resultados por classe, tempos e checksums ficam em
search.json/diretório trials de cada fase.

<a id="4-complete-grouped-cv-results"></a>

## 4. Resultados completos da validação cruzada agrupada

Média e desvio-padrão amostral são entre folds; são diagnósticos adaptativos
de desenvolvimento, não estimativas imparciais de generalização por validação
aninhada de ponta a ponta. A regra escolhe menor C e número de atributos a
até um erro-padrão da melhor média, com os demais desempates predefinidos.
Não escolhe simplesmente a maior média observada.

<a id="phase-a"></a>

### Fase A

96 candidatos, 288 folds, 96 elegíveis, 0 tempos esgotados e 0 avisos de
convergência. Verificação independente de previsões/grupos/métricas aprovada.
Janela dos artefatos: 294.200 segundos, incluindo pausas; a fase A reutiliza
72 folds anteriores.

| Candidato | Família | C | Peso de classe | Perda | Peso char | Macro-F1 por fold | F1 médio | Desvio-padrão F1 |
|---|---|---:|---|---|---:|---|---:|---:|
| 569222475a46aa053b8b | word | 0.005 | None | squared_hinge | 1.0 | 0.376153, 0.406377, 0.385976 | 0.38950202 | 0.01541744 |
| 14d70e57c8136b7ea39f | word | 0.005 | balanced | squared_hinge | 1.0 | 0.412484, 0.416179, 0.397607 | 0.40875660 | 0.00983066 |
| b0e85fa3410ef4e4d49a | word | 0.05 | None | squared_hinge | 1.0 | 0.442167, 0.430561, 0.409142 | 0.42728989 | 0.01675376 |
| 7d4a1b9044e3e855c6dd | word | 0.05 | balanced | squared_hinge | 1.0 | 0.443228, 0.434680, 0.415044 | 0.43098436 | 0.01445088 |
| ca9b2f15be386692d34c | word | 0.5 | None | squared_hinge | 1.0 | 0.431956, 0.427834, 0.406828 | 0.42220607 | 0.01347625 |
| 57182fccccc538f4679d | word | 0.5 | balanced | squared_hinge | 1.0 | 0.435608, 0.427936, 0.416597 | 0.42671362 | 0.00956416 |
| 458bdcf522d8c74bceff | word | 0.005 | None | squared_hinge | 1.0 | 0.395697, 0.406929, 0.391719 | 0.39811492 | 0.00788771 |
| 22783f23775e60546b38 | word | 0.005 | balanced | squared_hinge | 1.0 | 0.416002, 0.415574, 0.392532 | 0.40803614 | 0.01342863 |
| f882a33d94b809d9af5a | word | 0.05 | None | squared_hinge | 1.0 | 0.437548, 0.437564, 0.419042 | 0.43138502 | 0.01068914 |
| f271debc151bd17189c6 | word | 0.05 | balanced | squared_hinge | 1.0 | 0.439545, 0.437954, 0.421379 | 0.43295942 | 0.01006021 |
| 5fbc135c1617278cfda4 | word | 0.5 | None | squared_hinge | 1.0 | 0.430115, 0.427081, 0.413937 | 0.42371118 | 0.00859936 |
| 9b1ff4962006802d3fdb | word | 0.5 | balanced | squared_hinge | 1.0 | 0.433767, 0.427791, 0.415246 | 0.42560116 | 0.00945261 |
| 0720ee835be805c4167c | word | 0.005 | None | squared_hinge | 1.0 | 0.377849, 0.403222, 0.383154 | 0.38807499 | 0.01338344 |
| d069efdaf3b46a8b4018 | word | 0.005 | balanced | squared_hinge | 1.0 | 0.417373, 0.416969, 0.402218 | 0.41218646 | 0.00863516 |
| 4abd0751ffbf7671be55 | word | 0.05 | None | squared_hinge | 1.0 | 0.440303, 0.428881, 0.411438 | 0.42687408 | 0.01453718 |
| fa9e5de5cb4f88fc8aec | word | 0.05 | balanced | squared_hinge | 1.0 | 0.443083, 0.432961, 0.415157 | 0.43040038 | 0.01413789 |
| 4fc7a04595a19127b738 | word | 0.5 | None | squared_hinge | 1.0 | 0.413324, 0.427057, 0.409916 | 0.41676579 | 0.00907392 |
| c5a90c105b1af07264a3 | word | 0.5 | balanced | squared_hinge | 1.0 | 0.430828, 0.427435, 0.409152 | 0.42247163 | 0.01165905 |
| 439a2b45d45fb6515f62 | word | 0.005 | None | squared_hinge | 1.0 | 0.169330, 0.175822, 0.170614 | 0.17192208 | 0.00343782 |
| 82d8a1920c2ac4718c3c | word | 0.005 | balanced | squared_hinge | 1.0 | 0.170147, 0.179897, 0.170614 | 0.17355276 | 0.00549919 |
| 945d110947c8bfedd885 | word | 0.05 | None | squared_hinge | 1.0 | 0.177827, 0.309662, 0.174093 | 0.22052751 | 0.07721514 |
| 252613eff664e198615e | word | 0.05 | balanced | squared_hinge | 1.0 | 0.318443, 0.342181, 0.321959 | 0.32752746 | 0.01281115 |
| 82f51b044190f671998a | word | 0.5 | None | squared_hinge | 1.0 | 0.383246, 0.409971, 0.390041 | 0.39441912 | 0.01388986 |
| c7261f81372623be12ae | word | 0.5 | balanced | squared_hinge | 1.0 | 0.429888, 0.423692, 0.406876 | 0.42015173 | 0.01190735 |
| a13cf661f4a466a0d020 | char | 0.005 | None | squared_hinge | 1.0 | 0.421098, 0.413719, 0.398514 | 0.41111046 | 0.01151548 |
| 8142ce02c98fb55ce770 | char | 0.005 | balanced | squared_hinge | 1.0 | 0.428015, 0.424083, 0.407355 | 0.41981745 | 0.01097041 |
| 0de4c3a42402d9cd121a | char | 0.05 | None | squared_hinge | 1.0 | 0.434478, 0.443549, 0.422706 | 0.43357765 | 0.01045086 |
| 0da8ae8a3f525195dcf6 | char | 0.05 | balanced | squared_hinge | 1.0 | 0.436190, 0.444528, 0.425579 | 0.43543202 | 0.00949727 |
| c8b99f0c4c299e1c547f | char | 0.5 | None | squared_hinge | 1.0 | 0.441361, 0.428118, 0.416112 | 0.42853009 | 0.01262928 |
| b423b277a87df5cb3a98 | char | 0.5 | balanced | squared_hinge | 1.0 | 0.443392, 0.427525, 0.416311 | 0.42907594 | 0.01360651 |
| 5dbf49f0ab643d28536d | char | 0.005 | None | squared_hinge | 1.0 | 0.422426, 0.412786, 0.402839 | 0.41268370 | 0.00979426 |
| bca84be877955e42666a | char | 0.005 | balanced | squared_hinge | 1.0 | 0.432117, 0.415300, 0.406224 | 0.41788053 | 0.01313804 |
| 2dcdd2fbf6b717ad3f0d | char | 0.05 | None | squared_hinge | 1.0 | 0.431683, 0.440793, 0.420861 | 0.43111221 | 0.00997811 |
| ba862acfa33ee076ffba | char | 0.05 | balanced | squared_hinge | 1.0 | 0.432494, 0.439422, 0.421373 | 0.43109648 | 0.00910546 |
| 1ec39b3db95407102275 | char | 0.5 | None | squared_hinge | 1.0 | 0.439056, 0.427229, 0.415353 | 0.42721269 | 0.01185125 |
| 9448893c1daaeac90c7a | char | 0.5 | balanced | squared_hinge | 1.0 | 0.439748, 0.427159, 0.416263 | 0.42772342 | 0.01175269 |
| 81b614f6b1ca2e5f376f | char | 0.005 | None | squared_hinge | 1.0 | 0.421858, 0.407167, 0.398061 | 0.40902858 | 0.01200690 |
| 7bf2db620cfa8e07882f | char | 0.005 | balanced | squared_hinge | 1.0 | 0.429339, 0.419202, 0.408591 | 0.41904406 | 0.01037526 |
| 3fc7e11c335f0385120c | char | 0.05 | None | squared_hinge | 1.0 | 0.439389, 0.433941, 0.417521 | 0.43028349 | 0.01138350 |
| 5a11986a35f3e6b406c0 | char | 0.05 | balanced | squared_hinge | 1.0 | 0.440368, 0.438524, 0.418027 | 0.43230621 | 0.01240039 |
| 81754127ed3d37885cf6 | char | 0.5 | None | squared_hinge | 1.0 | 0.437003, 0.423695, 0.417612 | 0.42610335 | 0.00991709 |
| d94bda97165b2fa787b4 | char | 0.5 | balanced | squared_hinge | 1.0 | 0.437084, 0.423841, 0.418500 | 0.42647478 | 0.00956782 |
| 98e574b07e291ce5bb61 | char | 0.005 | None | squared_hinge | 1.0 | 0.169330, 0.174624, 0.170220 | 0.17139130 | 0.00283448 |
| 49d46b1233bfa4d05676 | char | 0.005 | balanced | squared_hinge | 1.0 | 0.169330, 0.174624, 0.170220 | 0.17139130 | 0.00283448 |
| df80cbcd522d572bb932 | char | 0.05 | None | squared_hinge | 1.0 | 0.169330, 0.175367, 0.170220 | 0.17163902 | 0.00325892 |
| 7f217b6f161faa439f95 | char | 0.05 | balanced | squared_hinge | 1.0 | 0.169751, 0.302066, 0.170243 | 0.21401976 | 0.07625062 |
| e9fccc351662918bbe2c | char | 0.5 | None | squared_hinge | 1.0 | 0.276214, 0.320404, 0.292020 | 0.29621270 | 0.02239149 |
| 11a1cfbe63da30f8c085 | char | 0.5 | balanced | squared_hinge | 1.0 | 0.365812, 0.372727, 0.389748 | 0.37609554 | 0.01231859 |
| 92bf38c9886e2afecdf9 | char_wb | 0.005 | None | squared_hinge | 1.0 | 0.419600, 0.413968, 0.401608 | 0.41172514 | 0.00920343 |
| e666a5cdce91ace11b17 | char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.431475, 0.422799, 0.408769 | 0.42101454 | 0.01145747 |
| 65bca93fa765bedd5f1f | char_wb | 0.05 | None | squared_hinge | 1.0 | 0.433237, 0.437960, 0.427407 | 0.43286788 | 0.00528600 |
| 7c4918c651255e81a42d | char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.437973, 0.437254, 0.427113 | 0.43411312 | 0.00607288 |
| 4fdb1947bc499bd7f1b0 | char_wb | 0.5 | None | squared_hinge | 1.0 | 0.444237, 0.419766, 0.421342 | 0.42844830 | 0.01369583 |
| 81049401320c580e71ee | char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.442717, 0.421097, 0.421942 | 0.42858527 | 0.01224602 |
| 97e5af1fbdbcbeff2647 | char_wb | 0.005 | None | squared_hinge | 1.0 | 0.425391, 0.413528, 0.404482 | 0.41446726 | 0.01048641 |
| 41b2a2b99ec33852408c | char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.433771, 0.416504, 0.409313 | 0.41986282 | 0.01257052 |
| d228e4d34091717a0d9e | char_wb | 0.05 | None | squared_hinge | 1.0 | 0.431987, 0.442157, 0.424597 | 0.43291371 | 0.00881697 |
| dfdc401d26c8e70d3cdf | char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.432610, 0.439030, 0.425224 | 0.43228827 | 0.00690862 |
| ad341409b4540856f68c | char_wb | 0.5 | None | squared_hinge | 1.0 | 0.440242, 0.429179, 0.416988 | 0.42880284 | 0.01163146 |
| c52ded347dc817a0f40a | char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.439411, 0.432975, 0.417721 | 0.43003596 | 0.01113985 |
| a382f512559163005524 | char_wb | 0.005 | None | squared_hinge | 1.0 | 0.418894, 0.404356, 0.405361 | 0.40953697 | 0.00811899 |
| c7bf2a9d56148441c159 | char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.430271, 0.420380, 0.410883 | 0.42051128 | 0.00969486 |
| a6e6617bf017ac79a966 | char_wb | 0.05 | None | squared_hinge | 1.0 | 0.436081, 0.436291, 0.420423 | 0.43093145 | 0.00910150 |
| b19fa99d5934b5d6dfbb | char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.433690, 0.436986, 0.418341 | 0.42967233 | 0.00995078 |
| c1dd92b854ef6b8e9bab | char_wb | 0.5 | None | squared_hinge | 1.0 | 0.440678, 0.418564, 0.420418 | 0.42655303 | 0.01226744 |
| e367a9ca84d52bcd2f2c | char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.443395, 0.420105, 0.419120 | 0.42754026 | 0.01373983 |
| 13940e777be8473a7f4b | char_wb | 0.005 | None | squared_hinge | 1.0 | 0.169330, 0.174624, 0.170220 | 0.17139130 | 0.00283448 |
| 3f8053c3863296618090 | char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.169330, 0.174624, 0.170220 | 0.17139130 | 0.00283448 |
| ad39085fe8bd1222d73c | char_wb | 0.05 | None | squared_hinge | 1.0 | 0.169330, 0.179630, 0.170220 | 0.17306012 | 0.00570717 |
| 9a1b6491e195bc276fea | char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.179913, 0.311976, 0.173776 | 0.22188849 | 0.07807839 |
| 6e6caf39bb8ad20e5482 | char_wb | 0.5 | None | squared_hinge | 1.0 | 0.318019, 0.347987, 0.316723 | 0.32757653 | 0.01768799 |
| de8f91ab0cd12cdd1fb6 | char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.399911, 0.392649, 0.409560 | 0.40070696 | 0.00848357 |
| d027903589d330fcd861 | word_char_wb | 0.005 | None | squared_hinge | 1.0 | 0.426256, 0.422455, 0.406072 | 0.41826093 | 0.01072593 |
| ae3ed9da69e523c52765 | word_char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.435330, 0.426402, 0.404177 | 0.42196966 | 0.01604218 |
| 0c57545cf1442c5a2542 | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.441734, 0.437747, 0.426672 | 0.43538425 | 0.00780401 |
| e61c17eaa9d72ae9738e | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.440939, 0.439637, 0.426812 | 0.43579594 | 0.00780716 |
| 1e0d216d0d5b98a53e03 | word_char_wb | 0.5 | None | squared_hinge | 1.0 | 0.438644, 0.422350, 0.417068 | 0.42602044 | 0.01124685 |
| e9c02f2b49e059bebb60 | word_char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.438192, 0.422541, 0.419873 | 0.42686868 | 0.00989630 |
| 0efcee00319798827bc4 | word_char_wb | 0.005 | None | squared_hinge | 1.0 | 0.424803, 0.422019, 0.405506 | 0.41744244 | 0.01043057 |
| edc7160ba542e65f67ae | word_char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.434557, 0.426471, 0.408284 | 0.42310396 | 0.01345624 |
| c477c00195a188d191b2 | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.434995, 0.439730, 0.423412 | 0.43271204 | 0.00839517 |
| 8f851da4ab0c212a0ab7 | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.435577, 0.440492, 0.424977 | 0.43368197 | 0.00792948 |
| f97972c3097135487a7f | word_char_wb | 0.5 | None | squared_hinge | 1.0 | 0.434389, 0.422551, 0.411669 | 0.42286981 | 0.01136368 |
| d3ecb5d524615b2ea1e2 | word_char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.431153, 0.421803, 0.410923 | 0.42129314 | 0.01012432 |
| 0a053b698f5105e3d8eb | word_char_wb | 0.005 | None | squared_hinge | 1.0 | 0.427750, 0.419670, 0.407633 | 0.41835063 | 0.01012305 |
| 1c1a4f2e58cfd440780f | word_char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.431134, 0.430955, 0.412818 | 0.42496912 | 0.01052342 |
| 42c22ab6e0398e33b3ea | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.441846, 0.437433, 0.421570 | 0.43361641 | 0.01066337 |
| d064ea0be480087dc339 | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.440189, 0.435579, 0.433883 | 0.43655039 | 0.00326322 |
| 446cfe7b5cb7b88774b3 | word_char_wb | 0.5 | None | squared_hinge | 1.0 | 0.435066, 0.420221, 0.422290 | 0.42585913 | 0.00803997 |
| 2ced7a9887348b398d9a | word_char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.435352, 0.421777, 0.419680 | 0.42560317 | 0.00850781 |
| 56c9abb8b163337ff9bb | word_char_wb | 0.005 | None | squared_hinge | 1.0 | 0.169330, 0.175822, 0.170614 | 0.17192208 | 0.00343782 |
| 2528b0092184a9406552 | word_char_wb | 0.005 | balanced | squared_hinge | 1.0 | 0.170147, 0.193053, 0.170614 | 0.17793815 | 0.01309204 |
| 1da7d3af675faf9290ed | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.191731, 0.319844, 0.190068 | 0.23388103 | 0.07445099 |
| df7ed8b1f63807065146 | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.336155, 0.353462, 0.359482 | 0.34969934 | 0.01211015 |
| 37e76704cd381891d5bb | word_char_wb | 0.5 | None | squared_hinge | 1.0 | 0.394610, 0.414529, 0.402853 | 0.40399746 | 0.01000862 |
| 02b3f016d110229f6087 | word_char_wb | 0.5 | balanced | squared_hinge | 1.0 | 0.444591, 0.426854, 0.409772 | 0.42707233 | 0.01741028 |

<a id="phase-b"></a>

### Fase B

20 candidatos, 100 folds, 20 elegíveis, 0 tempos esgotados e 0 avisos de
convergência. Verificação independente de previsões/grupos/métricas aprovada.
Janela dos artefatos: 424.760 segundos, incluindo pausas; a fase A reutiliza
72 folds anteriores.

| Candidato | Família | C | Peso de classe | Perda | Peso char | Macro-F1 por fold | F1 médio | Desvio-padrão F1 |
|---|---|---:|---|---|---:|---|---:|---:|
| 0c57545cf1442c5a2542 | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.445960, 0.436149, 0.430126, 0.426353, 0.450426 | 0.43780286 | 0.01023024 |
| 8538323c4ffa5dc11fe0 | word_char_wb | 0.015811388 | None | squared_hinge | 1.0 | 0.438855, 0.424592, 0.428356, 0.428116, 0.448428 | 0.43366946 | 0.00982550 |
| 5c49441438381915fc8c | word_char_wb | 0.015811388 | balanced | squared_hinge | 1.0 | 0.436982, 0.426196, 0.426368, 0.427122, 0.451175 | 0.43356859 | 0.01083266 |
| 42c22ab6e0398e33b3ea | word_char_wb | 0.05 | None | squared_hinge | 1.0 | 0.440792, 0.439593, 0.436418, 0.429135, 0.445524 | 0.43829260 | 0.00607455 |
| d064ea0be480087dc339 | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.440219, 0.441545, 0.438730, 0.430386, 0.451834 | 0.44054291 | 0.00766558 |
| 0c9940a26751ca882be5 | word_char_wb | 0.15811388 | None | squared_hinge | 1.0 | 0.439826, 0.431919, 0.425772, 0.430901, 0.441537 | 0.43399108 | 0.00656485 |
| f884dddbb631d5aa6492 | word_char_wb | 0.15811388 | balanced | squared_hinge | 1.0 | 0.442900, 0.431041, 0.431260, 0.432183, 0.449289 | 0.43733463 | 0.00832059 |
| 06b2e85e30027c9c0e5f | word_char_wb | 0.05 | None | hinge | 1.0 | 0.428256, 0.419492, 0.417342, 0.417453, 0.441496 | 0.42480771 | 0.01034989 |
| 73032230d3d3e25f3072 | word_char_wb | 0.05 | balanced | hinge | 1.0 | 0.425833, 0.428382, 0.418903, 0.423316, 0.441376 | 0.42756196 | 0.00847774 |
| a709b8872709affa998e | word_char_wb | 0.05 | balanced | squared_hinge | 0.5 | 0.441139, 0.440875, 0.438661, 0.429312, 0.451484 | 0.44029408 | 0.00789839 |
| f53c39423436a164a6d7 | word_char_wb | 0.05 | balanced | squared_hinge | 2.0 | 0.439239, 0.431975, 0.430167, 0.434442, 0.441084 | 0.43538148 | 0.00466573 |
| 2fe3df156068a57f027f | word_char_wb | 0.015811388 | None | squared_hinge | 1.0 | 0.440277, 0.436607, 0.423168, 0.423175, 0.447056 | 0.43405665 | 0.01062011 |
| 33726fa36fd5a4ad747c | word_char_wb | 0.015811388 | balanced | squared_hinge | 1.0 | 0.441802, 0.434512, 0.431021, 0.429231, 0.449596 | 0.43723216 | 0.00842257 |
| e61c17eaa9d72ae9738e | word_char_wb | 0.05 | balanced | squared_hinge | 1.0 | 0.445923, 0.442382, 0.432046, 0.426192, 0.449695 | 0.43924774 | 0.00982157 |
| 6d853e37e7f630bdb52b | word_char_wb | 0.15811388 | None | squared_hinge | 1.0 | 0.439477, 0.431908, 0.428389, 0.422984, 0.445584 | 0.43366844 | 0.00895614 |
| 964a853224ea90a16159 | word_char_wb | 0.15811388 | balanced | squared_hinge | 1.0 | 0.437895, 0.433316, 0.426935, 0.424042, 0.449617 | 0.43436113 | 0.01009884 |
| 6445e09f1ef80c3f3985 | word_char_wb | 0.05 | None | hinge | 1.0 | 0.419066, 0.417258, 0.419294, 0.419385, 0.445212 | 0.42404309 | 0.01186575 |
| 8ef2070271a73bebb784 | word_char_wb | 0.05 | balanced | hinge | 1.0 | 0.418796, 0.431335, 0.418113, 0.420572, 0.448570 | 0.42747737 | 0.01294706 |
| 6a4dde97c3ec03d11e51 | word_char_wb | 0.05 | balanced | squared_hinge | 0.5 | 0.441475, 0.437688, 0.434042, 0.436990, 0.456524 | 0.44134407 | 0.00888941 |
| d7a3508e1ecd64af38ae | word_char_wb | 0.05 | balanced | squared_hinge | 2.0 | 0.437919, 0.431218, 0.429189, 0.423231, 0.451978 | 0.43470694 | 0.01098763 |

<a id="5-selected-optimized-svm"></a>

## 5. SVM otimizado selecionado

```json
{
  "candidate_id": "42c22ab6e0398e33b3ea",
  "candidate": {
    "family": "word_char_wb",
    "vectors": {
      "word": {
        "analyzer": "word",
        "ngram_range": [
          1,
          2
        ],
        "min_df": 5,
        "max_df": 0.98,
        "sublinear_tf": false,
        "binary": true,
        "norm": "l2",
        "max_features": 60000
      },
      "char": {
        "analyzer": "char_wb",
        "ngram_range": [
          3,
          6
        ],
        "min_df": 3,
        "max_df": 0.95,
        "sublinear_tf": false,
        "binary": true,
        "norm": "l2",
        "max_features": 50000
      }
    },
    "char_weight": 1.0,
    "C": 0.05,
    "class_weight": null,
    "loss": "squared_hinge"
  },
  "mean_f1_macro": 0.43829260052390184,
  "std_f1_macro": 0.006074550169856932,
  "best_mean_f1_macro": 0.4413440666266143,
  "one_se_threshold": 0.43736860122766746,
  "criterion": "Within one SE of best mean macro-F1: smallest C, fewest features, squared hinge, no class weighting, mean F1, stable ID"
}
```

É a escolha de um erro-padrão da fase B. Uma média maior de outro candidato
não substitui a regra. Referência e selecionado da fase B usam os mesmos
cinco folds; a fase A tem outra divisão e não deve ser tratada como resultado
pareado da fase B.

A referência antiga reproduzida na meta-validação é:

| Configuração | Acurácia | Macro-F1 | Perda logarítmica | Brier | ECE (10 intervalos) |
|---|---:|---:|---:|---:|---:|
| old_native_svm | 0.46501014 | 0.46260019 | — | — | — |
| old_calibrated_svm | 0.46551724 | 0.46579698 | — | — | — |
| A_original_75_25 | 0.47515213 | 0.47533872 | — | — | — |

Ajuste final: 16093 linhas originais, 108327 atributos, 13 iterações, ajuste
21.459s, extração reservada 4.522s. Pesos/checkpoints históricos não foram
sobrescritos.

<a id="6-calibration"></a>

## 6. Calibração

Cinco folds externos agrupados estimam a qualidade da seleção do método nas
2027 linhas. Cada treino externo usa três folds internos para escolher
isotônica só quando a melhora média pareada de perda logarítmica frente à
sigmoide supera seu erro-padrão. Caso contrário, prefere sigmoide. O método
final usa a mesma regra de três folds em todas as linhas e depois é ajustado
nas 2027. Nenhum SVM-base é reajustado na calibração.

Método final: **sigmoide**. Escolhas externas: `{'sigmoid': 5}`.
OOF da política e diagnósticos OOF de método fixo são distintos:

| Configuração | Acurácia | Macro-F1 | Perda logarítmica | Brier | ECE (10 intervalos) |
|---|---:|---:|---:|---:|---:|
| sigmoid | 0.44055254 | 0.44002824 | 1.04728532 | 0.63220790 | 0.01903280 |
| isotonic | 0.43660582 | 0.43209464 | 1.15608727 | 0.63800783 | 0.01297631 |
| selection_policy | 0.44055254 | 0.44002824 | 1.04728532 | 0.63220790 | 0.01903280 |

A comparação não calibrada usa softmax das margens como substituto diagnóstico;
não é interface nativa de probabilidade do SVM. Brier é a média da soma dos
três erros quadráticos das probabilidades; ECE usa dez intervalos de confiança.
Intervalos de confiabilidade completos, perdas internas, composição dos folds
e parâmetros ficam em calibration.json.

<a id="7-soft-voting-and-conservative-development-recommendation"></a>

## 7. Votação suave e recomendação conservadora no desenvolvimento

A = SVM calibrado antigo .75 + NorBERTo calibrado inalterado .25. B substitui
só SVM e mantém .75/.25. C ajusta apenas esses dois pesos. Pesos candidatos
SVM: 0/.25/.5/.75/1. Um seletor agrupado aninhado cinco por três estima a
política de pesos nos grupos externos reservados. C final usa toda a
meta-validação e a mesma regra interna de um erro-padrão, preferindo proximidade a .75.

Peso SVM final de C: **0.75**. Pesos externos: `{'0.75': 5}`; coincide em 5/5 escolhas.

| Configuração | Acurácia | Macro-F1 | Perda logarítmica | Brier | ECE (10 intervalos) |
|---|---:|---:|---:|---:|---:|
| A | 0.47515213 | 0.47533872 | 1.03799024 | 0.62717157 | 0.03286970 |
| B | 0.47565923 | 0.47550303 | 1.03744302 | 0.62676388 | 0.03465168 |
| C_fixed_diagnostic | 0.47565923 | 0.47550303 | 1.03744302 | 0.62676388 | 0.03465168 |
| C_policy_oof | 0.47565923 | 0.47550303 | 1.03744302 | 0.62676388 | 0.03465168 |
| svm_original_calibrated | 0.46551724 | 0.46579698 | 1.03934613 | 0.62818121 | 0.02286285 |
| svm_optimized_calibrated | 0.46855984 | 0.46811481 | 1.03880824 | 0.62771249 | 0.02624389 |
| norberto_unchanged | 0.45283976 | 0.45183970 | 1.05750913 | 0.63857233 | 0.02540882 |

C_fixed_diagnostic usa peso escolhido nos mesmos rótulos e é adaptativo.
C_policy_oof usa peso escolhido sem seus próprios grupos externos de avaliação
e é a versão dos critérios de adoção. Nenhum é estimativa de teste final independente.

B só substitui A se o limite inferior de macro-F1 do bootstrap pareado por
grupos for positivo, ao menos quatro dos cinco folds tiverem diferenças
positivas e nenhuma classe perder mais que .02 em F1. C só substitui B com
as mesmas verificações OOF da política e peso final igual a pelo menos três
escolhas externas. B precisa passar contra A primeiro.

Verificações B sobre A: `{'positive_bootstrap_lower_bound': False, 'positive_in_at_least_four_folds': False, 'no_class_f1_loss_above_002': True}`.
Política C sobre B: `{'positive_bootstrap_lower_bound': False, 'positive_in_at_least_four_folds': False, 'no_class_f1_loss_above_002': True}`;
peso estável: `True`.

**Recomendação congelada antes de Teste1: A.** É decisão conservadora de
desenvolvimento, não garantia de generalização ou promoção automática à produção.

<a id="8-class-behavior-and-complementarity-on-meta-validation"></a>

## 8. Comportamento por classe e complementaridade na meta-validação

### A

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.48076923 | 0.48231511 | 0.48154093 | 622 |
| c234 | 0.44479005 | 0.40452617 | 0.42370370 | 707 |
| c5 | 0.49787234 | 0.54587869 | 0.52077151 | 643 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[300, 183, 139], [206, 286, 215], [118, 174, 351]]
```

### B

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.48960000 | 0.49196141 | 0.49077787 | 622 |
| c234 | 0.44677419 | 0.39179632 | 0.41748304 | 707 |
| c5 | 0.48830812 | 0.55209953 | 0.51824818 | 643 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[306, 170, 146], [204, 277, 226], [115, 173, 355]]
```

### C_fixed_diagnostic

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.48960000 | 0.49196141 | 0.49077787 | 622 |
| c234 | 0.44677419 | 0.39179632 | 0.41748304 | 707 |
| c5 | 0.48830812 | 0.55209953 | 0.51824818 | 643 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[306, 170, 146], [204, 277, 226], [115, 173, 355]]
```

### C_policy_oof

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.48960000 | 0.49196141 | 0.49077787 | 622 |
| c234 | 0.44677419 | 0.39179632 | 0.41748304 | 707 |
| c5 | 0.48830812 | 0.55209953 | 0.51824818 | 643 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[306, 170, 146], [204, 277, 226], [115, 173, 355]]
```

```json
{
  "original_svm_vs_norberto": {
    "both_correct": 658,
    "left_only_correct": 260,
    "right_only_correct": 235,
    "both_wrong": 819,
    "disagreements": 662,
    "per_class": {
      "c1": {
        "rows": 622,
        "left_only_correct": 100,
        "right_only_correct": 49
      },
      "c234": {
        "rows": 707,
        "left_only_correct": 86,
        "right_only_correct": 111
      },
      "c5": {
        "rows": 643,
        "left_only_correct": 74,
        "right_only_correct": 75
      }
    }
  },
  "optimized_svm_vs_norberto": {
    "both_correct": 659,
    "left_only_correct": 265,
    "right_only_correct": 234,
    "both_wrong": 814,
    "disagreements": 674,
    "per_class": {
      "c1": {
        "rows": 622,
        "left_only_correct": 98,
        "right_only_correct": 44
      },
      "c234": {
        "rows": 707,
        "left_only_correct": 82,
        "right_only_correct": 116
      },
      "c5": {
        "rows": 643,
        "left_only_correct": 85,
        "right_only_correct": 74
      }
    }
  }
}
```

<a id="9-paired-retrospective-test1-comparison"></a>

## 9. Comparação pareada retrospectiva em Teste1

Previsões A/B/C e individuais foram persistidas e verificadas por checksum
antes de ler os 900 rótulos. A reproduz exatamente probabilidades históricas,
IDs de linhas/textos, rótulos e macro-F1. As 900 linhas mantêm a ordem original;
não houve atualização retrospectiva de parâmetros nem mudança de recomendação.

| Configuração | Acurácia | Macro-F1 | Perda logarítmica | Brier | ECE (10 intervalos) |
|---|---:|---:|---:|---:|---:|
| A | 0.49666667 | 0.49584422 | 1.01823015 | 0.61206996 | 0.05745328 |
| B | 0.49555556 | 0.49518732 | 1.01841868 | 0.61235488 | 0.05763384 |
| C | 0.49555556 | 0.49518732 | 1.01841868 | 0.61235488 | 0.05763384 |
| svm_original_calibrated | 0.48888889 | 0.48770052 | 1.01956990 | 0.61310139 | 0.04522395 |
| svm_optimized_calibrated | 0.48777778 | 0.48692514 | 1.01989947 | 0.61355952 | 0.05204923 |
| norberto_unchanged | 0.48000000 | 0.47864644 | 1.03424436 | 0.62203431 | 0.03280910 |
| svm_original_native | 0.49444444 | 0.49059493 | — | — | — |
| svm_optimized_native | 0.48222222 | 0.47781071 | — | — | — |

### A

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.52851711 | 0.46333333 | 0.49378330 | 300 |
| c234 | 0.44262295 | 0.45000000 | 0.44628099 | 300 |
| c5 | 0.52108434 | 0.57666667 | 0.54746835 | 300 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[139, 93, 68], [74, 135, 91], [50, 77, 173]]
```

### B

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.53076923 | 0.46000000 | 0.49285714 | 300 |
| c234 | 0.44267516 | 0.46333333 | 0.45276873 | 300 |
| c5 | 0.51840491 | 0.56333333 | 0.53993610 | 300 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[138, 94, 68], [72, 139, 89], [50, 81, 169]]
```

### C

| Classe | Precisão | Revocação | F1 | Total |
|---|---:|---:|---:|---:|
| c1 | 0.53076923 | 0.46000000 | 0.49285714 | 300 |
| c234 | 0.44267516 | 0.46333333 | 0.45276873 | 300 |
| c5 | 0.51840491 | 0.56333333 | 0.53993610 | 300 |

Matriz de confusão: linhas verdadeiras / colunas preditas, c1/c234/c5:

```json
[[138, 94, 68], [72, 139, 89], [50, 81, 169]]
```

| Comparação (direita menos esquerda) | Corrigidos | Prejudicados | Diferença macro-F1 | IC 95% bootstrap de grupos | p McNemar descritivo |
|---|---:|---:|---:|---|---:|
| B_minus_A | 21 | 22 | -0.00065689 | [-0.015465774113581398, 0.013377443169504386] | 1.00000000 |
| C_minus_A | 21 | 22 | -0.00065689 | [-0.015465774113581398, 0.013377443169504386] | 1.00000000 |
| C_minus_B | 0 | 0 | 0.00000000 | [0.0, 0.0] | 1.00000000 |

Intervalos absolutos de acurácia/F1 de cada modelo, todos os intervalos pareados
de acurácia, correções/prejuízos por classe e confiabilidade completa estão
em test1/metrics.json.

Complementaridade final:

```json
{
  "original_svm_vs_norberto": {
    "both_correct": 318,
    "left_only_correct": 122,
    "right_only_correct": 114,
    "both_wrong": 346,
    "disagreements": 308,
    "per_class": {
      "c1": {
        "rows": 300,
        "left_only_correct": 53,
        "right_only_correct": 29
      },
      "c234": {
        "rows": 300,
        "left_only_correct": 29,
        "right_only_correct": 58
      },
      "c5": {
        "rows": 300,
        "left_only_correct": 40,
        "right_only_correct": 27
      }
    }
  },
  "optimized_svm_vs_norberto": {
    "both_correct": 318,
    "left_only_correct": 121,
    "right_only_correct": 114,
    "both_wrong": 347,
    "disagreements": 311,
    "per_class": {
      "c1": {
        "rows": 300,
        "left_only_correct": 52,
        "right_only_correct": 27
      },
      "c234": {
        "rows": 300,
        "left_only_correct": 30,
        "right_only_correct": 58
      },
      "c5": {
        "rows": 300,
        "left_only_correct": 39,
        "right_only_correct": 29
      }
    }
  }
}
```

<a id="10-statistical-and-methodological-limitations"></a>

## 10. Limitações estatísticas e metodológicas

O bootstrap percentil pareado usa 2000 reamostragens, semente 42, de componentes
históricos completos na meta-validação e grupos normalizados em Teste1.
Os mesmos pesos de grupos são aplicados a todos os modelos. Os intervalos
condicionam nas previsões ajustadas; não incluem toda a incerteza de treino,
calibração e seleção adaptativa. O corpus pequeno reutilizado e buscas
sucessivas limitam generalização. McNemar descritivo avalia acertos pareados,
não macro-F1, e ignora dependência residual dos grupos; múltiplas comparações
retrospectivas não são testes confirmatórios novos.

O conjunto reservado antigo e Teste1 têm exposição histórica. Duplicatas
exatas/normalizadas e agrupamento não garantem independência semântica ou
anotação consistente. Todas as linhas oficiais são mantidas, sem filtrar/
corrigir pelos rótulos. A investigação deixa Teste2 fora e não satisfaz a
ordem original Teste2 antes de Teste1. Gerar uma entrega depois não torna
avaliações anteriores retrospectivamente cegas.

<a id="11-final-recommendation"></a>

## 11. Recomendação final

**Manter A: SVM original .75 / NorBERTo inalterado .25.** Nas 1972 linhas de
meta-validação, B ganha só 0.0164 pontos percentuais de macro-F1, melhora em
dois de cinco folds e tem intervalo pareado de 95% que inclui zero. Falha
nos critérios predefinidos de substituição. A seleção aninhada escolhe .75
nos cinco folds externos e no ajuste final; C não acrescenta benefício por pesos.

A decisão foi congelada antes de Teste1, cuja comparação reproduz exatamente
cada ID, probabilidade e rótulo históricos de A. A acerta 447/900, macro-F1
49.5844%; B/C, 446/900 e 49.5187%. SVM novo isolado também não melhora no final:
macro-F1 calibrado 48.6925%, contra 48.7701% do antigo; macro-F1 das margens
nativas 47.7811%, contra 49.0595%. São comparações descritivas, sem justificar
nova busca. O intervalo pareado inclui zero; não se afirma deterioração ou
superioridade estatisticamente significativa.

As 900 linhas oficiais são mantidas. A auditoria independente encontra 88
correspondências exatas e 105 normalizadas com os três papéis originais:
treino 74/85, calibração 9/11 e meta-validação 5/9 (exatas/normalizadas).
Teste1 tem 877 grupos normalizados. Essa sobreposição reforça a limitação
retrospectiva; não foi usada para filtrar ou mudar decisões.

A verificação independente reconstruiu métricas de busca, composição/receita
original, papéis aninhados de calibração/pesos, métricas das oito previsões
finais e contagens pareadas. Outra implementação de matrizes ponderadas por
linha reproduziu os 92 intervalos reportados de desenvolvimento/final com os
mesmos 2000 sorteios pareados de grupos. Identidades e ordem dos arquivos
confirmam congelamento antes da predição, persistência de todas as previsões
antes do manifesto e avaliação posterior. Evidência:
`verification-development.json`, `verification-test1.json`,
`verification-confidence.json` e `closeout-audit.json` na finalização; hashes
e resultados ficam no agregado.

O ajuste final usou só as 16093 linhas originais, levou 21.459 segundos sem
avisos de convergência e avaliou os 3999 textos reservados em 4.522 segundos.
Checkpoints, probabilidades, saídas históricas e padrão ativo originais não
mudaram. Teste2 nunca foi acessado. Nenhum GPT, stacking, RAG ou treino de
outros modelos-base foi acrescentado. A investigação termina com resultado
negativo documentado.
