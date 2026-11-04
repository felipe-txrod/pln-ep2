# Orientações para agentes

## Ao iniciar uma sessão

Leia [.agents/CONTEXT.md](.agents/CONTEXT.md) para recuperar decisões, resultados
e próximos passos. Consulte o README para comandos e `docs/` para detalhes;
confira o estado real dos arquivos e do Git antes de continuar uma execução.
O repositório remoto contém o conteúdo de `EP2/` como sua raiz.

## Regras do projeto

- Converse e documente em português. Use Conventional Commits com descrição
  em português e commits coerentes com as alterações realizadas.
- O enunciado em `docs/ep2-enunciado.md` é a fonte principal de requisitos;
  registre decisões em conflito com pedidos anteriores. Contexto histórico
  não substitui a solicitação atual do usuário.
- Somente `resp_text` pode entrar como alvo no GPT. Nunca envie seu rótulo
  verdadeiro nem outras colunas da planilha. Rótulos de exemplos do treino
  podem compor demonstrações. Preserve a separação `Text`/`Example`.
- Escolha prompts, recuperação e parâmetros em validação agrupada do treino.
  Teste1 (900 linhas) serve à comparação final com EP1; teste2 (897) é a entrega.
  Não ajuste o método com seus gabaritos, não imponha cotas de previsões e
  não corrija ou remova linhas da avaliação.
- Preserve JSON validado, IDs/ordem, cache por pedido, limites explícitos,
  rastreabilidade e retomada. Falha não pode virar rótulo substituto.
- Não promova configurações experimentais nem amplie rodadas remotas
  automaticamente. Respeite o orçamento e o escopo solicitados; o usuário
  prioriza reduzir tempo e tokens. Alterações de documentação não exigem
  chamadas de classificação.
- Preserve resultados e caches existentes. Mudanças de código/ambiente podem
  invalidar fingerprints: use outro diretório em vez de apagar artefatos.
- Execute testes pertinentes às alterações. `.venv/bin/python -m pytest -q`
  usa transportes/executáveis simulados; não equivale a uma avaliação real.
- Não versione credenciais, `.env`, ambiente virtual, pesos, caches ou
  resultados brutos ignorados. Agregados relevantes podem ficar em `docs/`.
  Não altere o escopo do Git para incluir o diretório pai ou EP1.

## Manutenção do contexto

Atualize `.agents/CONTEXT.md` ao concluir mudanças relevantes, experimentos ou
decisões. Registre configuração, partição, métricas, consumo, caminho dos
artefatos e limitações. Distinga resultado medido, hipótese e trabalho pendente.
Mantenha o resumo compacto e aponte para a documentação detalhada.
