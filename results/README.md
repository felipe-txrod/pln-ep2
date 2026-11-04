# Resultados

Cada execução usa um diretório próprio com configuração e hashes. `dry_run` não
contém previsões nem métricas GPT. Execuções reais só ficam `completed` após
classificar todas as linhas; o cache permite retomar falhas e interrupções.
Saídas de desenvolvimento, teste2 e comparação teste1 devem ficar separadas.
Os resultados gerados e caches não são versionados por padrão: preserve os
diretórios completos e o snapshot do ambiente para reproduzir o experimento.
