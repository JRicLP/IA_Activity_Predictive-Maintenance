# Explicabilidade e relatórios

## Arquitetura

A API FastAPI reutiliza o `Predictor` e `build_features` já existentes. O
Shapash calcula explicações locais da classe positiva 1 e uma importância
global reproduzível. O botão **Explicar previsão** inicia o dashboard
Shapash em `http://127.0.0.1:8850` e abre-o em uma nova aba.

## Executar

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Endpoints novos:

- `POST /explain`
- `POST /dashboard/start`
- `POST /reports/generate`
- `GET /health`

As duas rotas de explicação recebem o mesmo JSON de `/predict`.

## Configuração Gemini

Copie `.env.example` para `.env` e configure:

```dotenv
GEMINI_API_KEY=COLOQUE_SUA_CHAVE_AQUI
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite
GEMINI_REPORTS_ENABLED=true
GEMINI_TIMEOUT_SECONDS=30
GEMINI_MAX_RETRIES=3
GEMINI_RETRY_DELAY_SECONDS=1
```

A chave nunca é enviada ao frontend. Sem chave ou com relatórios desativados,
`POST /reports/generate` retorna HTTP 503. `/health`, `/predict`, `/metricas`
e `/explain` continuam disponíveis. Falhas temporárias 408/5xx são repetidas
até três vezes com espera exponencial. HTTP 429 retorna imediatamente uma
mensagem de limite gratuito, sem transformar o erro em retry infinito.

## Evidências

`build_evidence` gera um objeto Pydantic versionado com medições, features
derivadas, score, alerta, contribuições, métricas da validação e limites
metodológicos. O Gemini recebe somente esse JSON, não HTML nem captura de tela.

## Limitações

- O score não é uma probabilidade calibrada.
- Contribuições SHAP representam associação com a predição.
- Features derivadas e brutas são correlacionadas.
- O relatório não substitui avaliação técnica humana.
- A amostra global é reconstruída dos 6.000 registros de treino da divisão
  original 60/20/20.
