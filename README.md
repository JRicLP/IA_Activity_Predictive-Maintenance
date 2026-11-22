# Sentinela

Aplicação de apoio à inspeção industrial baseada no conjunto de dados **AI4I
2020**. O projeto treina um modelo Random Forest para estimar a probabilidade
de falha de uma máquina e disponibiliza os resultados por uma API FastAPI e
uma interface web. Também oferece explicações locais e globais com Shapash e
relatórios técnicos estruturados com Gemini API.

## Estrutura

```text
.
├── ai4i2020.csv              # Dados usados no treinamento
├── backend/
│   ├── app.py                # API FastAPI e arquivos estáticos
│   ├── predict.py            # Carga e execução do modelo
│   ├── preprocessing.py       # Engenharia de atributos
│   ├── train_export.py       # Treinamento e exportação dos artefatos
│   ├── explain.py            # Explicações Shapash local/global
│   ├── evidence.py           # Evidências estruturadas para relatórios
│   ├── gemini_reports.py     # Integração segura com Gemini
│   ├── tests/                # Testes da API, explicações e relatórios
│   ├── requirements.txt
│   ├── models/
│   │   ├── random_forest.joblib
│   │   ├── metadata.json
│   │   └── metricas.json
│   └── README.md
├── front/
│   └── static/               # Interface web
├── notebooks/
│   ├── EDA_AI4I_2020.ipynb
│   ├── ProjetoIA_atual.ipynb
│   └── Shapash_Explainability.ipynb
├── docs/
│   └── EXPLICABILIDADE_E_RELATORIOS.md
└── .env.example              # Exemplo, sem chave real
```

## Requisitos

- Python **3.13.9** validado neste projeto.
- Ambiente virtual `.venv`.
- Dependências fixadas em `backend/requirements.txt`.
- Chave Gemini opcional para geração de relatórios.

O score do Random Forest não é uma probabilidade calibrada de quebra futura.
As contribuições Shapash descrevem influência na predição e não causalidade.

## Instalação

Na raiz do projeto, crie e ative o ambiente virtual com Python 3.13:

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### Linux/macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Use o arquivo de dependências em `backend/requirements.txt`.
Verifique a instalação com:

```bash
python -m pip check
```

## Configuração do Gemini

Copie `.env.example` para `.env` e configure somente no backend:

```dotenv
GEMINI_API_KEY=COLOQUE_SUA_CHAVE_AQUI
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite
GEMINI_REPORTS_ENABLED=true
GEMINI_TIMEOUT_SECONDS=30
```

Não adicione a chave ao frontend, notebooks ou Git. Sem chave, a previsão,
as métricas e a explicação Shapash continuam disponíveis; apenas a geração
de relatório retorna HTTP 503.

## Treinamento

Execute o treinamento dentro de `backend`:

```bash
cd backend
python train_export.py --dataset ..\ai4i2020.csv
```

O comando gera `random_forest.joblib`, `metadata.json` e `metricas.json`
dentro de `backend/models/`.

## Execução

Na pasta `backend`, inicie o servidor:

```bash
cd backend
uvicorn app:app --reload
```

Na raiz do projeto, também é possível iniciar com o executável do ambiente:

```bash
cd backend
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Acesse pelo navegador:

- Interface: http://127.0.0.1:8000
- Documentação da API: http://127.0.0.1:8000/docs
- Métricas: http://127.0.0.1:8000/metricas

## API

- `GET /health`: estado da API e versão do modelo.
- `GET /metricas`: métricas da divisão de validação.
- `POST /predict`: calcula o alerta e os indicadores derivados.
- `POST /explain`: recalcula a previsão e retorna contribuições locais e
  importância global da classe positiva `1`.
- `POST /dashboard/start`: inicia o dashboard interativo Shapash na porta `8850`.
- `POST /reports/generate`: gera um relatório Gemini a partir de evidências
  JSON calculadas no servidor.

O endpoint `POST /predict` recebe `type`, `air_temperature_k`,
`process_temperature_k`, `rotational_speed_rpm`, `torque_nm` e
`tool_wear_min` e retorna o score de falha, o alerta e os atributos derivados.

Exemplo:

```json
{
  "type": "L",
  "air_temperature_k": 301.9,
  "process_temperature_k": 310.4,
  "rotational_speed_rpm": 1286,
  "torque_nm": 62.8,
  "tool_wear_min": 22
}
```

## Explicabilidade e relatórios

A interface possui os botões **Explicar previsão** e
**Gerar relatório com Gemini**. Os dois são acionados explicitamente pelo
usuário e utilizam o mesmo conteúdo da última previsão.

Na versão de apresentação, o acesso ao painel de IA fica oculto por padrão.
Para reativá-lo, altere `AGENTE_HABILITADO` para `true` em
`front/static/index.html`; as integrações e rotas de explicabilidade e
relatórios permanecem disponíveis no código.

O notebook [Shapash_Explainability.ipynb](notebooks/Shapash_Explainability.ipynb)
executa a explicação reproduzível separadamente da aplicação web. Consulte
[EXPLICABILIDADE_E_RELATORIOS.md](docs/EXPLICABILIDADE_E_RELATORIOS.md) para
arquitetura, limitações e contratos.

## Testes

Na raiz do projeto:

```bash
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Os testes usam mocks para o Gemini e não realizam chamadas reais à API.
