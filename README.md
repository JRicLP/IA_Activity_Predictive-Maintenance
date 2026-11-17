# Sentinela

Aplicação de apoio à inspeção industrial baseada no conjunto de dados **AI4I
2020**. O projeto treina um modelo Random Forest para estimar a probabilidade
de falha de uma máquina e disponibiliza os resultados por uma API FastAPI e
uma interface web.

## Estrutura

```text
.
├── front/
│   ├── ai4i2020.csv          # Dados usados no treinamento
│   ├── treino.py             # Treinamento e geração dos artefatos
│   ├── main.py               # API FastAPI
│   ├── features.py           # Engenharia de atributos
│   ├── modelo_rf.joblib      # Modelo treinado
│   ├── metricas.json         # Métricas e dados exibidos pela interface
│   ├── requirements.txt
│   └── static/               # Interface web
└── notebooks/                # Análises exploratórias e referências
```

## Instalação

Na raiz do projeto, crie e ative o ambiente virtual:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-Location .\front
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
cd front
pip install -r requirements.txt
```

Use o mesmo ambiente virtual para treinar o modelo e executar a API.

## Treinamento

Execute o treinamento dentro de `front`:

```bash
python treino.py
```

O comando lê `ai4i2020.csv` e atualiza `modelo_rf.joblib` e `metricas.json`.
Esses arquivos são carregados pela API em tempo de inicialização.

## Execução

Ainda dentro de `front`, inicie o servidor:

```bash
uvicorn main:app --reload
```

Depois, acesse:

- Interface: http://127.0.0.1:8000
- Documentação da API: http://127.0.0.1:8000/docs
- Métricas: http://127.0.0.1:8000/api/metricas

O endpoint `POST /api/predict` recebe `type`, `air_temp`, `process_temp`,
`rpm`, `torque` e `tool_wear` e retorna o score de falha, o alerta e os
atributos derivados usados na previsão.
