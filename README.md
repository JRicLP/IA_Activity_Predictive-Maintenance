# Sentinela

Aplicação de apoio à inspeção industrial baseada no conjunto de dados **AI4I
2020**. O projeto treina um modelo Random Forest para estimar a probabilidade
de falha de uma máquina e disponibiliza os resultados por uma API FastAPI e
uma interface web.

## Estrutura

```text
.
├── ai4i2020.csv              # Dados usados no treinamento
├── backend/
│   ├── app.py                # API FastAPI e arquivos estáticos
│   ├── predict.py            # Carga e execução do modelo
│   ├── preprocessing.py       # Engenharia de atributos
│   ├── train_export.py       # Treinamento e exportação dos artefatos
│   ├── requirements.txt
│   ├── models/
│   │   ├── random_forest.joblib
│   │   ├── metadata.json
│   │   └── metricas.json
│   └── README.md
├── front/
│   └── static/               # Interface web
├── notebooks/                # Análises exploratórias e referências
└── ProjetoIA_atual.ipynb     # Notebook atual
```

## Instalação

Na raiz do projeto, crie e ative o ambiente virtual:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Use o arquivo de dependências em `backend/requirements.txt`.

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
uvicorn app:app --reload
```

Acesse:

- Interface: http://127.0.0.1:8000
- Documentação da API: http://127.0.0.1:8000/docs
- Métricas: http://127.0.0.1:8000/metricas

O endpoint `POST /predict` recebe `type`, `air_temperature_k`,
`process_temperature_k`, `rotational_speed_rpm`, `torque_nm` e
`tool_wear_min` e retorna o score de falha, o alerta e os atributos derivados.
